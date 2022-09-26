#include <cstdlib>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>

#ifndef __CUDACC__
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
void __syncthreads(); // workaround __syncthreads warning
#endif

#include <iostream>
const size_t BLOCK_SIZE = 16; // we assume that every block has equal blockDim.x and blockDim.y
const size_t BLOCK_M = 128;   // These const values decide how many thing a thread compute and the amount of shared memory to allocate.
const size_t BLOCK_N = 128;
const size_t BLOCK_K = 8; // don't set 64 here, it will cause bank conflict and lower occupancy.
const size_t BLOCK_M_COMPUTE = BLOCK_M / BLOCK_SIZE;
const size_t BLOCK_N_COMPUTE = BLOCK_N / BLOCK_SIZE;

const int shared_memory_A = BLOCK_M * BLOCK_K;
const int shared_memory_B = BLOCK_N * BLOCK_K;
const int shared_memory_element = shared_memory_A + shared_memory_B;
const int shared_memory_size = shared_memory_element * sizeof(float); // shared memory to use.
#define colM(a, i, j, lda) a[((j) * (lda)) + (i)]
#define rowM(a, i, j, lda) a[(j) + (i) * (lda)]

constexpr __forceinline__ __device__ auto convertColIdx(int idx, const float *begin, int subM, int subN, int N)
{
    int m = idx / subM, n = idx % subM;
    return begin + m + n * N;
}

constexpr __forceinline__ __device__ auto convertRowIdx(int idx, const float *begin, int subM, int subN, int N)
{
    int m = idx / subN, n = idx % subN;
    return begin + m * N + n;
}

__global__ void matrixMul(const float *A, const float *B, float *C,
                          int M, int N, int K, float alpha, float beta)
{
    const size_t baseX = blockIdx.x * blockDim.x * BLOCK_M_COMPUTE;
    const size_t baseY = blockIdx.y * blockDim.y * BLOCK_N_COMPUTE;

    const int moveNum = shared_memory_element / (BLOCK_SIZE * BLOCK_SIZE) / 2;
    const size_t baseIdx = threadIdx.y * blockDim.y + threadIdx.x;

    constexpr size_t threadsNum = BLOCK_SIZE * BLOCK_SIZE;

    float c[BLOCK_M_COMPUTE * BLOCK_N_COMPUTE] = {};
    float resC[BLOCK_M_COMPUTE * BLOCK_N_COMPUTE] = {};

    const float *baseA = A + baseX * K;
    const float *baseB = B + baseY;

    int colA = baseIdx / 2, colB = baseIdx / (BLOCK_N / 4), rowA = (baseIdx & 1) * 4, rowB = (baseIdx * 4) % BLOCK_N;
    int warpId = baseIdx / 32, warpBaseId = baseIdx % 32;
    int colC = (warpId / 2 * 4 + warpBaseId % 4) * BLOCK_M_COMPUTE, rowC = ((warpId % 2) * 8 + warpBaseId / 4) * BLOCK_N_COMPUTE;
    float *baseC = C + (baseX + colC) * N + baseY + rowC;

    __shared__ float subA[2][BLOCK_M * BLOCK_K];
    __shared__ float subB[2][BLOCK_N * BLOCK_K];

    float4 regB[2][BLOCK_M_COMPUTE / 4]; // hopefully, these should reside in register.
    float4 regA[2][BLOCK_M_COMPUTE / 4];

    size_t copy_stage_idx = 0;
    size_t compute_stage_idx = 0;

    regA[0][0] = *reinterpret_cast<const float4 *>(baseA + colA * K + rowA);
    *reinterpret_cast<float4 *>(&subB[copy_stage_idx][baseIdx * 4]) = *reinterpret_cast<const float4 *>(baseB + colB * N + rowB);
    subA[0][colA + rowA * BLOCK_M] = regA[0][0].x;
    subA[0][colA + (rowA + 1) * BLOCK_M] = regA[0][0].y;
    subA[0][colA + (rowA + 2) * BLOCK_M] = regA[0][0].z;
    subA[0][colA + (rowA + 3) * BLOCK_M] = regA[0][0].w;
    copy_stage_idx ^= 1;

    __syncthreads();

    for (int i = BLOCK_K; i < K; i += BLOCK_K)
    {
        regA[copy_stage_idx][0] = *reinterpret_cast<const float4 *>(baseA + i + colA * K + rowA);
        *reinterpret_cast<float4 *>(&subB[copy_stage_idx][baseIdx * 4]) = *reinterpret_cast<const float4 *>(baseB + i * N + colB * N + rowB);
        subA[copy_stage_idx][colA + rowA * BLOCK_M] = regA[copy_stage_idx][0].x;
        subA[copy_stage_idx][colA + (rowA + 1) * BLOCK_M] = regA[copy_stage_idx][0].y;
        subA[copy_stage_idx][colA + (rowA + 2) * BLOCK_M] = regA[copy_stage_idx][0].z;
        subA[copy_stage_idx][colA + (rowA + 3) * BLOCK_M] = regA[copy_stage_idx][0].w;
        copy_stage_idx ^= 1;

#pragma unroll
        for (int ii = 0; ii < BLOCK_K; ii++)
        {
            regA[compute_stage_idx][0] = *reinterpret_cast<float4 *>(&subA[compute_stage_idx][colC + ii * BLOCK_M]);
            regA[compute_stage_idx][1] = *reinterpret_cast<float4 *>(&subA[compute_stage_idx][(colC + 4) + ii * BLOCK_M]);

            regB[compute_stage_idx][0] = *reinterpret_cast<float4 *>(&subB[compute_stage_idx][rowC + BLOCK_N * ii]);
            regB[compute_stage_idx][1] = *reinterpret_cast<float4 *>(&subB[compute_stage_idx][rowC + 4 + BLOCK_N * ii]);

#pragma unroll
            for (int cpi = 0; cpi < BLOCK_M_COMPUTE / 4; cpi++)
            {
#pragma unroll
                for (int cpj = 0; cpj < BLOCK_N_COMPUTE / 4; cpj++)
                {
                    c[cpi * 4 * BLOCK_M_COMPUTE + cpj * 4] += regA[compute_stage_idx][cpi].x * regB[compute_stage_idx][cpj].x;
                    c[cpi * 4 * BLOCK_M_COMPUTE + cpj * 4 + 1] += regA[compute_stage_idx][cpi].x * regB[compute_stage_idx][cpj].y;
                    c[cpi * 4 * BLOCK_M_COMPUTE + cpj * 4 + 2] += regA[compute_stage_idx][cpi].x * regB[compute_stage_idx][cpj].z;
                    c[cpi * 4 * BLOCK_M_COMPUTE + cpj * 4 + 3] += regA[compute_stage_idx][cpi].x * regB[compute_stage_idx][cpj].w;

                    c[(cpi * 4 + 1) * BLOCK_M_COMPUTE + cpj * 4] += regA[compute_stage_idx][cpi].y * regB[compute_stage_idx][cpj].x;
                    c[(cpi * 4 + 1) * BLOCK_M_COMPUTE + cpj * 4 + 1] += regA[compute_stage_idx][cpi].y * regB[compute_stage_idx][cpj].y;
                    c[(cpi * 4 + 1) * BLOCK_M_COMPUTE + cpj * 4 + 2] += regA[compute_stage_idx][cpi].y * regB[compute_stage_idx][cpj].z;
                    c[(cpi * 4 + 1) * BLOCK_M_COMPUTE + cpj * 4 + 3] += regA[compute_stage_idx][cpi].y * regB[compute_stage_idx][cpj].w;

                    c[(cpi * 4 + 2) * BLOCK_M_COMPUTE + cpj * 4] += regA[compute_stage_idx][cpi].z * regB[compute_stage_idx][cpj].x;
                    c[(cpi * 4 + 2) * BLOCK_M_COMPUTE + cpj * 4 + 1] += regA[compute_stage_idx][cpi].z * regB[compute_stage_idx][cpj].y;
                    c[(cpi * 4 + 2) * BLOCK_M_COMPUTE + cpj * 4 + 2] += regA[compute_stage_idx][cpi].z * regB[compute_stage_idx][cpj].z;
                    c[(cpi * 4 + 2) * BLOCK_M_COMPUTE + cpj * 4 + 3] += regA[compute_stage_idx][cpi].z * regB[compute_stage_idx][cpj].w;

                    c[(cpi * 4 + 3) * BLOCK_M_COMPUTE + cpj * 4] += regA[compute_stage_idx][cpi].w * regB[compute_stage_idx][cpj].x;
                    c[(cpi * 4 + 3) * BLOCK_M_COMPUTE + cpj * 4 + 1] += regA[compute_stage_idx][cpi].w * regB[compute_stage_idx][cpj].y;
                    c[(cpi * 4 + 3) * BLOCK_M_COMPUTE + cpj * 4 + 2] += regA[compute_stage_idx][cpi].w * regB[compute_stage_idx][cpj].z;
                    c[(cpi * 4 + 3) * BLOCK_M_COMPUTE + cpj * 4 + 3] += regA[compute_stage_idx][cpi].w * regB[compute_stage_idx][cpj].w;
                }
            }
        }
        compute_stage_idx ^= 1;
        __syncthreads();
    }

#pragma unroll
    for (int ii = 0; ii < BLOCK_K; ii++)
    {
        regA[compute_stage_idx][0] = *reinterpret_cast<float4 *>(&subA[compute_stage_idx][colC + ii * BLOCK_M]);
        regA[compute_stage_idx][1] = *reinterpret_cast<float4 *>(&subA[compute_stage_idx][(colC + 4) + ii * BLOCK_M]);

        regB[compute_stage_idx][0] = *reinterpret_cast<float4 *>(&subB[compute_stage_idx][rowC + BLOCK_N * ii]);
        regB[compute_stage_idx][1] = *reinterpret_cast<float4 *>(&subB[compute_stage_idx][rowC + 4 + BLOCK_N * ii]);

#pragma unroll
        for (int cpi = 0; cpi < BLOCK_M_COMPUTE / 4; cpi++)
        {
#pragma unroll
            for (int cpj = 0; cpj < BLOCK_N_COMPUTE / 4; cpj++)
            {
                c[cpi * 4 * BLOCK_M_COMPUTE + cpj * 4] += regA[compute_stage_idx][cpi].x * regB[compute_stage_idx][cpj].x;
                c[cpi * 4 * BLOCK_M_COMPUTE + cpj * 4 + 1] += regA[compute_stage_idx][cpi].x * regB[compute_stage_idx][cpj].y;
                c[cpi * 4 * BLOCK_M_COMPUTE + cpj * 4 + 2] += regA[compute_stage_idx][cpi].x * regB[compute_stage_idx][cpj].z;
                c[cpi * 4 * BLOCK_M_COMPUTE + cpj * 4 + 3] += regA[compute_stage_idx][cpi].x * regB[compute_stage_idx][cpj].w;

                c[(cpi * 4 + 1) * BLOCK_M_COMPUTE + cpj * 4] += regA[compute_stage_idx][cpi].y * regB[compute_stage_idx][cpj].x;
                c[(cpi * 4 + 1) * BLOCK_M_COMPUTE + cpj * 4 + 1] += regA[compute_stage_idx][cpi].y * regB[compute_stage_idx][cpj].y;
                c[(cpi * 4 + 1) * BLOCK_M_COMPUTE + cpj * 4 + 2] += regA[compute_stage_idx][cpi].y * regB[compute_stage_idx][cpj].z;
                c[(cpi * 4 + 1) * BLOCK_M_COMPUTE + cpj * 4 + 3] += regA[compute_stage_idx][cpi].y * regB[compute_stage_idx][cpj].w;

                c[(cpi * 4 + 2) * BLOCK_M_COMPUTE + cpj * 4] += regA[compute_stage_idx][cpi].z * regB[compute_stage_idx][cpj].x;
                c[(cpi * 4 + 2) * BLOCK_M_COMPUTE + cpj * 4 + 1] += regA[compute_stage_idx][cpi].z * regB[compute_stage_idx][cpj].y;
                c[(cpi * 4 + 2) * BLOCK_M_COMPUTE + cpj * 4 + 2] += regA[compute_stage_idx][cpi].z * regB[compute_stage_idx][cpj].z;
                c[(cpi * 4 + 2) * BLOCK_M_COMPUTE + cpj * 4 + 3] += regA[compute_stage_idx][cpi].z * regB[compute_stage_idx][cpj].w;

                c[(cpi * 4 + 3) * BLOCK_M_COMPUTE + cpj * 4] += regA[compute_stage_idx][cpi].w * regB[compute_stage_idx][cpj].x;
                c[(cpi * 4 + 3) * BLOCK_M_COMPUTE + cpj * 4 + 1] += regA[compute_stage_idx][cpi].w * regB[compute_stage_idx][cpj].y;
                c[(cpi * 4 + 3) * BLOCK_M_COMPUTE + cpj * 4 + 2] += regA[compute_stage_idx][cpi].w * regB[compute_stage_idx][cpj].z;
                c[(cpi * 4 + 3) * BLOCK_M_COMPUTE + cpj * 4 + 3] += regA[compute_stage_idx][cpi].w * regB[compute_stage_idx][cpj].w;
            }
        }
    }

    for (int i = 0; i < BLOCK_M_COMPUTE; i++)
        for (int j = 0; j < BLOCK_N_COMPUTE; j++)
            C[(baseX + colC + i) * N + baseY + rowC + j] = beta * C[(baseX + colC + i) * N + baseY + rowC + j] + alpha * c[i * BLOCK_M_COMPUTE + j];
}

void sgemm(int M, int N, int K, float *a, float *b, float *c, float alpha = 1, float beta = 0)
{
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((M + BLOCK_M - 1) / BLOCK_M, (N + BLOCK_N - 1) / BLOCK_N);
#ifdef __CUDACC__ // workaround for stupid vscode intellisense
    matrixMul<<<numBlocks, threadsPerBlock>>>(a, b, c, M, N, K, alpha, beta);
#endif
}
