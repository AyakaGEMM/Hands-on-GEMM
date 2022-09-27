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
constexpr size_t BLOCK_SIZE = 16; // we assume that every block has equal blockDim.x and blockDim.y
constexpr size_t BLOCK_M = 128;   // These const values decide how many thing a thread compute and the amount of shared memory to allocate.
constexpr size_t BLOCK_N = 128;
constexpr size_t BLOCK_K = 8; // don't set 64 here, it will cause bank conflict and lower occupancy.
constexpr size_t BLOCK_M_COMPUTE = BLOCK_M / BLOCK_SIZE;
constexpr size_t BLOCK_N_COMPUTE = BLOCK_N / BLOCK_SIZE;

constexpr int shared_memory_A = BLOCK_M * BLOCK_K;
constexpr int shared_memory_B = BLOCK_N * BLOCK_K;
constexpr int shared_memory_element = shared_memory_A + shared_memory_B;
constexpr int shared_memory_size = shared_memory_element * sizeof(float); // shared memory to use.
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

    const float *baseA = A + baseX * K;
    const float *baseB = B + baseY;

    int colA = baseIdx >> 1, colB = baseIdx >> 5, rowA = (baseIdx & 1) << 2, rowB = (baseIdx << 2) & 127;
    int warpId = baseIdx >> 5, warpBaseId = baseIdx & 31;
    int colC = ((warpId >> 1 << 2) + (warpBaseId & 3)) << 3, rowC = (((warpId & 1) << 3) + (warpBaseId >> 2)) << 3;
    float *baseC = C + (baseX + colC) * N + baseY + rowC;

    __shared__ float subA[2][BLOCK_M * BLOCK_K];
    __shared__ float subB[2][BLOCK_N * BLOCK_K];

    float4 regB[BLOCK_M_COMPUTE / 4]; // hopefully, these should reside in register.
    float4 regA[BLOCK_M_COMPUTE / 4];

    float4 preA, preB;

    size_t compute_stage_idx = 0;

    preA = *reinterpret_cast<const float4 *>(baseA + colA * K + rowA);
    *reinterpret_cast<float4 *>(&subB[0][baseIdx * 4]) = *reinterpret_cast<const float4 *>(baseB + colB * N + rowB);
    subA[0][colA + rowA * BLOCK_M] = preA.x;
    subA[0][colA + (rowA + 1) * BLOCK_M] = preA.y;
    subA[0][colA + (rowA + 2) * BLOCK_M] = preA.z;
    subA[0][colA + (rowA + 3) * BLOCK_M] = preA.w;

    __syncthreads();

    for (int i = BLOCK_K; i < K; i += BLOCK_K)
    {
        preA = *reinterpret_cast<const float4 *>(baseA + i + colA * K + rowA);
        preB = *reinterpret_cast<const float4 *>(baseB + i * N + colB * N + rowB);

#pragma unroll
        for (int ii = 0; ii < BLOCK_K; ii++)
        {
            regA[0] = *reinterpret_cast<float4 *>(&subA[compute_stage_idx][colC + ii * BLOCK_M]);
            regA[1] = *reinterpret_cast<float4 *>(&subA[compute_stage_idx][(colC + 4) + ii * BLOCK_M]);

            regB[0] = *reinterpret_cast<float4 *>(&subB[compute_stage_idx][rowC + BLOCK_N * ii]);
            regB[1] = *reinterpret_cast<float4 *>(&subB[compute_stage_idx][rowC + 4 + BLOCK_N * ii]);

#pragma unroll
            for (int cpi = 0; cpi < BLOCK_M_COMPUTE / 4; cpi++)
            {
#pragma unroll
                for (int cpj = 0; cpj < BLOCK_N_COMPUTE / 4; cpj++)
                {
                    c[cpi * 4 * BLOCK_M_COMPUTE + cpj * 4] += regA[cpi].x * regB[cpj].x;
                    c[cpi * 4 * BLOCK_M_COMPUTE + cpj * 4 + 1] += regA[cpi].x * regB[cpj].y;
                    c[cpi * 4 * BLOCK_M_COMPUTE + cpj * 4 + 2] += regA[cpi].x * regB[cpj].z;
                    c[cpi * 4 * BLOCK_M_COMPUTE + cpj * 4 + 3] += regA[cpi].x * regB[cpj].w;

                    c[(cpi * 4 + 1) * BLOCK_M_COMPUTE + cpj * 4] += regA[cpi].y * regB[cpj].x;
                    c[(cpi * 4 + 1) * BLOCK_M_COMPUTE + cpj * 4 + 1] += regA[cpi].y * regB[cpj].y;
                    c[(cpi * 4 + 1) * BLOCK_M_COMPUTE + cpj * 4 + 2] += regA[cpi].y * regB[cpj].z;
                    c[(cpi * 4 + 1) * BLOCK_M_COMPUTE + cpj * 4 + 3] += regA[cpi].y * regB[cpj].w;

                    c[(cpi * 4 + 2) * BLOCK_M_COMPUTE + cpj * 4] += regA[cpi].z * regB[cpj].x;
                    c[(cpi * 4 + 2) * BLOCK_M_COMPUTE + cpj * 4 + 1] += regA[cpi].z * regB[cpj].y;
                    c[(cpi * 4 + 2) * BLOCK_M_COMPUTE + cpj * 4 + 2] += regA[cpi].z * regB[cpj].z;
                    c[(cpi * 4 + 2) * BLOCK_M_COMPUTE + cpj * 4 + 3] += regA[cpi].z * regB[cpj].w;

                    c[(cpi * 4 + 3) * BLOCK_M_COMPUTE + cpj * 4] += regA[cpi].w * regB[cpj].x;
                    c[(cpi * 4 + 3) * BLOCK_M_COMPUTE + cpj * 4 + 1] += regA[cpi].w * regB[cpj].y;
                    c[(cpi * 4 + 3) * BLOCK_M_COMPUTE + cpj * 4 + 2] += regA[cpi].w * regB[cpj].z;
                    c[(cpi * 4 + 3) * BLOCK_M_COMPUTE + cpj * 4 + 3] += regA[cpi].w * regB[cpj].w;
                }
            }
        }
        compute_stage_idx ^= 1;
        *reinterpret_cast<float4 *>(&subB[compute_stage_idx][baseIdx * 4]) = preB;
        subA[compute_stage_idx][colA + rowA * BLOCK_M] = preA.x;
        subA[compute_stage_idx][colA + (rowA + 1) * BLOCK_M] = preA.y;
        subA[compute_stage_idx][colA + (rowA + 2) * BLOCK_M] = preA.z;
        subA[compute_stage_idx][colA + (rowA + 3) * BLOCK_M] = preA.w;
        __syncthreads();
    }

#pragma unroll
    for (int ii = 0; ii < BLOCK_K; ii++)
    {
        regA[0] = *reinterpret_cast<float4 *>(&subA[compute_stage_idx][colC + ii * BLOCK_M]);
        regA[1] = *reinterpret_cast<float4 *>(&subA[compute_stage_idx][(colC + 4) + ii * BLOCK_M]);

        regB[0] = *reinterpret_cast<float4 *>(&subB[compute_stage_idx][rowC + BLOCK_N * ii]);
        regB[1] = *reinterpret_cast<float4 *>(&subB[compute_stage_idx][rowC + 4 + BLOCK_N * ii]);

#pragma unroll
        for (int cpi = 0; cpi < BLOCK_M_COMPUTE / 4; cpi++)
        {
#pragma unroll
            for (int cpj = 0; cpj < BLOCK_N_COMPUTE / 4; cpj++)
            {
                c[cpi * 4 * BLOCK_M_COMPUTE + cpj * 4] += regA[cpi].x * regB[cpj].x;
                c[cpi * 4 * BLOCK_M_COMPUTE + cpj * 4 + 1] += regA[cpi].x * regB[cpj].y;
                c[cpi * 4 * BLOCK_M_COMPUTE + cpj * 4 + 2] += regA[cpi].x * regB[cpj].z;
                c[cpi * 4 * BLOCK_M_COMPUTE + cpj * 4 + 3] += regA[cpi].x * regB[cpj].w;

                c[(cpi * 4 + 1) * BLOCK_M_COMPUTE + cpj * 4] += regA[cpi].y * regB[cpj].x;
                c[(cpi * 4 + 1) * BLOCK_M_COMPUTE + cpj * 4 + 1] += regA[cpi].y * regB[cpj].y;
                c[(cpi * 4 + 1) * BLOCK_M_COMPUTE + cpj * 4 + 2] += regA[cpi].y * regB[cpj].z;
                c[(cpi * 4 + 1) * BLOCK_M_COMPUTE + cpj * 4 + 3] += regA[cpi].y * regB[cpj].w;

                c[(cpi * 4 + 2) * BLOCK_M_COMPUTE + cpj * 4] += regA[cpi].z * regB[cpj].x;
                c[(cpi * 4 + 2) * BLOCK_M_COMPUTE + cpj * 4 + 1] += regA[cpi].z * regB[cpj].y;
                c[(cpi * 4 + 2) * BLOCK_M_COMPUTE + cpj * 4 + 2] += regA[cpi].z * regB[cpj].z;
                c[(cpi * 4 + 2) * BLOCK_M_COMPUTE + cpj * 4 + 3] += regA[cpi].z * regB[cpj].w;

                c[(cpi * 4 + 3) * BLOCK_M_COMPUTE + cpj * 4] += regA[cpi].w * regB[cpj].x;
                c[(cpi * 4 + 3) * BLOCK_M_COMPUTE + cpj * 4 + 1] += regA[cpi].w * regB[cpj].y;
                c[(cpi * 4 + 3) * BLOCK_M_COMPUTE + cpj * 4 + 2] += regA[cpi].w * regB[cpj].z;
                c[(cpi * 4 + 3) * BLOCK_M_COMPUTE + cpj * 4 + 3] += regA[cpi].w * regB[cpj].w;
            }
        }
    }

    for (int i = 0; i < BLOCK_M_COMPUTE; i++)
        for (int j = 0; j < BLOCK_N_COMPUTE; j += 4)
        {
            preA = *reinterpret_cast<float4 *>(&baseC[i * N + j]);
            c[i * BLOCK_M_COMPUTE + j] = c[i * BLOCK_M_COMPUTE + j] * alpha + preA.x * beta;
            c[i * BLOCK_M_COMPUTE + j + 1] = c[i * BLOCK_M_COMPUTE + j + 1] * alpha + preA.y * beta;
            c[i * BLOCK_M_COMPUTE + j + 2] = c[i * BLOCK_M_COMPUTE + j + 2] * alpha + preA.z * beta;
            c[i * BLOCK_M_COMPUTE + j + 3] = c[i * BLOCK_M_COMPUTE + j + 3] * alpha + preA.w * beta;
            *reinterpret_cast<float4 *>(&baseC[i * N + j]) = *reinterpret_cast<float4 *>(&c[i * BLOCK_M_COMPUTE + j]);
        }
}

void sgemm(int M, int N, int K, float *a, float *b, float *c, float alpha = 1, float beta = 0)
{
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((M + BLOCK_M - 1) / BLOCK_M, (N + BLOCK_N - 1) / BLOCK_N);
#ifdef __CUDACC__ // workaround for stupid vscode intellisense
    matrixMul<<<numBlocks, threadsPerBlock>>>(a, b, c, M, N, K, alpha, beta);
#endif
}
