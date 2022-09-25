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
    const int baseX = blockIdx.x * blockDim.x * BLOCK_M_COMPUTE;
    const int baseY = blockIdx.y * blockDim.y * BLOCK_N_COMPUTE;

    const int moveNum = shared_memory_element / (BLOCK_SIZE * BLOCK_SIZE) / 2;
    const size_t baseIdx = threadIdx.y * blockDim.y + threadIdx.x;

    float c[BLOCK_M_COMPUTE * BLOCK_N_COMPUTE] = {};

    constexpr size_t stage_count = 2;

    __shared__ float subA[stage_count][BLOCK_M * BLOCK_K];
    __shared__ float subB[stage_count][BLOCK_N * BLOCK_K];

    float4 regB[stage_count][BLOCK_N_COMPUTE / 4]; // hopefully, these should reside in register.
    float4 regA[stage_count];
    float4 regCopy[stage_count];

    size_t copy_stage_idx = 0;
    size_t compute_stage_idx = 0;

#pragma unroll
    for (int idx = 0; idx < moveNum; idx++)
    {
        subA[0][baseIdx + idx * BLOCK_SIZE * BLOCK_SIZE] = *convertColIdx(baseIdx + idx * BLOCK_SIZE * BLOCK_SIZE, A + baseX * K, BLOCK_M, BLOCK_K, K);
    }
#pragma unroll
    for (int idx = 0; idx < moveNum; idx += 4)
    {
        *reinterpret_cast<float4 *>(&subB[0][baseIdx * 4]) = *(reinterpret_cast<const float4 *>(convertRowIdx(baseIdx * 4 + idx * BLOCK_SIZE * BLOCK_SIZE * 4, B + baseY, BLOCK_K, BLOCK_N, N)));
    }

    copy_stage_idx ^= 1;

    __syncthreads();

    for (int i = BLOCK_K; i < K; i += BLOCK_K)
    {
#pragma unroll
        for (int idx = 0; idx < moveNum; idx++)
        {
            regCopy[copy_stage_idx] = *(reinterpret_cast<const float4 *>(convertRowIdx(baseIdx * 4, A + baseX * K + i, BLOCK_M, BLOCK_K, K)));
            const auto m = (baseIdx * 4) / BLOCK_K, n = (baseIdx * 4) % BLOCK_K;
            subA[copy_stage_idx][m + n * BLOCK_M] = regCopy[copy_stage_idx].x;
            subA[copy_stage_idx][m + (n + 1) * BLOCK_M] = regCopy[copy_stage_idx].y;
            subA[copy_stage_idx][m + (n + 2) * BLOCK_M] = regCopy[copy_stage_idx].z;
            subA[copy_stage_idx][m + (n + 3) * BLOCK_M] = regCopy[copy_stage_idx].w;
        }
#pragma unroll
        for (int idx = 0; idx < moveNum; idx += 4)
        {
            *reinterpret_cast<float4 *>(&subB[copy_stage_idx][baseIdx * 4]) = *(reinterpret_cast<const float4 *>(convertRowIdx(baseIdx * 4, B + baseY + i * N, BLOCK_K, BLOCK_N, N)));
        }

        copy_stage_idx ^= 1;

#pragma unroll
        for (int ii = 0; ii < BLOCK_K; ii++)
        {
#pragma unroll
            for (int cpj = 0; cpj < BLOCK_N_COMPUTE / 4; cpj++)
            {
                regB[compute_stage_idx][cpj] = *reinterpret_cast<float4 *>(&subB[compute_stage_idx][threadIdx.y * BLOCK_N_COMPUTE + cpj * 4 + BLOCK_N * ii]);
            }
#pragma unroll
            for (int cpi = 0; cpi < BLOCK_M_COMPUTE / 4; cpi++)
            {
                regA[compute_stage_idx] = *reinterpret_cast<float4 *>(&subA[compute_stage_idx][(threadIdx.x * BLOCK_M_COMPUTE + cpi * 4) + ii * BLOCK_M]);
#pragma unroll
                for (int cpj = 0; cpj < BLOCK_N_COMPUTE / 4; cpj++)
                {
                    c[cpi * 4 * BLOCK_M_COMPUTE + cpj * 4] += regA[compute_stage_idx].x * regB[compute_stage_idx][cpj].x;
                    c[cpi * 4 * BLOCK_M_COMPUTE + cpj * 4 + 1] += regA[compute_stage_idx].x * regB[compute_stage_idx][cpj].y;
                    c[cpi * 4 * BLOCK_M_COMPUTE + cpj * 4 + 2] += regA[compute_stage_idx].x * regB[compute_stage_idx][cpj].z;
                    c[cpi * 4 * BLOCK_M_COMPUTE + cpj * 4 + 3] += regA[compute_stage_idx].x * regB[compute_stage_idx][cpj].w;

                    c[(cpi * 4 + 1) * BLOCK_M_COMPUTE + cpj * 4] += regA[compute_stage_idx].y * regB[compute_stage_idx][cpj].x;
                    c[(cpi * 4 + 1) * BLOCK_M_COMPUTE + cpj * 4 + 1] += regA[compute_stage_idx].y * regB[compute_stage_idx][cpj].y;
                    c[(cpi * 4 + 1) * BLOCK_M_COMPUTE + cpj * 4 + 2] += regA[compute_stage_idx].y * regB[compute_stage_idx][cpj].z;
                    c[(cpi * 4 + 1) * BLOCK_M_COMPUTE + cpj * 4 + 3] += regA[compute_stage_idx].y * regB[compute_stage_idx][cpj].w;

                    c[(cpi * 4 + 2) * BLOCK_M_COMPUTE + cpj * 4] += regA[compute_stage_idx].z * regB[compute_stage_idx][cpj].x;
                    c[(cpi * 4 + 2) * BLOCK_M_COMPUTE + cpj * 4 + 1] += regA[compute_stage_idx].z * regB[compute_stage_idx][cpj].y;
                    c[(cpi * 4 + 2) * BLOCK_M_COMPUTE + cpj * 4 + 2] += regA[compute_stage_idx].z * regB[compute_stage_idx][cpj].z;
                    c[(cpi * 4 + 2) * BLOCK_M_COMPUTE + cpj * 4 + 3] += regA[compute_stage_idx].z * regB[compute_stage_idx][cpj].w;

                    c[(cpi * 4 + 3) * BLOCK_M_COMPUTE + cpj * 4] += regA[compute_stage_idx].w * regB[compute_stage_idx][cpj].x;
                    c[(cpi * 4 + 3) * BLOCK_M_COMPUTE + cpj * 4 + 1] += regA[compute_stage_idx].w * regB[compute_stage_idx][cpj].y;
                    c[(cpi * 4 + 3) * BLOCK_M_COMPUTE + cpj * 4 + 2] += regA[compute_stage_idx].w * regB[compute_stage_idx][cpj].z;
                    c[(cpi * 4 + 3) * BLOCK_M_COMPUTE + cpj * 4 + 3] += regA[compute_stage_idx].w * regB[compute_stage_idx][cpj].w;
                }
            }
        }
        compute_stage_idx ^= 1;
        __syncthreads();
    }

#pragma unroll
    for (int ii = 0; ii < BLOCK_K; ii++)
    {
#pragma unroll
        for (int cpj = 0; cpj < BLOCK_N_COMPUTE / 4; cpj++)
        {
            regB[compute_stage_idx][cpj] = *reinterpret_cast<float4 *>(&subB[compute_stage_idx][threadIdx.y * BLOCK_N_COMPUTE + cpj * 4 + BLOCK_N * ii]);
        }
#pragma unroll
        for (int cpi = 0; cpi < BLOCK_M_COMPUTE / 4; cpi++)
        {
            regA[compute_stage_idx] = *reinterpret_cast<float4 *>(&subA[compute_stage_idx][(threadIdx.x * BLOCK_M_COMPUTE + cpi * 4) + ii * BLOCK_M]);
#pragma unroll
            for (int cpj = 0; cpj < BLOCK_N_COMPUTE / 4; cpj++)
            {
                c[cpi * 4 * BLOCK_M_COMPUTE + cpj * 4] += regA[compute_stage_idx].x * regB[compute_stage_idx][cpj].x;
                c[cpi * 4 * BLOCK_M_COMPUTE + cpj * 4 + 1] += regA[compute_stage_idx].x * regB[compute_stage_idx][cpj].y;
                c[cpi * 4 * BLOCK_M_COMPUTE + cpj * 4 + 2] += regA[compute_stage_idx].x * regB[compute_stage_idx][cpj].z;
                c[cpi * 4 * BLOCK_M_COMPUTE + cpj * 4 + 3] += regA[compute_stage_idx].x * regB[compute_stage_idx][cpj].w;

                c[(cpi * 4 + 1) * BLOCK_M_COMPUTE + cpj * 4] += regA[compute_stage_idx].y * regB[compute_stage_idx][cpj].x;
                c[(cpi * 4 + 1) * BLOCK_M_COMPUTE + cpj * 4 + 1] += regA[compute_stage_idx].y * regB[compute_stage_idx][cpj].y;
                c[(cpi * 4 + 1) * BLOCK_M_COMPUTE + cpj * 4 + 2] += regA[compute_stage_idx].y * regB[compute_stage_idx][cpj].z;
                c[(cpi * 4 + 1) * BLOCK_M_COMPUTE + cpj * 4 + 3] += regA[compute_stage_idx].y * regB[compute_stage_idx][cpj].w;

                c[(cpi * 4 + 2) * BLOCK_M_COMPUTE + cpj * 4] += regA[compute_stage_idx].z * regB[compute_stage_idx][cpj].x;
                c[(cpi * 4 + 2) * BLOCK_M_COMPUTE + cpj * 4 + 1] += regA[compute_stage_idx].z * regB[compute_stage_idx][cpj].y;
                c[(cpi * 4 + 2) * BLOCK_M_COMPUTE + cpj * 4 + 2] += regA[compute_stage_idx].z * regB[compute_stage_idx][cpj].z;
                c[(cpi * 4 + 2) * BLOCK_M_COMPUTE + cpj * 4 + 3] += regA[compute_stage_idx].z * regB[compute_stage_idx][cpj].w;

                c[(cpi * 4 + 3) * BLOCK_M_COMPUTE + cpj * 4] += regA[compute_stage_idx].w * regB[compute_stage_idx][cpj].x;
                c[(cpi * 4 + 3) * BLOCK_M_COMPUTE + cpj * 4 + 1] += regA[compute_stage_idx].w * regB[compute_stage_idx][cpj].y;
                c[(cpi * 4 + 3) * BLOCK_M_COMPUTE + cpj * 4 + 2] += regA[compute_stage_idx].w * regB[compute_stage_idx][cpj].z;
                c[(cpi * 4 + 3) * BLOCK_M_COMPUTE + cpj * 4 + 3] += regA[compute_stage_idx].w * regB[compute_stage_idx][cpj].w;
            }
        }
    }

    for (int i = 0; i < BLOCK_M_COMPUTE; i++)
        for (int j = 0; j < BLOCK_N_COMPUTE; j++)
            C[(baseX + threadIdx.x * BLOCK_M_COMPUTE + i) * N + baseY + threadIdx.y * BLOCK_N_COMPUTE + j] = beta * C[(baseX + threadIdx.x * BLOCK_M_COMPUTE + i) * N + baseY + threadIdx.y * BLOCK_N_COMPUTE + j] + alpha * c[i * BLOCK_M_COMPUTE + j];
}

void sgemm(int M, int N, int K, float *a, float *b, float *c, float alpha = 1, float beta = 0)
{
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((M + BLOCK_M - 1) / BLOCK_M, (N + BLOCK_N - 1) / BLOCK_N);
#ifdef __CUDACC__ // workaround for stupid vscode intellisense
    matrixMul<<<numBlocks, threadsPerBlock>>>(a, b, c, M, N, K, alpha, beta);
#endif
}
