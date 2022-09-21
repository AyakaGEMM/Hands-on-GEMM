#include <cstdlib>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>

#ifndef __CUDACC__
#define __CUDACC__
#define __HAHA__
#endif

#include <cooperative_groups/memcpy_async.h>

#ifdef __HAHA__
#undef __CUDACC__
#endif
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
const size_t BLOCK_K_COMPUTE = BLOCK_K / BLOCK_SIZE;

const int shared_memory_A = BLOCK_M * BLOCK_K;
const int shared_memory_B = BLOCK_N * BLOCK_K;
const int shared_memory_element = shared_memory_A + shared_memory_B;
const int shared_memory_size = shared_memory_element * sizeof(float); // shared memory to use.
#define colM(a, i, j, lda) a[((j) * (lda)) + (i)]
#define rowM(a, i, j, lda) a[(j) + (i) * (lda)]

__forceinline__ __device__ auto convertColIdx(int idx, const float *begin, int subM, int subN, int N)
{
    int m = idx / subM, n = idx % subM;
    return begin + m + n * N;
}

__forceinline__ __device__ auto convertRowIdx(int idx, const float *begin, int subM, int subN, int N)
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
    const size_t baseIdx = threadIdx.x * blockDim.x + threadIdx.y;

    auto block = cooperative_groups::this_thread_block();

    float c[BLOCK_M_COMPUTE * BLOCK_N_COMPUTE] = {};

    constexpr size_t stage_count = 2;
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, stage_count> shared_state;
    auto pipeline = cuda::make_pipeline(block, &shared_state);

    __shared__ float subA[stage_count][BLOCK_M * BLOCK_K];
    __shared__ float subB[stage_count][BLOCK_N * BLOCK_K];

    pipeline.producer_acquire();
#pragma unroll
    for (int idx = 0; idx < moveNum; idx++)
    {
        cuda::memcpy_async(block, subA[0] + baseIdx + idx * BLOCK_SIZE * BLOCK_SIZE, convertColIdx(baseIdx + idx * BLOCK_SIZE * BLOCK_SIZE, A + baseX * K, BLOCK_M, BLOCK_K, K), sizeof(float), pipeline);
        cuda::memcpy_async(block, subB[0] + baseIdx + idx * BLOCK_SIZE * BLOCK_SIZE, convertRowIdx(baseIdx + idx * BLOCK_SIZE * BLOCK_SIZE, B + baseY, BLOCK_K, BLOCK_N, N), sizeof(float), pipeline);
        // subA[0][baseIdx + idx * BLOCK_SIZE * BLOCK_SIZE] = *convertColIdx(baseIdx + idx * BLOCK_SIZE * BLOCK_SIZE, A + baseX * K, BLOCK_M, BLOCK_K, K);
        // subB[0][baseIdx + idx * BLOCK_SIZE * BLOCK_SIZE] = *convertRowIdx(baseIdx + idx * BLOCK_SIZE * BLOCK_SIZE, B + baseY, BLOCK_K, BLOCK_N, N);
    }
    pipeline.producer_commit();

    for (int i = BLOCK_K; i < K; i += BLOCK_K)
    {
        size_t copy_stage_idx = (i / BLOCK_K) % 2;
        size_t compute_stage_idx = (i / BLOCK_K - 1) % 2;

        pipeline.producer_acquire();
#pragma unroll
        for (int idx = 0; idx < moveNum; idx++)
        {
            cuda::memcpy_async(block, subA[0] + baseIdx + idx * BLOCK_SIZE * BLOCK_SIZE, convertColIdx(baseIdx + idx * BLOCK_SIZE * BLOCK_SIZE, A + baseX * K, BLOCK_M, BLOCK_K, K), sizeof(float), pipeline);
            cuda::memcpy_async(block, subB[0] + baseIdx + idx * BLOCK_SIZE * BLOCK_SIZE, convertRowIdx(baseIdx + idx * BLOCK_SIZE * BLOCK_SIZE, B + baseY, BLOCK_K, BLOCK_N, N), sizeof(float), pipeline);
            // subA[copy_stage_idx][baseIdx + idx * BLOCK_SIZE * BLOCK_SIZE] = *convertColIdx(baseIdx + idx * BLOCK_SIZE * BLOCK_SIZE, A + baseX * K + i, BLOCK_M, BLOCK_K, K);
            // subB[copy_stage_idx][baseIdx + idx * BLOCK_SIZE * BLOCK_SIZE] = *convertRowIdx(baseIdx + idx * BLOCK_SIZE * BLOCK_SIZE, B + baseY + i * N, BLOCK_K, BLOCK_N, N);
        }
        pipeline.producer_commit();

        pipeline.consumer_wait();
#pragma unroll(4)
        for (int ii = 0; ii < BLOCK_K; ii++)
        {
            float regB[BLOCK_M_COMPUTE]; // hopefully, these should reside in register.
#pragma unroll
            for (int cpj = 0; cpj < BLOCK_N_COMPUTE; cpj++)
            {
                regB[cpj] = subB[compute_stage_idx][threadIdx.y * BLOCK_N_COMPUTE + cpj + BLOCK_N * ii];
            }
#pragma unroll
            for (int cpi = 0; cpi < BLOCK_M_COMPUTE; cpi++)
            {
                float regA = subA[compute_stage_idx][(threadIdx.x * BLOCK_M_COMPUTE + cpi) + ii * BLOCK_M];
#pragma unroll
                for (int cpj = 0; cpj < BLOCK_N_COMPUTE; cpj++)
                {
                    c[cpi * BLOCK_M_COMPUTE + cpj] += regA * regB[cpj];
                }
            }
        }
        pipeline.consumer_release();
    }

    pipeline.consumer_wait();
#pragma unroll(4)
    for (int ii = 0; ii < BLOCK_K; ii++)
    {
        float regB[BLOCK_M_COMPUTE]; // hopefully, these should reside in register.
#pragma unroll
        for (int cpj = 0; cpj < BLOCK_N_COMPUTE; cpj++)
        {
            regB[cpj] = subB[(K / BLOCK_K - 1) % 2][threadIdx.y * BLOCK_N_COMPUTE + cpj + BLOCK_N * ii];
        }
#pragma unroll
        for (int cpi = 0; cpi < BLOCK_M_COMPUTE; cpi++)
        {
            float regA = subA[(K / BLOCK_K - 1) % 2][(threadIdx.x * BLOCK_M_COMPUTE + cpi) + ii * BLOCK_M];
#pragma unroll
            for (int cpj = 0; cpj < BLOCK_N_COMPUTE; cpj++)
            {
                c[cpi * BLOCK_M_COMPUTE + cpj] += regA * regB[cpj];
            }
        }
    }
    pipeline.consumer_release();

    for (int i = 0; i < BLOCK_M_COMPUTE; i++)
        for (int j = 0; j < BLOCK_N_COMPUTE; j++)
            C[(baseX + threadIdx.x * BLOCK_M_COMPUTE + i) * N + baseY + threadIdx.y * BLOCK_N_COMPUTE + j] = 0 * C[(baseX + threadIdx.x * BLOCK_M_COMPUTE + i) * N + baseY + threadIdx.y * BLOCK_N_COMPUTE + j] + alpha * c[i * BLOCK_M_COMPUTE + j];
}

void sgemm(int M, int N, int K, float *a, float *b, float *c, float alpha = 1, float beta = 0)
{
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((M + BLOCK_M - 1) / BLOCK_M, (N + BLOCK_N - 1) / BLOCK_N);
#ifdef __CUDACC__ // workaround for stupid vscode intellisense
    matrixMul<<<numBlocks, threadsPerBlock>>>(a, b, c, M, N, K, alpha, beta);
#endif
}
