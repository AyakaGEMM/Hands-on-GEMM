#include <cstdlib>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
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
constexpr size_t BLOCK_K_COMPUTE = BLOCK_K / BLOCK_SIZE;

constexpr int shared_memory_A = BLOCK_M * BLOCK_K;
constexpr int shared_memory_B = BLOCK_N * BLOCK_K;
constexpr int shared_memory_element = shared_memory_A + shared_memory_B;
constexpr int shared_memory_size = shared_memory_element * sizeof(float); // shared memory to use.
#define colM(a, i, j, lda) a[((j) * (lda)) + (i)]
#define rowM(a, i, j, lda) a[(j) + (i) * (lda)]
constexpr size_t stage_count = 2;

__global__ void matrixMul(const float *A, const float *B, float *C,
                          int M, int N, int K, float alpha, float beta)
{
    const size_t baseX = blockIdx.x * blockDim.x * BLOCK_M_COMPUTE;
    const size_t baseY = blockIdx.y * blockDim.y * BLOCK_N_COMPUTE;
    const size_t baseIdx = threadIdx.y * blockDim.x + threadIdx.x;

    constexpr size_t threadsNum = BLOCK_SIZE * BLOCK_SIZE;

    float c[BLOCK_M_COMPUTE * BLOCK_N_COMPUTE] = {};
    float resC[BLOCK_M_COMPUTE * BLOCK_N_COMPUTE] = {};

    const float *baseA = A + baseX * K;
    const float *baseB = B + baseY;

    int colA = baseIdx / 2, colB = baseIdx / (BLOCK_N / 4), rowA = (baseIdx & 1) * 4, rowB = (baseIdx * 4) % BLOCK_N;
    int warpId = baseIdx / 32, warpBaseId = baseIdx % 32;
    int colC = (warpId / 2 * 4 + warpBaseId % 4) * BLOCK_M_COMPUTE, rowC = ((warpId % 2) * 8 + warpBaseId / 4) * BLOCK_N_COMPUTE;
    float *baseC = C + (baseX + colC) * N + baseY + rowC;

    auto block = cooperative_groups::this_thread_block();

    __shared__ cuda::pipeline_shared_state<cuda::thread_scope::thread_scope_block, stage_count> shared_state;
    __shared__ float subA[stage_count][BLOCK_M * BLOCK_K];
    __shared__ float subB[stage_count][BLOCK_N * BLOCK_K];

    float4 regB[stage_count][BLOCK_M_COMPUTE / 4]; // hopefully, these should reside in register.
    float4 regA[stage_count][BLOCK_M_COMPUTE / 4];
    auto pipeline = cuda::make_pipeline(block, &shared_state);
    size_t copy_stage_idx = 0;
    size_t compute_stage_idx = 1;

    pipeline.producer_acquire();

    cuda::memcpy_async(block, regA[0], baseA + colA * K + rowA, sizeof(float4), pipeline);
    cuda::memcpy_async(block, subB[0] + baseIdx * 4, baseB + colB * N + rowB, sizeof(float4), pipeline);

    cuda::memcpy_async(block, subA[0] + colA + rowA * BLOCK_M, &(regA[0][0].x), sizeof(float), pipeline);
    cuda::memcpy_async(block, subA[0] + colA + (rowA + 1) * BLOCK_M, &(regA[0][0].y), sizeof(float), pipeline);
    cuda::memcpy_async(block, subA[0] + colA + (rowA + 2) * BLOCK_M, &(regA[0][0].z), sizeof(float), pipeline);
    cuda::memcpy_async(block, subA[0] + colA + (rowA + 3) * BLOCK_M, &(regA[0][0].w), sizeof(float), pipeline);

    copy_stage_idx ^= 1;
    pipeline.producer_commit();

    for (int i = BLOCK_K; i < K; i += BLOCK_K)
    {

        pipeline.producer_acquire();

        cuda::memcpy_async(block, regA[copy_stage_idx], baseA + i + colA * K + rowA, sizeof(float4), pipeline);
        cuda::memcpy_async(block, subB[copy_stage_idx] + baseIdx * 4, baseB + i * N + colB * N + rowB, sizeof(float4), pipeline);

        cuda::memcpy_async(block, subA[copy_stage_idx] + colA + rowA * BLOCK_M, &(regA[copy_stage_idx][0].x), sizeof(float), pipeline);
        cuda::memcpy_async(block, subA[copy_stage_idx] + colA + (rowA + 1) * BLOCK_M, &(regA[copy_stage_idx][0].y), sizeof(float), pipeline);
        cuda::memcpy_async(block, subA[copy_stage_idx] + colA + (rowA + 2) * BLOCK_M, &(regA[copy_stage_idx][0].z), sizeof(float), pipeline);
        cuda::memcpy_async(block, subA[copy_stage_idx] + colA + (rowA + 3) * BLOCK_M, &(regA[copy_stage_idx][0].w), sizeof(float), pipeline);
        copy_stage_idx ^= 1;
        pipeline.producer_commit();

        pipeline.consumer_wait();
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
        pipeline.consumer_release();
    }

    pipeline.consumer_wait();
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
    pipeline.consumer_release();

#pragma unroll
    for (int i = 0; i < BLOCK_M_COMPUTE; i++)
#pragma unroll
        for (int j = 0; j < BLOCK_N_COMPUTE; j += 4)
            *reinterpret_cast<float4 *>(&resC[i * BLOCK_M_COMPUTE + j]) = *reinterpret_cast<float4 *>(&baseC[i * N + j]);

#pragma unroll
    for (int i = 0; i < BLOCK_M_COMPUTE; i++)
#pragma unroll
        for (int j = 0; j < BLOCK_N_COMPUTE; j++)
            resC[i * BLOCK_M_COMPUTE + j] = resC[i * BLOCK_M_COMPUTE + j] * beta + alpha * c[i * BLOCK_M_COMPUTE + j];

#pragma unroll
    for (int i = 0; i < BLOCK_M_COMPUTE; i++)
#pragma unroll
        for (int j = 0; j < BLOCK_N_COMPUTE; j += 4)
            *reinterpret_cast<float4 *>(&baseC[i * N + j]) = *reinterpret_cast<float4 *>(&resC[i * BLOCK_M_COMPUTE + j]);
}

void sgemm(int M, int N, int K, float *a, float *b, float *c, float alpha = 1, float beta = 0)
{
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((M + BLOCK_M - 1) / BLOCK_M, (N + BLOCK_N - 1) / BLOCK_N);
#ifdef __CUDACC__ // workaround for stupid vscode intellisense
    matrixMul<<<numBlocks, threadsPerBlock>>>(a, b, c, M, N, K, alpha, beta);
#endif
}
