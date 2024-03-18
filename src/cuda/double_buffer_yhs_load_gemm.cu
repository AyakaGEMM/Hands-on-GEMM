#include <cstdlib>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#ifndef __CUDACC__
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
void __syncthreads(); // workaround __syncthreads warning
void __syncwarp();
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

__global__ void matrixMul(const float *A, const float *B, float *C,
                          int M, int N, int K, float alpha, float beta)
{
    const size_t baseX = blockIdx.x * blockDim.x * BLOCK_M_COMPUTE;
    const size_t baseY = blockIdx.y * blockDim.y * BLOCK_N_COMPUTE;

    const int moveNum = shared_memory_element / (BLOCK_SIZE * BLOCK_SIZE) / 2;
    const size_t baseIdx = threadIdx.y * blockDim.x + threadIdx.x;

    constexpr size_t threadsNum = BLOCK_SIZE * BLOCK_SIZE;

    float c[BLOCK_M_COMPUTE * BLOCK_N_COMPUTE] = {};
    constexpr size_t subAlda = BLOCK_M + 4; // plus 4 here to avoid bank conflict and maintain float4 read

    __shared__ float subA[2][subAlda * BLOCK_K];
    __shared__ float subB[2][BLOCK_N * BLOCK_K];

    float4 regB[BLOCK_M_COMPUTE / 4]; // hopefully, these should reside in register.
    float4 regA[BLOCK_M_COMPUTE / 4];

    const float *baseA = A + baseY * K;
    const float *baseB = B + baseX;

    auto compute_stage_idx = 0;

    int rowA = baseIdx >> 1, rowB = baseIdx >> 5, colA = (baseIdx & 1) << 2, colB = baseIdx & 31;
    int warpId = baseIdx >> 5, warpBaseId = baseIdx & 31;
    int rowC = ((warpId >> 1 << 3) + ((warpBaseId >> 4) << 1) + (warpBaseId & 1)) << 2, colC = (((warpId & 1) << 4) + ((warpBaseId & 15) >> 1)) << 2;
    float *baseC = C + (baseY + rowC) * N + baseX + colC;

    float4 preA, preB;

    preB.x = baseB[rowB * N + colB];
    preB.y = baseB[rowB * N + colB + 32];
    preB.z = baseB[rowB * N + colB + 32 * 2];
    preB.w = baseB[rowB * N + colB + 32 * 3];
    preA = *reinterpret_cast<const float4 *>(baseA + rowA * K + colA);

    subB[0][(baseIdx / 32) * BLOCK_N + (baseIdx & 31)] = preB.x;
    subB[0][(baseIdx / 32) * BLOCK_N + (baseIdx & 31) + 32] = preB.y;
    subB[0][(baseIdx / 32) * BLOCK_N + (baseIdx & 31) + 32 * 2] = preB.z;
    subB[0][(baseIdx / 32) * BLOCK_N + (baseIdx & 31) + 32 * 3] = preB.w;

    subA[0][rowA + colA * subAlda] = preA.x;
    subA[0][rowA + (colA + 1) * subAlda] = preA.y;
    subA[0][rowA + (colA + 2) * subAlda] = preA.z;
    subA[0][rowA + (colA + 3) * subAlda] = preA.w;

    __syncthreads();

    for (int i = BLOCK_K; i < K; i += BLOCK_K)
    {
        preB.x = baseB[rowB * N + i * N + colB];
        preB.y = baseB[rowB * N + i * N + colB + 32];
        preB.z = baseB[rowB * N + i * N + colB + 32 * 2];
        preB.w = baseB[rowB * N + i * N + colB + 32 * 3];
        preA = *reinterpret_cast<const float4 *>(baseA + i + rowA * K + colA);

#pragma unroll
        for (int ii = 0; ii < BLOCK_K; ii++)
        {
            regB[0] = *reinterpret_cast<float4 *>(&subB[compute_stage_idx][colC + BLOCK_N * ii]);
            regB[1] = *reinterpret_cast<float4 *>(&subB[compute_stage_idx][colC + 32 + BLOCK_N * ii]);

            regA[0] = *reinterpret_cast<float4 *>(&subA[compute_stage_idx][rowC + ii * subAlda]);
            regA[1] = *reinterpret_cast<float4 *>(&subA[compute_stage_idx][rowC + 16 + ii * subAlda]);

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

        subB[compute_stage_idx][(baseIdx / 32) * BLOCK_N + (baseIdx & 31)] = preB.x;
        subB[compute_stage_idx][(baseIdx / 32) * BLOCK_N + (baseIdx & 31) + 32] = preB.y;
        subB[compute_stage_idx][(baseIdx / 32) * BLOCK_N + (baseIdx & 31) + 32 * 2] = preB.z;
        subB[compute_stage_idx][(baseIdx / 32) * BLOCK_N + (baseIdx & 31) + 32 * 3] = preB.w;

        subA[compute_stage_idx][rowA + colA * subAlda] = preA.x;
        subA[compute_stage_idx][rowA + (colA + 1) * subAlda] = preA.y;
        subA[compute_stage_idx][rowA + (colA + 2) * subAlda] = preA.z;
        subA[compute_stage_idx][rowA + (colA + 3) * subAlda] = preA.w;
        __syncthreads();
    }

#pragma unroll
    for (int ii = 0; ii < BLOCK_K; ii++)
    {
        regB[0] = *reinterpret_cast<float4 *>(&subB[compute_stage_idx][colC + BLOCK_N * ii]);
        regB[1] = *reinterpret_cast<float4 *>(&subB[compute_stage_idx][colC + 32 + BLOCK_N * ii]);

        regA[0] = *reinterpret_cast<float4 *>(&subA[compute_stage_idx][rowC + ii * subAlda]);
        regA[1] = *reinterpret_cast<float4 *>(&subA[compute_stage_idx][(rowC + 16) + ii * subAlda]);

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

#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        *reinterpret_cast<float4 *>(&preA) = *reinterpret_cast<float4 *>(&baseC[i * N]);
        preA.x = preA.x * beta + alpha * c[i * BLOCK_N_COMPUTE];
        preA.y = preA.y * beta + alpha * c[1 + i * BLOCK_N_COMPUTE];
        preA.z = preA.z * beta + alpha * c[2 + i * BLOCK_N_COMPUTE];
        preA.w = preA.w * beta + alpha * c[3 + i * BLOCK_N_COMPUTE];
        *reinterpret_cast<float4 *>(&baseC[i * N]) = *reinterpret_cast<float4 *>(&preA);

        *reinterpret_cast<float4 *>(&preA) = *reinterpret_cast<float4 *>(&baseC[i * N + 32]);
        preA.x = preA.x * beta + alpha * c[4 + i * BLOCK_N_COMPUTE];
        preA.y = preA.y * beta + alpha * c[5 + i * BLOCK_N_COMPUTE];
        preA.z = preA.z * beta + alpha * c[6 + i * BLOCK_N_COMPUTE];
        preA.w = preA.w * beta + alpha * c[7 + i * BLOCK_N_COMPUTE];
        *reinterpret_cast<float4 *>(&baseC[i * N + 32]) = *reinterpret_cast<float4 *>(&preA);

        *reinterpret_cast<float4 *>(&preA) = *reinterpret_cast<float4 *>(&baseC[(i + 16) * N]);
        preA.x = preA.x * beta + alpha * c[32 + i * BLOCK_N_COMPUTE];
        preA.y = preA.y * beta + alpha * c[33 + i * BLOCK_N_COMPUTE];
        preA.z = preA.z * beta + alpha * c[34 + i * BLOCK_N_COMPUTE];
        preA.w = preA.w * beta + alpha * c[35 + i * BLOCK_N_COMPUTE];
        *reinterpret_cast<float4 *>(&baseC[(i + 16) * N]) = *reinterpret_cast<float4 *>(&preA);

        *reinterpret_cast<float4 *>(&preA) = *reinterpret_cast<float4 *>(&baseC[(i + 16) * N + 32]);
        preA.x = preA.x * beta + alpha * c[36 + i * BLOCK_N_COMPUTE];
        preA.y = preA.y * beta + alpha * c[37 + i * BLOCK_N_COMPUTE];
        preA.z = preA.z * beta + alpha * c[38 + i * BLOCK_N_COMPUTE];
        preA.w = preA.w * beta + alpha * c[39 + i * BLOCK_N_COMPUTE];
        *reinterpret_cast<float4 *>(&baseC[(i + 16) * N + 32]) = *reinterpret_cast<float4 *>(&preA);
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
