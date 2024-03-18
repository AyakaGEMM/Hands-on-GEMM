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
    float resC[BLOCK_M_COMPUTE * BLOCK_N_COMPUTE] = {};

    __shared__ float subA[BLOCK_M * BLOCK_K];
    __shared__ float subB[BLOCK_N * BLOCK_K];

    float4 regB[BLOCK_M_COMPUTE / 4]; // hopefully, these should reside in register.
    float4 regA[BLOCK_M_COMPUTE / 4];

    const float *baseA = A + baseY * K;
    const float *baseB = B + baseX;

    int rowA = baseIdx >> 1, rowB = baseIdx >> 5, colA = (baseIdx & 1) << 2, colB = (baseIdx << 2) & 127;
    int warpId = baseIdx >> 5, warpBaseId = baseIdx & 31;
    int rowC = ((warpId >> 1 << 3) + (warpBaseId & 3)) << 2, colC = (((warpId & 1) << 4) + (warpBaseId >> 2)) << 2;
    float *baseC = C + (baseY + rowC) * N + baseX + colC;

    for (int i = 0; i < K; i += BLOCK_K)
    {
        regB[0] = *reinterpret_cast<const float4 *>(baseB + i * N + rowB * N + colB);
        regA[0] = *reinterpret_cast<const float4 *>(baseA + i + rowA * K + colA);
        *reinterpret_cast<float4 *>(&subB[baseIdx * 4]) = regB[0];
        subA[rowA + colA * BLOCK_M] = regA[0].x;
        subA[rowA + (colA + 1) * BLOCK_M] = regA[0].y;
        subA[rowA + (colA + 2) * BLOCK_M] = regA[0].z;
        subA[rowA + (colA + 3) * BLOCK_M] = regA[0].w;

        __syncthreads();
#pragma unroll
        for (int ii = 0; ii < BLOCK_K; ii++)
        {
            regB[0] = *reinterpret_cast<float4 *>(&subB[colC + BLOCK_N * ii]);
            regB[1] = *reinterpret_cast<float4 *>(&subB[colC + 32 + BLOCK_N * ii]);

            regA[0] = *reinterpret_cast<float4 *>(&subA[rowC + ii * BLOCK_M]);
            regA[1] = *reinterpret_cast<float4 *>(&subA[(rowC + 16) + ii * BLOCK_M]);

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
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        *reinterpret_cast<float4 *>(&regA[0]) = *reinterpret_cast<float4 *>(&baseC[i * N]);
        regA[0].x = regA[0].x * beta + alpha * c[i * BLOCK_N_COMPUTE];
        regA[0].y = regA[0].y * beta + alpha * c[1 + i * BLOCK_N_COMPUTE];
        regA[0].z = regA[0].z * beta + alpha * c[2 + i * BLOCK_N_COMPUTE];
        regA[0].w = regA[0].w * beta + alpha * c[3 + i * BLOCK_N_COMPUTE];
        *reinterpret_cast<float4 *>(&baseC[i * N]) = *reinterpret_cast<float4 *>(&regA[0]);

        *reinterpret_cast<float4 *>(&regA[0]) = *reinterpret_cast<float4 *>(&baseC[i * N + 32]);
        regA[0].x = regA[0].x * beta + alpha * c[4 + i * BLOCK_N_COMPUTE];
        regA[0].y = regA[0].y * beta + alpha * c[5 + i * BLOCK_N_COMPUTE];
        regA[0].z = regA[0].z * beta + alpha * c[6 + i * BLOCK_N_COMPUTE];
        regA[0].w = regA[0].w * beta + alpha * c[7 + i * BLOCK_N_COMPUTE];
        *reinterpret_cast<float4 *>(&baseC[i * N + 32]) = *reinterpret_cast<float4 *>(&regA[0]);

        *reinterpret_cast<float4 *>(&regA[0]) = *reinterpret_cast<float4 *>(&baseC[(i + 16) * N]);
        regA[0].x = regA[0].x * beta + alpha * c[32 + i * BLOCK_N_COMPUTE];
        regA[0].y = regA[0].y * beta + alpha * c[33 + i * BLOCK_N_COMPUTE];
        regA[0].z = regA[0].z * beta + alpha * c[34 + i * BLOCK_N_COMPUTE];
        regA[0].w = regA[0].w * beta + alpha * c[35 + i * BLOCK_N_COMPUTE];
        *reinterpret_cast<float4 *>(&baseC[(i + 16) * N]) = *reinterpret_cast<float4 *>(&regA[0]);

        *reinterpret_cast<float4 *>(&regA[0]) = *reinterpret_cast<float4 *>(&baseC[(i + 16) * N + 32]);
        regA[0].x = regA[0].x * beta + alpha * c[36 + i * BLOCK_N_COMPUTE];
        regA[0].y = regA[0].y * beta + alpha * c[37 + i * BLOCK_N_COMPUTE];
        regA[0].z = regA[0].z * beta + alpha * c[38 + i * BLOCK_N_COMPUTE];
        regA[0].w = regA[0].w * beta + alpha * c[39 + i * BLOCK_N_COMPUTE];
        *reinterpret_cast<float4 *>(&baseC[(i + 16) * N + 32]) = *reinterpret_cast<float4 *>(&regA[0]);
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
