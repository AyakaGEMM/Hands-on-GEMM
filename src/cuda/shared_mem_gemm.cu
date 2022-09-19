#include <cstdlib>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#ifndef __CUDACC__
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
void __syncthreads(); // workaround __syncthreads warning
#endif
#include <iostream>
#define BLOCK_COMPUTE 4
const size_t BLOCK_SIZE = 16; // we assume that every block has equal blockDim.x and blockDim.y
const size_t BLOCK_M = 64;    // These const values decide how many thing a thread compute and the amount of shared memory to allocate.
const size_t BLOCK_N = 64;
const size_t BLOCK_K = 16; // don't set 64 here, it will cause bank conflict and lower occupancy.
const size_t BLOCK_M_COMPUTE = BLOCK_M / BLOCK_SIZE;
const size_t BLOCK_N_COMPUTE = BLOCK_N / BLOCK_SIZE;
const size_t BLOCK_K_COMPUTE = BLOCK_K / BLOCK_SIZE;

const int shared_memory_A = BLOCK_M * BLOCK_K;
const int shared_memory_B = BLOCK_N * BLOCK_K;
const int shared_memory_element = shared_memory_A + shared_memory_B;
const int shared_memory_size = shared_memory_element * sizeof(float); // shared memory to use.
#define colM(a, i, j, lda) a[((j) * (lda)) + (i)]
#define rowM(a, i, j, lda) a[(j) + (i) * (lda)]

__forceinline__ __device__ float convertColIdx(int idx, const float *begin, int subM, int subN, int N)
{
    int m = idx / subM, n = idx % subM;
    return begin[m + n * N];
}

__forceinline__ __device__ float convertRowIdx(int idx, const float *begin, int subM, int subN, int N)
{
    int m = idx / subN, n = idx % subN;
    return begin[m * N + n];
}

__global__ void matrixMul(const float *A, const float *B, float *C,
                          int M, int N, int K, float alpha, float beta)
{
    int tx = blockIdx.x * blockDim.x * BLOCK_COMPUTE + threadIdx.x;
    int ty = blockIdx.y * blockDim.y * BLOCK_COMPUTE + threadIdx.y;

    int baseX = blockIdx.x * blockDim.x * BLOCK_M_COMPUTE;
    int baseY = blockIdx.y * blockDim.y * BLOCK_N_COMPUTE;

    const int moveNum = shared_memory_element / (BLOCK_SIZE * BLOCK_SIZE);
    const float *baseA = A + baseX * K + baseY;
    // const float *baseB = B;

    float c[BLOCK_M_COMPUTE * BLOCK_N_COMPUTE] = {};

    __shared__ float subA[BLOCK_SIZE * BLOCK_SIZE * BLOCK_M_COMPUTE * BLOCK_K_COMPUTE];
    __shared__ float subB[BLOCK_SIZE * BLOCK_SIZE * BLOCK_N_COMPUTE * BLOCK_K_COMPUTE];
    for (int i = 0; i < K; i += BLOCK_K)
    {
        for (int idx = (threadIdx.x * blockDim.x + threadIdx.y) * moveNum; idx < (threadIdx.x * blockDim.x + threadIdx.y) * moveNum + moveNum; idx++)
        {
            if (idx < shared_memory_A)
                subA[idx] = convertColIdx(idx, A + baseX * K + i, BLOCK_M, BLOCK_K, K);
            else
                subB[idx - shared_memory_A] = convertRowIdx(idx - shared_memory_A, B + baseY + i * N, BLOCK_K, BLOCK_N, N);
        }
        // for (int cpi = 0; cpi < BLOCK_COMPUTE; cpi++)
        //     for (int cpj = 0; cpj < BLOCK_COMPUTE; cpj++)
        //     {
        //         colM(subA, threadIdx.x + cpi * BLOCK_SIZE, threadIdx.y + cpj * BLOCK_SIZE, blockDim.x * BLOCK_COMPUTE) = A[K * (tx + cpi * BLOCK_SIZE) + i + threadIdx.y + cpj * BLOCK_SIZE]; // Here we use col major to store smemA to avoid cache miss.
        //         subB[blockDim.y * BLOCK_COMPUTE * (threadIdx.x + cpi * BLOCK_SIZE) + (threadIdx.y + cpj * BLOCK_SIZE)] = B[N * (i + threadIdx.x + cpi * BLOCK_SIZE) + ty + cpj * BLOCK_SIZE];
        //     }
        __syncthreads();
#pragma unroll(4)
        for (int ii = 0; ii < BLOCK_K; ii++)
        {
            float regB[BLOCK_M_COMPUTE]; // hopefully, these should reside in register.
#pragma unroll
            for (int cpj = 0; cpj < BLOCK_N_COMPUTE; cpj++)
            {
                regB[cpj] = subB[threadIdx.y * BLOCK_N_COMPUTE + cpj + BLOCK_N * ii];
            }
#pragma unroll
            for (int cpi = 0; cpi < BLOCK_M_COMPUTE; cpi++)
            {
                float regA = subA[(threadIdx.x * BLOCK_M_COMPUTE + cpi) + ii * BLOCK_M];
#pragma unroll
                for (int cpj = 0; cpj < BLOCK_N_COMPUTE; cpj++)
                {
                    c[cpi * BLOCK_M_COMPUTE + cpj] += regA * regB[cpj];
                }
            }
        }
        __syncthreads();
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
