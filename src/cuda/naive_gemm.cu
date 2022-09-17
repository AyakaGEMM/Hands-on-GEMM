#include <cstdlib>
#include <cuda_runtime.h>
#ifndef __CUDACC__
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
void __syncthreads(); // workaround __syncthreads warning
#endif
#include <iostream>
#define BLOCK_SIZE 16 // we assume that every block has equal blockDim.x and y
#define BLOCK_COMPUTE 4

__global__ void matrixMul(const float *A, const float *B, float *C,
                          int M, int N, int K, float alpha, float beta)
{
    int tx = blockIdx.x * blockDim.x * BLOCK_COMPUTE + threadIdx.x;
    int ty = blockIdx.y * blockDim.y * BLOCK_COMPUTE + threadIdx.y;

    int baseX = blockIdx.x * blockDim.x * BLOCK_COMPUTE;
    int baseY = blockIdx.y * blockDim.y * BLOCK_COMPUTE;

    float c[BLOCK_COMPUTE * BLOCK_COMPUTE] = {};

    __shared__ float subA[BLOCK_SIZE * BLOCK_SIZE * BLOCK_COMPUTE * BLOCK_COMPUTE];
    __shared__ float subB[BLOCK_SIZE * BLOCK_SIZE * BLOCK_COMPUTE * BLOCK_COMPUTE];
    for (int i = 0; i < K; i += BLOCK_SIZE * BLOCK_COMPUTE)
    {
        __syncthreads();
        for (int cpi = 0; cpi < BLOCK_COMPUTE; cpi++)
            for (int cpj = 0; cpj < BLOCK_COMPUTE; cpj++)
            {
                subA[blockDim.x * BLOCK_COMPUTE * (threadIdx.x + cpi * BLOCK_SIZE) + (threadIdx.y + cpj * BLOCK_SIZE)] = A[K * (tx + cpi * BLOCK_SIZE) + i + threadIdx.y + cpj * BLOCK_SIZE];
                subB[blockDim.y * BLOCK_COMPUTE * (threadIdx.x + cpi * BLOCK_SIZE) + (threadIdx.y + cpj * BLOCK_SIZE)] = B[N * (i + threadIdx.x + cpi * BLOCK_SIZE) + ty + cpj * BLOCK_SIZE];
            }
        __syncthreads();
#pragma unroll(4)
        for (int ii = 0; ii < BLOCK_COMPUTE * BLOCK_SIZE; ii++)
        {
#pragma unroll
            for (int cpi = 0; cpi < BLOCK_COMPUTE; cpi++)
#pragma unroll
                for (int cpj = 0; cpj < BLOCK_COMPUTE; cpj++)
                {
                    c[cpi * BLOCK_COMPUTE + cpj] += subA[(threadIdx.x * BLOCK_COMPUTE + cpi) * blockDim.x * BLOCK_COMPUTE + ii] * subB[threadIdx.y * BLOCK_COMPUTE + cpj + blockDim.x * BLOCK_COMPUTE * ii];
                }
        }
    }

    for (int i = 0; i < BLOCK_COMPUTE; i++)
        for (int j = 0; j < BLOCK_COMPUTE; j++)
            C[(baseX + threadIdx.x * BLOCK_COMPUTE + i) * N + baseY + threadIdx.y * BLOCK_COMPUTE + j] = beta * C[(baseX + threadIdx.x * BLOCK_COMPUTE + i) * N + baseY + threadIdx.y * BLOCK_COMPUTE + j] + alpha * c[i * BLOCK_COMPUTE + j];
}

void sgemm(int M, int N, int K, float *a, float *b, float *c, float alpha = 1, float beta = 0)
{
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((M + threadsPerBlock.x * BLOCK_COMPUTE - 1) / (threadsPerBlock.x * BLOCK_COMPUTE), (N + threadsPerBlock.y * BLOCK_COMPUTE - 1) / (threadsPerBlock.y * BLOCK_COMPUTE));
#ifdef __CUDACC__ // workaround for stupid vscode intellisense
    matrixMul<<<numBlocks, threadsPerBlock>>>(a, b, c, M, N, K, alpha, beta);
#endif
}
