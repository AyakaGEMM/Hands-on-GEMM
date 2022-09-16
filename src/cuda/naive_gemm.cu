#include <cstdlib>
#include <cuda_runtime.h>
#ifndef __CUDACC__
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
void __syncthreads(); // workaround __syncthreads warning
#endif
#include <iostream>
#define BLOCK_SIZE 16 // we assume that every block has equal blockDim.x and y

__global__ void matrixMul(const float *A, const float *B, float *C,
                          int M, int N, int K, float alpha, float beta)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    float c = beta * C[tx * N + ty];

    __shared__ float subA[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float subB[BLOCK_SIZE * BLOCK_SIZE];
    for (int i = 0; i < K; i += blockDim.x)
    {
        __syncthreads();
        subA[blockDim.x * threadIdx.x + threadIdx.y] = A[K * tx + i + threadIdx.y];
        subB[blockDim.y * threadIdx.x + threadIdx.y] = B[N * (i + threadIdx.x) + ty];
        __syncthreads();
        for (int ii = 0; ii < blockDim.x; ii++)
        {
            c += alpha * subA[threadIdx.x * blockDim.x + ii] * subB[threadIdx.y + blockDim.x * ii];
        }
    }

    C[tx * N + ty] = c;
}

void sgemm(int M, int N, int K, float *a, float *b, float *c, float alpha = 1, float beta = 0)
{
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
#ifdef __CUDACC__ // workaround for stupid vscode intellisense
    matrixMul<<<numBlocks, threadsPerBlock>>>(a, b, c, M, N, K, alpha, beta);
#endif
}
