#include <cstdlib>
#include <cuda_runtime.h>
#include <algorithm>
#ifndef __CUDACC__
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
void __syncthreads(); // workaround __syncthreads warning
#endif
#include <iostream>
#define BLOCK_SIZE 16 // we assume that every block has equal blockDim.x and blockDim.y

__global__ void matrixMul(const float *A, const float *B, float *C,
                          int M, int N, int K, float alpha, float beta)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    int baseX = blockIdx.x * blockDim.x;
    int baseY = blockIdx.y * blockDim.y;

    float c = 0;
    float aa[64];

    for (int i = 0; i < 8; i++)
        for (int m = 0; m < 8; m++)
            for (int n = 0; n < 8; n++)
            {
                aa[m * 8 + n] += A[(tx + m) * K + i] * B[i * N + ty + n];
            }
    for (int m = 0; m < 8; m++)
        for (int n = 0; n < 8; n++)
            C[(tx + m) * N + ty + n] = beta * C[(tx + m) * N + ty + n] + alpha * aa[m * 8 + n]; // we multiply alpha here to reduce the alpha cal num.
}

void sgemm(int M, int N, int K, float *a, float *b, float *c, float alpha = 1, float beta = 0)
{
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((M + threadsPerBlock.x - 1) / (threadsPerBlock.x), (N + threadsPerBlock.y - 1) / (threadsPerBlock.y));
#ifdef __CUDACC__ // workaround for stupid vscode intellisense
    matrixMul<<<numBlocks, threadsPerBlock>>>(a, b, c, M, N, K, alpha, beta);
#endif
}
