#include <cstdlib>
#include <cuda_runtime.h>
#ifndef __CUDACC__
#include "device_launch_parameters.h"
#endif
#include <iostream>

__global__ void matrixMul(const float *A, const float *B, float *C,
                          int M, int N, int K, float alpha, float beta)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (tx < M && ty < N)
    {
        float c = beta * C[tx * N + ty];
        for (int i = 0; i < K; ++i)
        {
            c += alpha * A[tx * K + i] * B[i * N + ty];
        }
        C[tx * N + ty] = c;
    }
}

void sgemm(int M, int N, int K, float *a, float *b, float *c, float alpha = 1, float beta = 0)
{
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matrixMul<<<numBlocks, threadsPerBlock>>>(a, b, c, M, N, K, alpha, beta);
}
