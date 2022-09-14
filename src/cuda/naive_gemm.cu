#include <cstdlib>
#include <cuda_runtime.h>
#ifndef __CUDACC__
#include "device_launch_parameters.h"
#endif
#include <iostream>

__global__ void matrixMul(const float *A, const float *B, float *C,
                          int M, int N, int K)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    if (tx < M && ty < N)
    {
        float c = 0;
        for (int i = 0; i < K; ++i)
        {
            c += A[tx * K + i] * B[i * N + ty];
        }
        C[tx * N + ty] = c;
    }
}

void sgemm(int M, int N, int K, float *a, float *b, float *c, bool beta = false)
{
    int numBlocks = 1;
    dim3 threadsPerBlock(M, N);
    matrixMul<<<numBlocks, threadsPerBlock>>>(a, b, c, M, N, K);
}
