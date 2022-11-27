#include <cstdlib>
#include <cuda_runtime.h>
#include <algorithm>
#ifndef __CUDACC__
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
void __syncthreads(); // workaround __syncthreads warning
#endif
#include <iostream>
constexpr size_t BLOCK_SIZE =
    16; // we assume that every block has equal blockDim.x and blockDim.y
constexpr size_t BLOCK_M =
    128; // These const values decide how many thing a thread compute and the
         // amount of shared memory to allocate.
constexpr size_t BLOCK_N = 128;
constexpr size_t BLOCK_K =
    8; // don't set 64 here, it will cause bank conflict and lower occupancy.
constexpr size_t BLOCK_M_COMPUTE = BLOCK_M / BLOCK_SIZE;
constexpr size_t BLOCK_N_COMPUTE = BLOCK_N / BLOCK_SIZE;

__global__ void matrixMul(const float *A, const float *B, float *C,
                          int M, int N, int K, float alpha, float beta)
{
  int tx = (blockIdx.x * blockDim.x + threadIdx.x) * BLOCK_M_COMPUTE;
  int ty = (blockIdx.y * blockDim.y + threadIdx.y) * BLOCK_N_COMPUTE;

  float aa[BLOCK_M_COMPUTE * BLOCK_N_COMPUTE] = {};

  for (int i = 0; i < K; i++)
    for (int m = 0; m < BLOCK_M_COMPUTE; m++)
      for (int n = 0; n < BLOCK_N_COMPUTE; n++) {
        aa[m * BLOCK_N_COMPUTE + n] += A[(tx + m) * K + i] * B[i * N + ty + n];
      }

  for (int m = 0; m < BLOCK_M_COMPUTE; m++)
    for (int n = 0; n < BLOCK_N_COMPUTE; n++)
      C[(tx + m) * N + ty + n] =
          beta * C[(tx + m) * N + ty + n] +
          alpha * aa[m * BLOCK_N_COMPUTE +
                     n]; // we multiply alpha here to reduce the alpha cal num.
}

void sgemm(int M, int N, int K, float *a, float *b, float *c, float alpha = 1, float beta = 0)
{
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((M + BLOCK_M - 1) / BLOCK_M, (N + BLOCK_N - 1) / BLOCK_N);
#ifdef __CUDACC__ // workaround for stupid vscode intellisense
    matrixMul<<<numBlocks, threadsPerBlock>>>(a, b, c, M, N, K, alpha, beta);
#endif
}
