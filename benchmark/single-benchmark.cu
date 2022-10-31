// Copied from https://github.com/Cjkkkk/CUDA_gemm/blob/14b517370609d322647c55fe9136b6d81c2ba9a7/benchmark/benchmark_dense.cu

#include <stdio.h>
#include <stdlib.h>
#include <vector>

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "cuda_help_func.hpp"
#include "utils.hpp"

#define ASIZE(type) (sizeof(type) * M * K)
#define BSIZE(type) (sizeof(type) * K * N)
#define CSIZE(type) (sizeof(type) * M * N)
#define MAXSIZE(type) (sizeof(type) * nmax * nmax)

extern void sgemm(int, int, int, float *, float *, float *, float, float);

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        printf("usage: ./main [MAX_TEST_SIZE]\n");
        exit(0);
    }
    std::vector<int> test_sizes;
    size_t nmax = atoi(argv[1]);

    float *h_A = new float[nmax * nmax];
    float *h_B = new float[nmax * nmax];
    float *h_C = new float[nmax * nmax];
    float *h_C1 = new float[nmax * nmax];

    float *d_A;
    float *d_B;
    float *d_C;

    checkCudaErrors(cudaMalloc(&d_A, MAXSIZE(float)));
    checkCudaErrors(cudaMalloc(&d_B, MAXSIZE(float)));
    checkCudaErrors(cudaMalloc(&d_C, MAXSIZE(float)));

    cublasHandle_t blas_handle;
    checkCuBlasErrors(cublasCreate(&blas_handle));

    size_t M = nmax;
    size_t K = nmax;
    size_t N = nmax;

    printf("\nSize M: %u, N: %u, K: %u\n", M, N, K);

    double msecPerMatrixMul[2] = {0, 0};
    double gigaFlops[2] = {0, 0};
    double flopsPerMatrixMul = 2.0 * M * N * K;

    float alpha = 2.0;
    float beta = 2.0;

    // 生成A的数据
    genRandomMatrix(h_A, M, K);
    genRandomMatrix(h_B, K, N);
    genRandomMatrix(h_C, M, N);
    copyMatrix(h_C1, h_C, M, N);

    checkCudaErrors(cudaMemcpy(d_A, h_A, ASIZE(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, BSIZE(float), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float msecTotal = 0;
    int nIter = 10;

    checkCudaErrors(cudaMemcpy(d_C, h_C, CSIZE(float), cudaMemcpyHostToDevice));

    // dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
    // dim3 dimGrid(N / BLOCK_SIZE_N, M / BLOCK_SIZE_M);
    // if (N % BLOCK_SIZE_N != 0)
    //     dimGrid.x++;
    // if (M % BLOCK_SIZE_M != 0)
    //     dimGrid.y++;

    // warm up here (not sure whether we need this or not)
    sgemm(M, N, K, d_A, d_B, d_C, alpha, beta);

    checkCudaErrors(cudaEventRecord(start));
    for (int run = 0; run < nIter; run++)
    {
        sgemm(M, N, K, d_A, d_B, d_C, alpha, beta);
    }
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    msecPerMatrixMul[0] = msecTotal / nIter;
    gigaFlops[0] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[0] / 1000.0f);
    printf("My gemm Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
           gigaFlops[0],
           msecPerMatrixMul[0],
           flopsPerMatrixMul);

    // Free Memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C1;
}