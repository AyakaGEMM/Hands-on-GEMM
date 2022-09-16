#include <stdio.h>
#include <stdlib.h>
#include <iostream>

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "cuda_help_func.hpp"
#include "utils.hpp"

#define ASIZE(type) (sizeof(type) * M * K)
#define BSIZE(type) (sizeof(type) * K * N)
#define CSIZE(type) (sizeof(type) * M * N)

extern void sgemm(int, int, int, float *, float *, float *, float, float);

int main(int argc, char **argv)
{
    // if (argc != 4)
    //{
    //     printf("usage: ./main [M] [K] [N]\n");
    //     exit(0);
    // }
    size_t M = atoi(argv[1]);
    size_t N = atoi(argv[2]);
    size_t K = atoi(argv[3]);

    std::cout << M << " " << N << " " << K << std::endl;

    float *h_A = new float[M * K];
    float *h_B = new float[N * K];
    float *h_C = new float[M * N];
    float *h_C1 = new float[M * N];

    float *d_A;
    float *d_B;
    float *d_C;

    checkCudaErrors(cudaMalloc(&d_A, ASIZE(float)));
    checkCudaErrors(cudaMalloc(&d_B, BSIZE(float)));
    checkCudaErrors(cudaMalloc(&d_C, CSIZE(float)));

    std::cout << d_C << std::endl;

    double msecPerMatrixMul[2] = {0, 0};
    double gigaFlops[2] = {0, 0};
    double flopsPerMatrixMul = 2.0 * M * N * K;

    const int BLOCK_SIZE_M = 96;
    const int BLOCK_SIZE_K = 32;
    const int BLOCK_SIZE_N = 64;
    const int THREAD_SIZE_Y = 6;
    const int THREAD_SIZE_X = 4;
    const bool ENABLE_DOUBLE_BUFFER = false;

    float alpha = 2;
    float beta = 2;

    // 生成A的数据
    genRandomMatrix(h_A, M, K);
    genRandomMatrix(h_B, K, N);
    genRandomMatrix(h_C, M, N);
    copyMatrix(h_C1, h_C, M, N);

    checkCudaErrors(cudaMemcpy(d_A, h_A, ASIZE(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, BSIZE(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_C, h_C, CSIZE(float), cudaMemcpyHostToDevice)); // Free Memory

    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
        {
            h_C[i * N + j] = beta * h_C[i * N + j];
            for (int k = 0; k < K; k++)
                h_C[i * N + j] += alpha * h_A[i * K + k] * h_B[k * N + j];
        }
    showMatrix(h_C, M, N, "Matrix Ref");
    copyMatrix(h_C, h_C1, M, N);

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float msecTotal = 0;
    int nIter = 100;

    sgemm(M, N, K, d_A, d_B, d_C, alpha, beta);

    checkCudaErrors(cudaMemcpy(h_C, d_C, CSIZE(float), cudaMemcpyDeviceToHost));

    cublasHandle_t blas_handle;
    checkCuBlasErrors(cublasCreate(&blas_handle));
    checkCudaErrors(cudaMemcpy(d_C, h_C1, CSIZE(float), cudaMemcpyHostToDevice));

    checkCuBlasErrors(
        cublasSgemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha,
                    d_B, N, d_A, K, &beta, d_C, N));

    checkCudaErrors(cudaMemcpy(h_C1, d_C, CSIZE(float), cudaMemcpyDeviceToHost));

    showMatrix(h_C, M, N, "Matrix C1");
    showMatrix(h_C1, M, N, "Matrix C2");

    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));

    delete[] h_A;
    printf("Ok A\n");
    delete[] h_B;
    printf("Ok B\n");
    delete[] h_C;
    printf("Ok C\n");
    delete[] h_C1;
    printf("Ok C1\n");
}