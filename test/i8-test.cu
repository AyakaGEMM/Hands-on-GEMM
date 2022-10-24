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

int MAX;

extern void i8gemm(int, int, int, int8_t *, int8_t *, int32_t *, int32_t, int32_t);

void refgemm(int M, int N, int K, int8_t *a, int8_t *b, int32_t *c, int32_t alpha, int32_t beta)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            c[i * N + j] *= beta;
            for (int k = 0; k < K; k++)
            {
                c[i * N + j] += alpha * a[i * K + k] * b[k * N + j];
            }
        }
    }
}

int main(int argc, char **argv)
{
    // if (argc != 4)
    //{
    //     printf("usage: ./main [M] [K] [N]\n");
    //     exit(0);
    // }
    size_t M = 256;
    size_t N = 256;
    size_t K = 256;

    std::cout << M << " " << N << " " << K << std::endl;

    int8_t *h_A = new int8_t[M * M];
    int8_t *h_B = new int8_t[M * M];
    int32_t *h_C = new int32_t[M * M];
    int32_t *h_C1 = new int32_t[M * M];

    int8_t *d_A;
    int8_t *d_B;
    int32_t *d_C;

    checkCudaErrors(cudaMalloc(&d_A, ASIZE(int8_t)));
    checkCudaErrors(cudaMalloc(&d_B, BSIZE(int8_t)));
    checkCudaErrors(cudaMalloc(&d_C, CSIZE(int32_t)));

    std::cout << d_C << std::endl;

    double msecPerMatrixMul[2] = {0, 0};
    double gigaFlops[2] = {0, 0};
    double flopsPerMatrixMul = 2.0 * M * N * K;

    constexpr int32_t alpha = 1;
    constexpr int32_t beta = 0;

    // 生成A的数据
    genRandomMatrix(h_A, M, K);
    genRandomMatrix(h_B, K, N);
    genRandomMatrix(h_C, M, N);
    copyMatrix(h_C1, h_C, M, N);

    checkCudaErrors(cudaMemcpy(d_A, h_A, ASIZE(int8_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, BSIZE(int8_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_C, h_C, CSIZE(int32_t), cudaMemcpyHostToDevice)); // Free Memory

    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
        {
            h_C[i * N + j] = beta * h_C[i * N + j];
            for (int k = 0; k < K; k++)
                h_C[i * N + j] += alpha * h_A[i * K + k] * h_B[k * N + j];
        }
    // showMatrix(h_C, M, N, "Matrix Ref");
    copyMatrix(h_C, h_C1, M, N);

    cudaEvent_t start, stop;
    cublasHandle_t blas_handle;
    checkCuBlasErrors(cublasCreate(&blas_handle));
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float msecTotal = 0;
    int nIter = 100;

    i8gemm(M, N, K, d_A, d_B, d_C, alpha, beta);
    checkCudaErrors(cudaMemcpy(h_C, d_C, CSIZE(int32_t), cudaMemcpyDeviceToHost));
    checkCuBlasErrors(
        cublasGemmEx(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                     N, M, K,
                     &alpha,
                     d_B, CUDA_R_8I, N,
                     d_A, CUDA_R_8I, K,
                     &beta,
                     d_C, CUDA_R_32I, N,
                     CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    checkCudaErrors(cudaMemcpy(h_C1, d_C, CSIZE(int32_t), cudaMemcpyDeviceToHost));

    showMatrix(h_A, M, K, "Matrix A");
    showMatrix(h_B, K, N, "Matrix B");
    showMatrix(h_C, M, N, "Matrix C1");
    showMatrix(h_C1, M, N, "Matrix C2");

    i8gemm(M, N, K, d_A, d_B, d_C, alpha, beta);
    checkCudaErrors(cudaMemcpy(h_C, d_C, CSIZE(int32_t), cudaMemcpyDeviceToHost));
    showMatrix(h_C, M, N, "Matrix C1");

    refgemm(M, N, K, h_A, h_B, h_C, 1, -1);
    showMatrix(h_C, M, N, "Matrix C3");

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