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

extern void sgemm(int, int, int, float *, float *, float *, cublasHandle_t, float, float);

template <typename T, typename S>
S climp(const T &a)
{
    T min = std::numeric_limits<S>::min(), max = std::numeric_limits<S>::max();
    return std::min(std::max(min, a), max);
}

void intQuantMatrix(float *a, int8_t *b, int mode, float &scale, int32_t &zeroPoint)
{
    float min = std::numeric_limits<float>::max(), max = std::numeric_limits<float>::min();
    for (int i = 0; i < MAX * MAX; i++)
    {
        min = std::min(a[i], min);
        max = std::max(a[i], max);
    }
    scale = (max - min) / 255;
    if (mode == 0)
    {
        for (int j = 0; j < MAX; j++)
        {
            float sum = 0, psum = 0;
            for (int i = 0; i < MAX; i++)
            {
                sum += a[i * MAX + j];
                psum += a[i * MAX + j] * a[i * MAX + j];
            }
            auto mean = sum / MAX;
            auto stdDevs = sqrt(psum / MAX - mean * mean);
            auto range = 14 * stdDevs;
            scale = range / 255;
            zeroPoint = std::nearbyintf(127 - (mean + 7 * stdDevs) / scale);
            for (int i = 0; i < MAX; i++)
            {
                b[i * MAX + j] = climp<float, int8_t>(std::nearbyint((a[i * MAX + j] / scale + zeroPoint)));
            }
        }
    }
    else
    {
        zeroPoint = (127 - max / scale);
        float invScale = 1 / (scale + 1e-8f);

        for (int i = 0; i < MAX * MAX; i++)
        {
            b[i] = climp<float, int8_t>(std::nearbyint((a[i] * invScale + zeroPoint)));
        }
    }
}

void deQuantMatrix(int8_t *a, int8_t *b, int32_t *c, float *c_out, int M, int N, int K, float scale_a, float scale_b, int32_t z_a, int32_t z_b)
{
    float scale = scale_a * scale_b;
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
        {
            int32_t sum_a = 0, sum_b = 0;
            for (int k = 0; k < K; k++)
            {
                sum_a += a[i * K + k];
                sum_b += b[j + k * N];
            }
            c_out[i * N + j] = scale * (c[i * N + j] - z_a * sum_b - z_b * sum_a + K * z_a * z_b);
        }
}

int main(int argc, char **argv)
{
    // if (argc != 4)
    //{
    //     printf("usage: ./main [M] [K] [N]\n");
    //     exit(0);
    // }
    size_t M = 16;
    size_t N = 16;
    size_t K = 16;

    std::cout << M << " " << N << " " << K << std::endl;

    float *h_A = new float[M * K];
    float *h_B = new float[N * K];
    float *h_C = new float[M * N];
    float *h_C1 = new float[M * N];

    int8_t *ia = new int8_t[M * K], *ib = new int8_t[K * N];

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

    float alpha = 1;
    float beta = 0;

    // 生成A的数据
    genRandomMatrix(h_A, M, K);
    genRandomMatrix(h_B, K, N);
    genRandomMatrix(h_C, M, N);
    copyMatrix(h_C1, h_C, M, N);

    float scale;
    int32_t zeroPoint;
    MAX = M;
    intQuantMatrix(h_A, ia, 1, scale, zeroPoint);
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < K; j++)
        {
            std::cout << int16_t(ia[i * N + j]) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << scale << " " << zeroPoint << std::endl;
    intQuantMatrix(h_B, ib, 0, scale, zeroPoint);
    std::cout << std::endl;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < K; j++)
        {
            std::cout << int16_t(ib[i * N + j]) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << scale << " " << zeroPoint << std::endl;
    std::cout << std::endl;

    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int32_t c = 0;
            for (int k = 0; k < K; k++)
                c += ia[i * K + k] * ib[k * N + j];
            std::cout << c << " ";
        }
        std::cout << std::endl;
    }

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
    // showMatrix(h_C, M, N, "Matrix Ref");
    copyMatrix(h_C, h_C1, M, N);

    cudaEvent_t start, stop;
    cublasHandle_t blas_handle;
    checkCuBlasErrors(cublasCreate(&blas_handle));
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float msecTotal = 0;
    int nIter = 100;

    sgemm(M, N, K, d_A, d_B, d_C, blas_handle, alpha, beta);
    checkCudaErrors(cudaMemcpy(h_C, d_C, CSIZE(float), cudaMemcpyDeviceToHost));
    checkCuBlasErrors(
        cublasSgemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha,
                    d_B, N, d_A, K, &beta, d_C, N));

    checkCudaErrors(cudaMemcpy(h_C1, d_C, CSIZE(float), cudaMemcpyDeviceToHost));

    showMatrix(h_A, M, K, "Matrix A");
    showMatrix(h_B, K, N, "Matrix B");
    showMatrix(h_C, M, N, "Matrix C1");
    showMatrix(h_C1, M, N, "Matrix C2");

    sgemm(M, N, K, d_A, d_B, d_C, blas_handle, alpha, beta);
    checkCudaErrors(cudaMemcpy(h_C, d_C, CSIZE(float), cudaMemcpyDeviceToHost));
    showMatrix(h_C, M, N, "Matrix C1");

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