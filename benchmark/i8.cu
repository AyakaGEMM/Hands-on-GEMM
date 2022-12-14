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

extern void i8gemm(int, int, int, int8_t *, int8_t *, int32_t *, int32_t, int32_t);

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        printf("usage: ./main [MAX_TEST_SIZE]\n");
        exit(0);
    }
    std::vector<int> test_sizes;
    size_t nmax = atoi(argv[1]);
    bool miss_align = true, ignore_error = false;
    if (argc > 2)
        miss_align = atoi(argv[2]) == 1;
    if (argc > 3)
        ignore_error = atoi(argv[3]) == 1;
    for (int i = 256; i <= nmax + 255; i += 256)
    {
        if (miss_align)
        {
            test_sizes.emplace_back(i - 1);
            test_sizes.emplace_back(i + 1);
        }
        test_sizes.emplace_back(i);
    }

    nmax = test_sizes[test_sizes.size() - 1]; // we assume the last element is the largest one

    int8_t *h_A = new int8_t[nmax * nmax];
    int8_t *h_B = new int8_t[nmax * nmax];
    int32_t *h_C = new int32_t[nmax * nmax];
    int32_t *h_C1 = new int32_t[nmax * nmax];

    int8_t *d_A;
    int8_t *d_B;
    int32_t *d_C;

    checkCudaErrors(cudaMalloc(&d_A, MAXSIZE(float)));
    checkCudaErrors(cudaMalloc(&d_B, MAXSIZE(float)));
    checkCudaErrors(cudaMalloc(&d_C, MAXSIZE(float)));

    cublasHandle_t blas_handle;
    checkCuBlasErrors(cublasCreate(&blas_handle));

    for (const auto &i : test_sizes)
    {
        size_t M = i;
        size_t K = i;
        size_t N = i;

        printf("\nSize M: %u, N: %u, K: %u\n", M, N, K);

        double msecPerMatrixMul[3] = {0, 0};
        double gigaFlops[3] = {0, 0};
        double flopsPerMatrixMul = 2.0 * M * N * K;

        int32_t alpha = 1;
        int32_t beta = 0;

        // 生成A的数据
        genRandomMatrix(h_A, M, K);
        genRandomMatrix(h_B, K, N);
        genRandomMatrix(h_C, M, N);
        copyMatrix(h_C1, h_C, M, N);

        checkCudaErrors(cudaMemcpy(d_A, h_A, ASIZE(int8_t), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_B, h_B, BSIZE(int8_t), cudaMemcpyHostToDevice));

        cudaEvent_t start, stop;
        checkCudaErrors(cudaEventCreate(&start));
        checkCudaErrors(cudaEventCreate(&stop));
        float msecTotal = 0;
        int nIter = 10;

        checkCudaErrors(cudaMemcpy(d_C, h_C, CSIZE(int32_t), cudaMemcpyHostToDevice));

        i8gemm(M, N, K, d_A, d_B, d_C, alpha, beta);

        checkCudaErrors(cudaEventRecord(start));
        for (int run = 0; run < nIter; run++)
        {
            i8gemm(M, N, K, d_A, d_B, d_C, alpha, beta);
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

        // cublas
        checkCudaErrors(cudaMemcpy(d_C, h_C1, CSIZE(int32_t), cudaMemcpyHostToDevice));
        // warmup here (not sure whether we need this or not)
        checkCuBlasErrors(
            cublasGemmEx(blas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                         N, M, K,
                         &alpha,
                         d_B, CUDA_R_8I, N,
                         d_A, CUDA_R_8I, K,
                         &beta,
                         d_C, CUDA_R_32I, N,
                         CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        checkCudaErrors(cudaEventRecord(start));
        for (int run = 0; run < nIter; run++)
        {
            checkCuBlasErrors(
                cublasGemmEx(blas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                             N, M, K,
                             &alpha,
                             d_B, CUDA_R_8I, N,
                             d_A, CUDA_R_8I, K,
                             &beta,
                             d_C, CUDA_R_32I, N,
                             CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
        checkCudaErrors(cudaEventRecord(stop));
        checkCudaErrors(cudaEventSynchronize(stop));
        checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

        msecPerMatrixMul[1] = msecTotal / nIter;
        gigaFlops[1] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[1] / 1000.0f);
        printf("CuBlas Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
               gigaFlops[1],
               msecPerMatrixMul[1],
               flopsPerMatrixMul);

        half al = 1, be = 0;

        checkCuBlasErrors(
            cublasGemmEx(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                         N, M, K,
                         &alpha,
                         d_B, CUDA_R_8I, N,
                         d_A, CUDA_R_8I, K,
                         &beta,
                         d_C, CUDA_R_32I, N,
                         CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        checkCudaErrors(cudaEventRecord(start));
        for (int run = 0; run < nIter; run++)
        {
            checkCuBlasErrors(
                cublasHgemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            N, M, K,
                            &al,
                            reinterpret_cast<half *>(d_B), N,
                            reinterpret_cast<half *>(d_A), K,
                            &be,
                            reinterpret_cast<half *>(d_C), N));
        }
        checkCudaErrors(cudaEventRecord(stop));
        checkCudaErrors(cudaEventSynchronize(stop));
        checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

        msecPerMatrixMul[2] = msecTotal / nIter;
        gigaFlops[2] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[2] / 1000.0f);
        printf("CuBlas fp16 Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
               gigaFlops[2],
               msecPerMatrixMul[2],
               flopsPerMatrixMul);

        if (!ignore_error)
        {
            checkCudaErrors(cudaMemcpy(d_C, h_C1, CSIZE(int32_t), cudaMemcpyHostToDevice));
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
            transposeMatrix(h_B, N, K);
            checkCudaErrors(cudaMemcpy(d_B, h_B, BSIZE(int8_t), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d_C, h_C, CSIZE(int32_t), cudaMemcpyHostToDevice));
            i8gemm(M, N, K, d_A, d_B, d_C, alpha, beta);
            checkCudaErrors(cudaMemcpy(h_C, d_C, CSIZE(int32_t), cudaMemcpyDeviceToHost));

            for (int i = 0; i < M * N; i++)
            {
                int err = h_C1[i] - h_C[i];
                if (err != 0)
                {
                    printf("Error! Matrix[%d]=%d, ref=%d\n",
                           i, h_C[i], h_C1[i]);
                    exit(1);
                }
            }
            printf("No Error\n");
        }
        else
        {
            printf("Ignore the error.\n");
        }

        printf("ratio (int8) = %f\n", gigaFlops[0] / gigaFlops[1]);
        printf("ratio (fp16) = %f\n", gigaFlops[0] / gigaFlops[2]);
    }
    cublasDestroy(blas_handle);

    // Free Memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C1;
}