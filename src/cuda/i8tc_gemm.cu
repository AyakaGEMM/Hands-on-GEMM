#include <cstdlib>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <mma.h>
using namespace nvcuda;
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;
#ifndef __CUDACC__
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
void __syncthreads(); // workaround __syncthreads warning
void __syncwarp();
#endif
#include <iostream>
constexpr size_t BLOCK_SIZE = 16; // we assume that every block has equal blockDim.x and blockDim.y
constexpr size_t BLOCK_M = 128;   // These const values decide how many thing a thread compute and the amount of shared memory to allocate.
constexpr size_t BLOCK_N = 128;
constexpr size_t BLOCK_K = 8; // don't set 64 here, it will cause bank conflict and lower occupancy.
constexpr size_t BLOCK_M_COMPUTE = BLOCK_M / BLOCK_SIZE;
constexpr size_t BLOCK_N_COMPUTE = BLOCK_N / BLOCK_SIZE;

constexpr int shared_memory_A = BLOCK_M * BLOCK_K;
constexpr int shared_memory_B = BLOCK_N * BLOCK_K;
constexpr int shared_memory_element = shared_memory_A + shared_memory_B;
constexpr int shared_memory_size = shared_memory_element * sizeof(float); // shared memory to use.

//__device__ __forceinline__ void sts32(const int8_t &reg0, const int8_t &reg1,
//                                      const int8_t &reg2, const int8_t &reg3,
//                                      const int8_t *addr)
//{
//    asm volatile(
//        "st.shared.v4.s8 [%0], {%1, %2, %3, %4};\n"
//        :
//        : "l"(addr), "l"(reg0), "l"(reg1), "l"(reg2), "l"(reg3));
//}
//
//__device__ __forceinline__ void lds32(int8_t &reg0, int8_t &reg1,
//                                      int8_t &reg2, int8_t &reg3,
//                                      const int8_t *addr)
//{
//    asm volatile(
//        "ld.shared.v4.s8 {%0, %1, %2, %3}, [%4];\n"
//        : "=l"(reg0), "=l"(reg1), "=l"(reg2), "=l"(reg3)
//        : "l"(addr));
//}

__global__ void i8gemm64x64(const int8_t *A, const int8_t *B, int32_t *C,
                            int M, int N, int K, int32_t alpha, int32_t beta)
{
    int lda = K;
    int ldb = N;
    int ldc = N;

    const size_t baseIdx = threadIdx.y * blockDim.y + threadIdx.x;

    const auto warpM = (baseIdx / 32) / 4;
    const auto warpN = (baseIdx / 32) % 4;
    const auto rowA = blockIdx.x * 64 + warpM * WMMA_M;
    const auto colB = blockIdx.y * 64 + warpN * WMMA_N;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> c_frag;

    wmma::fill_fragment(acc_frag, 0);

    for (int i = 0; i < K; i += WMMA_K)
    {
        // Bounds checking
        // Load the inputs
        wmma::load_matrix_sync(a_frag, A + rowA * lda + i, lda);
        wmma::load_matrix_sync(b_frag, B + colB + i * ldb, ldb);

        // Perform the matrix multiplication
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    // Load in current value of c, scale by beta, and add to result scaled by alpha
    wmma::load_matrix_sync(c_frag, C + rowA * ldc + colB, ldc, wmma::mem_row_major);
    for (int i = 0; i < c_frag.num_elements; i++)
    {
        c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
    }
    // Store the output
    wmma::store_matrix_sync(C + rowA * ldc + colB, c_frag, ldc, wmma::mem_row_major);
}

void i8gemm(int M, int N, int K, int8_t *a, int8_t *b, int32_t *c, int32_t alpha, int32_t beta)
{
    dim3 threadsPerBlock(512);
    dim3 numBlocks((M + 64 - 1) / 64, (N + 64 - 1) / 64);
    i8gemm64x64<<<numBlocks, threadsPerBlock>>>(a, b, c, M, N, K, alpha, beta);
}