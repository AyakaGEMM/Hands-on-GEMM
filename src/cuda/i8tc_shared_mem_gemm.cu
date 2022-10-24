#include <cstdlib>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
//#define __CUDA_ARCH__ 750
#include <mma.h>
using namespace nvcuda;
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;
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

__global__ void i8gemm128x128(const int8_t *A, const int8_t *B, int32_t *C,
                              int M, int N, int K, int32_t alpha, int32_t beta)
{
    const int lda = K;
    const int ldb = N;
    const int ldc = N;

    constexpr int sharedLda = 32;
    constexpr int sharedLdb = 128;

    const size_t baseIdx = threadIdx.x;

    const auto warpM = (baseIdx / 32) / 4;
    const auto warpN = (baseIdx / 32) % 4;
    const auto laneId = baseIdx % 32;
    const auto warpId = baseIdx / 32;
    const auto rowA = blockIdx.x * 128 + warpM * WMMA_M;
    const auto colB = blockIdx.y * 128 + warpN * WMMA_N;

    const auto baseA = A + blockIdx.x * 128 * lda;
    const auto baseB = B + blockIdx.y * 128;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::row_major> b_frag[2];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> acc_frag[4];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> c_frag[4];

    __shared__ int8_t shared_mem[128 * 32 * 2];

    auto sharedA = shared_mem;
    auto sharedB = shared_mem + 128 * 32;

#pragma unroll
    for (int i = 0; i < 4; i++)
        wmma::fill_fragment(acc_frag[i], 0);

    for (int i = 0; i < K; i += WMMA_K * 2)
    {
#pragma unroll
        for (int count = 0; count < 8; count++)
        {
            sharedA[(warpId + count * 16) * sharedLda + laneId] = baseA[(warpId + count * 16) * lda + laneId + i];
            sharedB[(baseIdx / 128 + count * 4) * sharedLdb + baseIdx % 128] = baseB[(baseIdx / 128 + i + count * 4) * ldb + baseIdx % 128];
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < 2; j++)
        {
            wmma::load_matrix_sync(a_frag[0], sharedA + warpM * WMMA_M * sharedLda + j * WMMA_K, sharedLda);
            wmma::load_matrix_sync(a_frag[1], sharedA + (64 + warpM * WMMA_M) * sharedLda + j * WMMA_K, sharedLda);
            wmma::load_matrix_sync(b_frag[0], sharedB + warpN * WMMA_N + j * WMMA_K * sharedLdb, sharedLdb);
            wmma::load_matrix_sync(b_frag[1], sharedB + warpN * WMMA_N + 64 + j * WMMA_K * sharedLdb, sharedLdb);

            wmma::mma_sync(acc_frag[0], a_frag[0], b_frag[0], acc_frag[0]);
            wmma::mma_sync(acc_frag[1], a_frag[0], b_frag[1], acc_frag[1]);
            wmma::mma_sync(acc_frag[2], a_frag[1], b_frag[0], acc_frag[2]);
            wmma::mma_sync(acc_frag[3], a_frag[1], b_frag[1], acc_frag[3]);
        }
    }

#pragma unroll
    for (int ii = 0; ii < 4; ii++)
    {
        wmma::load_matrix_sync(c_frag[ii], C + (rowA + 64 * (ii >> 1)) * ldc + colB + 64 * (ii & 1), ldc, wmma::mem_row_major);
#pragma unroll
        for (int i = 0; i < c_frag[ii].num_elements; i++)
        {
            c_frag[ii].x[i] = alpha * acc_frag[ii].x[i] + beta * c_frag[ii].x[i];
        }
        wmma::store_matrix_sync(C + (rowA + 64 * (ii >> 1)) * ldc + colB + 64 * (ii & 1), c_frag[ii], ldc, wmma::mem_row_major);
    }
}

void i8gemm(int M, int N, int K, int8_t *a, int8_t *b, int32_t *c, int32_t alpha, int32_t beta)
{
    dim3 threadsPerBlock(512);
    dim3 numBlocks((M + 128 - 1) / 128, (N + 128 - 1) / 128);
    i8gemm128x128<<<numBlocks, threadsPerBlock>>>(a, b, c, M, N, K, alpha, beta);
}