#include <cstdlib>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <mma.h>
using namespace nvcuda;
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int BLOCK_M = 256;
constexpr int BLOCK_N = 128;
constexpr int BLOCK_K = 64;
#include <iostream>

__global__ void i8gemm256x128x64(const int8_t *A, const int8_t *B, int32_t *C,
                                 int M, int N, int K, const int32_t alpha, const int32_t beta)
{
    const int lda = K;
    const int ldb = K;
    const int ldc = N;

    constexpr int sharedLda = 64;
    constexpr int sharedLdb = 64;

    const size_t baseIdx = threadIdx.x;

    const auto warpM = (baseIdx / 32) / 4;
    const auto warpN = (baseIdx / 32) % 4;
    const auto laneId = baseIdx % 32;
    const auto warpId = baseIdx / 32;

    const auto baseA = A + blockIdx.x * BLOCK_M * lda;
    const auto baseB = B + blockIdx.y * BLOCK_N * ldb;
    const auto baseC = C + blockIdx.x * BLOCK_M * ldc + blockIdx.y * BLOCK_N + (warpId / 2) * 64 * ldc + (warpId & 1) * 64;

    __shared__ int8_t shared_mem[BLOCK_M * sharedLda + BLOCK_N * sharedLdb];
    auto sharedA = shared_mem;
    auto sharedB = shared_mem + BLOCK_M * sharedLda;

    int32_t frag_c[64][2] = {};         // Initialize to 0.
    int32_t frag_a[8][4], frag_b[8][4]; // Use streaming read to release the reg pressure.

#pragma unroll
    for (int k = 0; k < K; k += BLOCK_K)
    {
// Do 64x64x64 (mnk) mma at a time according to cutlass.
#pragma unroll
        for (int i = 0; i < 16; i++)
        {
            *reinterpret_cast<int32_t *>(&sharedA[(baseIdx / 16 + i * 16) * sharedLda + (baseIdx % 16) * 4]) = *reinterpret_cast<const int32_t *>(&baseA[(baseIdx / 16 + i * 16) * lda + (baseIdx % 16) * 4 + k]);
        }

// Need transpose here, I leave it here for now.
#pragma unroll
        for (int i = 0; i < 8; i++)
        {
            *reinterpret_cast<int32_t *>(&sharedB[(baseIdx / 16 + i * 16) * sharedLdb + (baseIdx % 16) * 4]) = *reinterpret_cast<const int32_t *>(&baseB[(baseIdx / 16 + i * 16) * ldb + (baseIdx % 16) * 4 + k]);
        }

        __syncthreads();
        // Load matrix in 4 stages, could try warp shuff and overlap in the future.
        for (int i = 0; i < 8; i++)
        {
            auto ldA = __cvta_generic_to_shared(&sharedA[((warpId / 2) * 64 + i * 8 + laneId % 8) * sharedLda + (laneId / 8) * 16]);
            auto ldB = __cvta_generic_to_shared(&sharedB[((warpId % 2) * 64 + i * 8 + laneId % 8) * sharedLdb + (laneId / 8) * 16]);
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%8];"
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%4, %5, %6, %7}, [%9];"
                : "=r"(frag_a[i][0]), "=r"(frag_a[i][1]), "=r"(frag_a[i][2]), "=r"(frag_a[i][3]), "=r"(frag_b[i][0]), "=r"(frag_b[i][1]), "=r"(frag_b[i][2]), "=r"(frag_b[i][3])
                : "l"(ldA), "l"(ldB));
        }

#pragma unroll
        for (int ik = 0; ik < 4; ik++)
        {
#pragma unroll
            for (int im = 0; im < 8; im++)
            {
#pragma unroll
                for (int in = 0; in < 8; in++)
                {
                    asm volatile(
                        "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 \
                        {%0, %1}, \
                        {%2}, {%3}, \
                        {%0, %1};"
                        : "+r"(frag_c[im * 8 + in][0]), "+r"(frag_c[im * 8 + in][1])
                        : "r"(frag_a[im][ik]), "r"(frag_b[in][ik])); // With an implicit __syncwarp() here.
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int im = 0; im < 8; im++)
    {
#pragma unroll
        for (int in = 0; in < 8; in++)
        {
            auto idx = im * 8 + in;
            int32_t frag_d[2];
            *reinterpret_cast<int64_t *>(frag_d) = *reinterpret_cast<int64_t *>(&baseC[(im * 8 + laneId / 4) * ldc + in * 8 + (laneId & 3) * 2]); // I'm the reinterpret_cast master!
            frag_d[0] = frag_c[idx][0] * alpha + frag_d[0] * beta;
            frag_d[1] = frag_c[idx][1] * alpha + frag_d[1] * beta;
            *reinterpret_cast<int64_t *>(&baseC[(im * 8 + laneId / 4) * ldc + in * 8 + (laneId & 3) * 2]) = *reinterpret_cast<int64_t *>(frag_d);
        }
    }
}

void i8gemm(int M, int N, int K, int8_t *a, int8_t *b, int32_t *c, int32_t alpha, int32_t beta)
{
    dim3 threadsPerBlock(256);
    dim3 numBlocks((M + BLOCK_M - 1) / BLOCK_M, (N + BLOCK_N - 1) / BLOCK_N);
    i8gemm256x128x64<<<numBlocks, threadsPerBlock>>>(a, b, c, M, N, K, alpha, beta);
}