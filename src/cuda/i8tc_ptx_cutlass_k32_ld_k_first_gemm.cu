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
constexpr int BLOCK_K = 32;
#include <iostream>

__global__ void i8gemm256x128x32(const int8_t *A, const int8_t *B, int32_t *C,
                                 int M, int N, int K, const int32_t alpha, const int32_t beta)
{
    const int lda = K;
    const int ldb = K;
    const int ldc = N;

    constexpr int sharedLda = BLOCK_K / 2;
    constexpr int sharedLdb = BLOCK_K / 2;

    constexpr int offset = 4;

    const size_t baseIdx = threadIdx.x;

    const auto warpM = (baseIdx / 32) / 4;
    const auto warpN = (baseIdx / 32) % 4;
    const auto laneId = baseIdx % 32;
    const auto warpId = baseIdx / 32;

    const auto baseA = A + blockIdx.x * BLOCK_M * lda;
    const auto baseB = B + blockIdx.y * BLOCK_N * ldb;
    const auto baseC = C + blockIdx.x * BLOCK_M * ldc + blockIdx.y * BLOCK_N + (warpId / 2) * 64 * ldc + (warpId & 1) * 64;

    __shared__ int8_t shared_mem[BLOCK_M * BLOCK_K + offset * sharedLda + BLOCK_N * BLOCK_K + offset * sharedLdb];
    auto sharedA = shared_mem;
    auto sharedB = shared_mem + BLOCK_M * BLOCK_K + offset * sharedLda;

    int32_t frag_c[64][2] = {}; // Initialize to 0.
    int32_t frag_a[8 * 2], frag_b[8 * 2];

    for (int k = 0; k < K; k += BLOCK_K)
    {
        using copy_t = int32_t;
        // Do 64x64x64 (mnk) mma at a time according to cutlass.

#pragma unroll
        for (int i = 0; i < 8; i++)
            frag_a[i] = *reinterpret_cast<const copy_t *>(&baseA[(baseIdx / 8 + 32 * i) * lda + (baseIdx % 8) * 4 + k]);

#pragma unroll
        for (int i = 0; i < 4; i++)
            frag_b[i] = *reinterpret_cast<const copy_t *>(&baseB[(baseIdx / 8 + 32 * i) * ldb + (baseIdx % 8) * 4 + k]);

#pragma unroll
        for (int i = 0; i < 8; i++)
            *reinterpret_cast<copy_t *>(&sharedA[(baseIdx / 8 + 32 * i + ((baseIdx >> 2) & 1) * (BLOCK_M + offset)) * sharedLda + (baseIdx % 4) * 4]) = frag_a[i];
#pragma unroll
        for (int i = 0; i < 4; i++)
            *reinterpret_cast<copy_t *>(&sharedB[(baseIdx / 8 + 32 * i + ((baseIdx >> 2) & 1) * (BLOCK_N + offset)) * sharedLdb + (baseIdx % 4) * 4]) = frag_b[i];

        __syncthreads();
        // Load matrix in 4 stages, could try warp shuff and overlap in the future.

#pragma unroll
        for (int i = 0; i < 2; i++)
        {
            auto ldA = __cvta_generic_to_shared(&sharedA[((warpId / 2) * 64 + laneId + (BLOCK_M + offset) * i) * sharedLda]);
            auto ldB = __cvta_generic_to_shared(&sharedB[((warpId & 1) * 64 + laneId + (BLOCK_N + offset) * i) * sharedLdb]);

            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%8];"
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%4 ,%5, %6, %7}, [%9];"
                : "=r"(frag_a[0 + i * 8]), "=r"(frag_a[1 + i * 8]), "=r"(frag_a[2 + i * 8]), "=r"(frag_a[3 + i * 8]), "=r"(frag_b[0 + i * 8]), "=r"(frag_b[1 + i * 8]), "=r"(frag_b[2 + i * 8]), "=r"(frag_b[3 + i * 8])
                : "l"(ldA), "l"(ldB));

            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%8];"
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%4 ,%5, %6, %7}, [%9];"
                : "=r"(frag_a[4 + i * 8]), "=r"(frag_a[5 + i * 8]), "=r"(frag_a[6 + i * 8]), "=r"(frag_a[7 + i * 8]), "=r"(frag_b[4 + i * 8]), "=r"(frag_b[5 + i * 8]), "=r"(frag_b[6 + i * 8]), "=r"(frag_b[7 + i * 8])
                : "l"(ldA + 32 * sharedLda * sizeof(int8_t)), "l"(ldB + 32 * sharedLdb * sizeof(int8_t)));
        }

#pragma unroll
        for (int ik = 0; ik < 2; ik++)
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
                        : "r"(frag_a[im + ik * 8]), "r"(frag_b[in + ik * 8])); // With an implicit __syncwarp() here.
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
    i8gemm256x128x32<<<numBlocks, threadsPerBlock>>>(a, b, c, M, N, K, alpha, beta);
}