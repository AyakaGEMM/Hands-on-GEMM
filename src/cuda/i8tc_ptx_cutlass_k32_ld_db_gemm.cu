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

    constexpr auto sharedASize = BLOCK_M * BLOCK_K + offset * sharedLda;
    constexpr auto sharedBSize = BLOCK_N * BLOCK_K + offset * sharedLdb;

    __shared__ int8_t shared_mem[(sharedASize + sharedBSize) * 2];

    auto sharedA = shared_mem;
    auto sharedB = shared_mem + sharedASize * 2;

    int32_t frag_c[64][2] = {}; // Initialize to 0.
    int32_t frag_a[8 * 2], frag_b[8 * 2];
    int32_t preA[8], preB[4];

    int loadK = BLOCK_K;
    int stage = 0;

    using copy_t = int32_t;

    // Preload
#pragma unroll
    for (int i = 0; i < 8; i++)
        preA[i] = *reinterpret_cast<const copy_t *>(&baseA[(baseIdx / 8 + 32 * i) * lda + (baseIdx % 8) * 4]);

#pragma unroll
    for (int i = 0; i < 4; i++)
        preB[i] = *reinterpret_cast<const copy_t *>(&baseB[(baseIdx / 8 + 32 * i) * ldb + (baseIdx % 8) * 4]);

#pragma unroll
    for (int i = 0; i < 8; i++)
        *reinterpret_cast<copy_t *>(&sharedA[(baseIdx / 8 + 32 * i + ((baseIdx >> 2) & 1) * (BLOCK_M + offset)) * sharedLda + (baseIdx % 4) * 4]) = preA[i];
#pragma unroll
    for (int i = 0; i < 4; i++)
        *reinterpret_cast<copy_t *>(&sharedB[(baseIdx / 8 + 32 * i + ((baseIdx >> 2) & 1) * (BLOCK_N + offset)) * sharedLdb + (baseIdx % 4) * 4]) = preB[i];

    __syncthreads();

#pragma unroll
    for (int i = 0; i < 8; i++)
    {
        auto ldA = __cvta_generic_to_shared(&sharedA[((warpId / 2) * 64 + i * 8 + laneId % 8) * sharedLda + stage * sharedASize]);
        auto ldB = __cvta_generic_to_shared(&sharedB[((warpId & 1) * 64 + i * 8 + laneId % 8) * sharedLdb + stage * sharedBSize]);
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%2];"
            "ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%1}, [%3];"
            : "=r"(frag_a[i]), "=r"(frag_b[i])
            : "l"(ldA), "l"(ldB));
    }

#pragma unroll
    for (int i = 0; i < 8; i++)
    {
        auto ldA = __cvta_generic_to_shared(&sharedA[((warpId / 2) * 64 + i * 8 + laneId % 8) * sharedLda + stage * sharedASize]);
        auto ldB = __cvta_generic_to_shared(&sharedB[((warpId & 1) * 64 + i * 8 + laneId % 8) * sharedLdb + stage * sharedBSize]);
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%2];"
            "ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%1}, [%3];"
            : "=r"(frag_a[i]), "=r"(frag_b[i])
            : "l"(ldA), "l"(ldB));
    }

    for (int k = 0; k < K; k += BLOCK_K)
    {
        if (loadK < K)
        {
            // Do 64x64x64 (mnk) mma at a time according to cutlass.
#pragma unroll
            for (int i = 0; i < 8; i++)
                preA[i] = *reinterpret_cast<const copy_t *>(&baseA[(baseIdx / 8 + 32 * i) * lda + (baseIdx % 8) * 4 + loadK]);

#pragma unroll
            for (int i = 0; i < 4; i++)
                preB[i] = *reinterpret_cast<const copy_t *>(&baseB[(baseIdx / 8 + 32 * i) * ldb + (baseIdx % 8) * 4 + loadK]);
            loadK += BLOCK_K;
        }

#pragma unroll
        for (int i = 0; i < 8; i++)
        {
            auto ldA = __cvta_generic_to_shared(&sharedA[((warpId / 2) * 64 + i * 8 + laneId % 8 + BLOCK_M + offset) * sharedLda + stage * sharedASize]);
            auto ldB = __cvta_generic_to_shared(&sharedB[((warpId & 1) * 64 + i * 8 + laneId % 8 + BLOCK_N + offset) * sharedLdb + stage * sharedBSize]);
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%2];"
                "ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%1}, [%3];"
                : "=r"(frag_a[i + 8]), "=r"(frag_b[i + 8])
                : "l"(ldA), "l"(ldB));
        }

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
                    : "r"(frag_a[im]), "r"(frag_b[in])); // With an implicit __syncwarp() here.
            }
        }

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
                    : "r"(frag_a[im + 8]), "r"(frag_b[in + 8])); // With an implicit __syncwarp() here.
            }
        }

        stage ^= 1;

#pragma unroll
        for (int i = 0; i < 8; i++)
            *reinterpret_cast<copy_t *>(&sharedA[(baseIdx / 8 + 32 * i + ((baseIdx >> 2) & 1) * (BLOCK_M + offset)) * sharedLda + (baseIdx % 4) * 4 + stage * sharedASize]) = preA[i];
#pragma unroll
        for (int i = 0; i < 4; i++)
            *reinterpret_cast<copy_t *>(&sharedB[(baseIdx / 8 + 32 * i + ((baseIdx >> 2) & 1) * (BLOCK_N + offset)) * sharedLdb + (baseIdx % 4) * 4 + stage * sharedBSize]) = preB[i];

        __syncthreads();

#pragma unroll
        for (int i = 0; i < 8; i++)
        {
            auto ldA = __cvta_generic_to_shared(&sharedA[((warpId / 2) * 64 + i * 8 + laneId % 8) * sharedLda + stage * sharedASize]);
            auto ldB = __cvta_generic_to_shared(&sharedB[((warpId & 1) * 64 + i * 8 + laneId % 8) * sharedLdb + stage * sharedBSize]);
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%2];"
                "ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%1}, [%3];"
                : "=r"(frag_a[i]), "=r"(frag_b[i])
                : "l"(ldA), "l"(ldB));
        }
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
            *reinterpret_cast<int64_t *>(&baseC[(im * 8 + laneId / 4) * ldc + in * 8 + (laneId & 3) * 2]) = *reinterpret_cast<int64_t *>(&frag_c[idx]);
        }
    }
}

void i8gemm(int M, int N, int K, int8_t *a, int8_t *b, int32_t *c, int32_t alpha, int32_t beta)
{
    dim3 threadsPerBlock(256);
    dim3 numBlocks((M + BLOCK_M - 1) / BLOCK_M, (N + BLOCK_N - 1) / BLOCK_N);
    i8gemm256x128x32<<<numBlocks, threadsPerBlock>>>(a, b, c, M, N, K, alpha, beta);
}