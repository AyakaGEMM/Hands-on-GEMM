#include <cstdlib>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <mma.h>
using namespace nvcuda;
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 128;
constexpr int BLOCK_K = 128;
#include <iostream>

__global__ void i8gemm128x128(const int8_t *A, const int8_t *B, int32_t *C,
                              int M, int N, int K, const int32_t alpha, const int32_t beta)
{
    const int lda = K;
    const int ldb = K;
    const int ldc = N;

    constexpr int sharedLda = 16;
    constexpr int sharedLdb = 16;

    const size_t baseIdx = threadIdx.x;

    const auto warpM = (baseIdx / 32) / 4;
    const auto warpN = (baseIdx / 32) % 4;
    const auto laneId = baseIdx % 32;
    const auto warpId = baseIdx / 32;

    const auto baseA = A + blockIdx.x * 128 * lda;
    const auto baseB = B + blockIdx.y * 128 * ldb;
    const auto baseC = C + blockIdx.x * 128 * ldc + blockIdx.y * 128 + (warpId / 4) * 32 * ldc + (warpId & 3) * 32;

    __shared__ int8_t shared_mem[128 * sharedLda * 2];
    auto sharedA = shared_mem;
    auto sharedB = shared_mem + 128 * sharedLda;

    int32_t frag_c[16][2] = {}, frag_d[16][2] = {};
    int32_t frag_a[4], frag_b[4];

#pragma unroll
    for (int k = 0; k < K; k += WMMA_K)
    {
        // Do 32x32x16 (mnk) mma at a time.
        *reinterpret_cast<int32_t *>(&sharedA[(baseIdx / 4) * sharedLda + ((baseIdx & 3) << 2)]) = *reinterpret_cast<const int32_t *>(&baseA[(baseIdx / 4) * lda + ((baseIdx & 3) << 2) + k]);

        // Need transpose here, I leave it here for now.
        *reinterpret_cast<int32_t *>(&sharedB[(baseIdx / 4) * sharedLdb + ((baseIdx & 3) << 2)]) = *reinterpret_cast<const int32_t *>(&baseB[(baseIdx / 4) * ldb + ((baseIdx & 3) << 2) + k]);

        __syncthreads();
        // Load matrix in 4 stages, could try warp shuff and overlap in the future.
#pragma unroll
        for (int i = 0; i < 4; i++) // 8 byte load.
        {
            frag_a[i] = *reinterpret_cast<const int32_t *>(&sharedA[(laneId << 2) + i * 32 * 4 + (warpId / 4) * 32 * 16]);
            frag_b[i] = *reinterpret_cast<const int32_t *>(&sharedB[(laneId << 2) + i * 32 * 4 + (warpId & 3) * 32 * 16]);
        }

        __syncwarp();
        // Do mma.
#pragma unroll
        for (int i = 0; i < 16; i++)
        {
            asm volatile( // Do 8x8x16=1024 int8 fmma at a time.
                "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 \
                {%0, %1}, \
                {%2}, {%3}, \
                {%0, %1};"
                : "+r"(frag_c[i][0]), "+r"(frag_c[i][1])
                : "r"(frag_a[i >> 2]), "r"(frag_b[i & 3])); // With an implicit __syncwarp() here.
        }

        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < 16; i++)
    {
        *reinterpret_cast<int64_t *>(frag_d[i]) = *reinterpret_cast<int64_t *>(&baseC[((i >> 2) * 8 + (laneId / 4)) * ldc + (i & 3) * 8 + ((laneId & 3) << 1)]); // I'm the reinterpret_cast master!
        frag_d[i][0] = frag_d[i][0] * beta + frag_c[i][0] * alpha;
        frag_d[i][1] = frag_d[i][1] * beta + frag_c[i][1] * alpha;
        *reinterpret_cast<int64_t *>(&baseC[((i >> 2) * 8 + (laneId / 4)) * ldc + (i & 3) * 8 + ((laneId & 3) << 1)]) = *reinterpret_cast<int64_t *>(frag_d[i]);
    }
}

void i8gemm(int M, int N, int K, int8_t *a, int8_t *b, int32_t *c, int32_t alpha, int32_t beta)
{
    dim3 threadsPerBlock(512);
    dim3 numBlocks((M + BLOCK_M - 1) / BLOCK_M, (N + BLOCK_N - 1) / BLOCK_N);
    i8gemm128x128<<<numBlocks, threadsPerBlock>>>(a, b, c, M, N, K, alpha, beta);
}