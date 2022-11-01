#include <cstdlib>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <mma.h>
using namespace nvcuda;
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int WARP_M = 128;
constexpr int WARP_N = 128;
constexpr int WARP_K = 128;
#include <iostream>

__global__ void i8gemm128x128(const int8_t *A, const int8_t *B, int32_t *C,
                              int M, int N, int K, const int32_t alpha, const int32_t beta)
{
    const int lda = K;
    const int ldb = K;
    const int ldc = N;

    constexpr int sharedLda = 48;
    constexpr int sharedLdb = 48;

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
    int32_t frag_a[8], frag_b[8];

#pragma unroll
    for (int k = 0; k < K; k += WMMA_K * 2)
    {
        // Do 32x32x32 (mnk) mma at a time.
        *reinterpret_cast<int32_t *>(&sharedA[(baseIdx / 8) * sharedLda + ((baseIdx & 7) << 2)]) = *reinterpret_cast<const int32_t *>(&baseA[(baseIdx / 8) * lda + ((baseIdx & 7) << 2) + k]);
        *reinterpret_cast<int32_t *>(&sharedA[(baseIdx / 8 + 64) * sharedLda + ((baseIdx & 7) << 2)]) = *reinterpret_cast<const int32_t *>(&baseA[(baseIdx / 8 + 64) * lda + ((baseIdx & 7) << 2) + k]);

        // Need transpose here, I leave it here for now.
        *reinterpret_cast<int32_t *>(&sharedB[(baseIdx / 8) * sharedLdb + ((baseIdx & 7) << 2)]) = *reinterpret_cast<const int32_t *>(&baseB[(baseIdx / 8) * ldb + ((baseIdx & 7) << 2) + k]);
        *reinterpret_cast<int32_t *>(&sharedB[(baseIdx / 8 + 64) * sharedLdb + ((baseIdx & 7) << 2)]) = *reinterpret_cast<const int32_t *>(&baseB[(baseIdx / 8 + 64) * ldb + ((baseIdx & 7) << 2) + k]);

        __syncthreads();
        // Load matrix in 4 stages, could try warp shuff and overlap in the future.
#pragma unroll
        for (int i = 0; i < 4; i++) // 8 byte load.
        {
            frag_a[i * 2] = *reinterpret_cast<const int32_t *>(&sharedA[(laneId % 4) * 4 + ((warpId / 4) * 32 + laneId / 4 + i * 8) * sharedLda]);
            frag_a[i * 2 + 1] = *reinterpret_cast<const int32_t *>(&sharedA[(laneId % 4) * 4 + 16 + ((warpId / 4) * 32 + laneId / 4 + i * 8) * sharedLda]);
            frag_b[i * 2] = *reinterpret_cast<const int32_t *>(&sharedB[(laneId % 4) * 4 + ((warpId & 3) * 32 + laneId / 4 + i * 8) * sharedLdb]);
            frag_b[i * 2 + 1] = *reinterpret_cast<const int32_t *>(&sharedB[(laneId % 4) * 4 + 16 + ((warpId & 3) * 32 + laneId / 4 + i * 8) * sharedLdb]);
        }

        __syncwarp();
        // Do mma.
#pragma unroll
        for (int i = 0; i < 16; i++)
        {
            asm( // Do 8x8x32=2048 int8 fmma at a time.
                "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 \
                {%0, %1}, \
                {%2}, {%3}, \
                {%0, %1};"
                "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 \
                {%0, %1}, \
                {%4}, {%5}, \
                {%0, %1};"
                : "+r"(frag_c[i][0]), "+r"(frag_c[i][1])
                : "r"(frag_a[(i >> 2) << 1]), "r"(frag_b[(i & 3) << 1]), "r"(frag_a[((i >> 2) << 1) + 1]), "r"(frag_b[((i & 3) << 1) + 1])); // With an implicit __syncwarp() here.
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
    dim3 numBlocks((M + WARP_M - 1) / WARP_M, (N + WARP_N - 1) / WARP_N);
    i8gemm128x128<<<numBlocks, threadsPerBlock>>>(a, b, c, M, N, K, alpha, beta);
}