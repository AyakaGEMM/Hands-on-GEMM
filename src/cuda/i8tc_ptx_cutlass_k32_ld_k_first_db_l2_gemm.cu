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
constexpr int MAX_BLOCK_N = 9;
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
    const auto baseBlockIdx = blockIdx.x + gridDim.x * blockIdx.y;

    const auto warpM = (baseIdx / 32) / 4;
    const auto warpN = (baseIdx / 32) % 4;
    const auto laneId = baseIdx % 32;
    const auto warpId = baseIdx / 32;

    const auto totalPanel = (gridDim.x * gridDim.y + MAX_BLOCK_N * gridDim.x - 1) / (MAX_BLOCK_N * gridDim.x);
    const auto totalBlock = gridDim.x * gridDim.y;

    const auto panelIdx = baseBlockIdx / (MAX_BLOCK_N * gridDim.x);
    const auto strideLd = panelIdx + 1 < totalPanel ? MAX_BLOCK_N : (totalBlock - panelIdx * (MAX_BLOCK_N * gridDim.x)) / gridDim.x;

    const auto blockM = (panelIdx & 1) ? gridDim.x - (baseBlockIdx - panelIdx * MAX_BLOCK_N * gridDim.x) / strideLd - 1 : (baseBlockIdx - panelIdx * MAX_BLOCK_N * gridDim.x) / strideLd;
    const auto blockN = (baseBlockIdx - panelIdx * MAX_BLOCK_N * gridDim.x) % strideLd + panelIdx * MAX_BLOCK_N;

    const auto baseA = A + blockM * BLOCK_M * lda;
    const auto baseB = B + blockN * BLOCK_N * ldb;
    const auto baseC = C + blockM * BLOCK_M * ldc + blockN * BLOCK_N + (warpId / 2) * 64 * ldc + (warpId & 1) * 64;

    constexpr auto sharedASize = BLOCK_M * BLOCK_K + offset * sharedLda;
    constexpr auto sharedBSize = BLOCK_N * BLOCK_K + offset * sharedLdb;

    using copy_t = __int128_t;

    __shared__ int8_t shared_mem[(sharedASize + sharedBSize) * 2];
    auto sharedA = shared_mem;
    auto sharedB = shared_mem + sharedASize * 2;

    int32_t frag_c[64][2] = {}; // Initialize to 0.
    int32_t frag_a[8 * 2], frag_b[8 * 2];

    copy_t preA[2], preB;

    int stage = 0;
    int loadK = BLOCK_K;

#pragma unroll
    for (int i = 0; i < 2; i++)
        preA[i] = *reinterpret_cast<const copy_t *>(&baseA[(baseIdx / 2 + 128 * i) * lda + (baseIdx % 2) * 16]);

    preB = *reinterpret_cast<const copy_t *>(&baseB[(baseIdx / 2) * ldb + (baseIdx % 2) * 16]);

#pragma unroll
    for (int i = 0; i < 2; i++)
        *reinterpret_cast<copy_t *>(&sharedA[(baseIdx / 2 + 128 * i + (baseIdx & 1) * (BLOCK_M + offset)) * sharedLda]) = preA[i];

    *reinterpret_cast<copy_t *>(&sharedB[(baseIdx / 2 + (baseIdx & 1) * (BLOCK_N + offset)) * sharedLdb]) = preB;

    __syncthreads();

    {
        auto ldA = __cvta_generic_to_shared(&sharedA[((warpId / 2) * 64 + laneId) * sharedLda + stage * sharedASize]);
        auto ldB = __cvta_generic_to_shared(&sharedB[((warpId & 1) * 64 + laneId) * sharedLdb + stage * sharedBSize]);

        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%8];"
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%4 ,%5, %6, %7}, [%9];"
            : "=r"(frag_a[0]), "=r"(frag_a[1]), "=r"(frag_a[2]), "=r"(frag_a[3]), "=r"(frag_b[0]), "=r"(frag_b[1]), "=r"(frag_b[2]), "=r"(frag_b[3])
            : "l"(ldA), "l"(ldB));

        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%8];"
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%4 ,%5, %6, %7}, [%9];"
            : "=r"(frag_a[4]), "=r"(frag_a[5]), "=r"(frag_a[6]), "=r"(frag_a[7]), "=r"(frag_b[4]), "=r"(frag_b[5]), "=r"(frag_b[6]), "=r"(frag_b[7])
            : "l"(ldA + 32 * sharedLda * sizeof(int8_t)), "l"(ldB + 32 * sharedLdb * sizeof(int8_t)));
    }

    for (int k = 0; k < K; k += BLOCK_K)
    {
        if (loadK < K)
        {
#pragma unroll
            for (int i = 0; i < 2; i++)
                preA[i] = *reinterpret_cast<const copy_t *>(&baseA[(baseIdx / 2 + 128 * i) * lda + (baseIdx % 2) * 16 + loadK]);

            preB = *reinterpret_cast<const copy_t *>(&baseB[(baseIdx / 2) * ldb + (baseIdx % 2) * 16 + loadK]);

            loadK += BLOCK_K;
        }

        auto ldA = __cvta_generic_to_shared(&sharedA[((warpId / 2) * 64 + laneId + BLOCK_M + offset) * sharedLda + stage * sharedASize]);
        auto ldB = __cvta_generic_to_shared(&sharedB[((warpId & 1) * 64 + laneId + BLOCK_N + offset) * sharedLdb + stage * sharedBSize]);

        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%8];"
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%4 ,%5, %6, %7}, [%9];"
            : "=r"(frag_a[0 + 8]), "=r"(frag_a[1 + 8]), "=r"(frag_a[2 + 8]), "=r"(frag_a[3 + 8]), "=r"(frag_b[0 + 8]), "=r"(frag_b[1 + 8]), "=r"(frag_b[2 + 8]), "=r"(frag_b[3 + 8])
            : "l"(ldA), "l"(ldB));

        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%8];"
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%4 ,%5, %6, %7}, [%9];"
            : "=r"(frag_a[4 + 8]), "=r"(frag_a[5 + 8]), "=r"(frag_a[6 + 8]), "=r"(frag_a[7 + 8]), "=r"(frag_b[4 + 8]), "=r"(frag_b[5 + 8]), "=r"(frag_b[6 + 8]), "=r"(frag_b[7 + 8])
            : "l"(ldA + 32 * sharedLda * sizeof(int8_t)), "l"(ldB + 32 * sharedLdb * sizeof(int8_t)));

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

        stage ^= 1;
#pragma unroll
        for (int i = 0; i < 2; i++)
            *reinterpret_cast<copy_t *>(&sharedA[(baseIdx / 2 + 128 * i + (baseIdx & 1) * (BLOCK_M + offset)) * sharedLda + stage * sharedASize]) = preA[i];

        *reinterpret_cast<copy_t *>(&sharedB[(baseIdx / 2 + (baseIdx & 1) * (BLOCK_N + offset)) * sharedLdb + stage * sharedBSize]) = preB;

        __syncthreads();

        {
            auto ldA = __cvta_generic_to_shared(&sharedA[((warpId / 2) * 64 + laneId) * sharedLda + stage * sharedASize]);
            auto ldB = __cvta_generic_to_shared(&sharedB[((warpId & 1) * 64 + laneId) * sharedLdb + stage * sharedBSize]);

            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%8];"
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%4 ,%5, %6, %7}, [%9];"
                : "=r"(frag_a[0]), "=r"(frag_a[1]), "=r"(frag_a[2]), "=r"(frag_a[3]), "=r"(frag_b[0]), "=r"(frag_b[1]), "=r"(frag_b[2]), "=r"(frag_b[3])
                : "l"(ldA), "l"(ldB));

            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%8];"
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%4 ,%5, %6, %7}, [%9];"
                : "=r"(frag_a[4]), "=r"(frag_a[5]), "=r"(frag_a[6]), "=r"(frag_a[7]), "=r"(frag_b[4]), "=r"(frag_b[5]), "=r"(frag_b[6]), "=r"(frag_b[7])
                : "l"(ldA + 32 * sharedLda * sizeof(int8_t)), "l"(ldB + 32 * sharedLdb * sizeof(int8_t)));
        }
    }

#pragma unroll
    for (int im = 0; im < 8; im++)
    {
#pragma unroll
        for (int in = 0; in < 8; in++)
        {
            auto idx = im * 8 + in;
            // int32_t frag_d[2];
            //*reinterpret_cast<int64_t *>(frag_d) = *reinterpret_cast<int64_t *>(&baseC[(im * 8 + laneId / 4) * ldc + in * 8 + (laneId & 3) * 2]); // I'm the reinterpret_cast master!
            // frag_d[0] = frag_c[idx][0] * alpha + frag_d[0] * beta;
            // frag_d[1] = frag_c[idx][1] * alpha + frag_d[1] * beta;
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