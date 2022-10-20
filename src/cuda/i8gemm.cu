#include <cstdlib>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
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
#define colM(a, i, j, lda) a[((j) * (lda)) + (i)]
#define rowM(a, i, j, lda) a[(j) + (i) * (lda)]

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

__global__ void i8gemm8x8x128(const int8_t *A, const int8_t *B, int32_t *C,
                              int M, int N, int K, int32_t alpha, int32_t beta)
{
    const size_t baseX = blockIdx.x * blockDim.x * BLOCK_M_COMPUTE;
    const size_t baseY = blockIdx.y * blockDim.y * BLOCK_N_COMPUTE;

    const int moveNum = shared_memory_element / (BLOCK_SIZE * BLOCK_SIZE) / 2;
    const size_t baseIdx = threadIdx.y * blockDim.y + threadIdx.x;

    constexpr size_t threadsNum = BLOCK_SIZE * BLOCK_SIZE;

    int32_t c[BLOCK_M_COMPUTE * BLOCK_N_COMPUTE] = {};
    constexpr size_t subAlda = BLOCK_M + 4; // plus 4 here to avoid bank conflict and maintain float4 read

    __shared__ int8_t subA[2][subAlda * BLOCK_K];
    __shared__ int8_t subB[2][BLOCK_N * BLOCK_K];

    int8_t regB[BLOCK_M_COMPUTE / 4][4]; // hopefully, these should reside in register.
    int8_t regA[BLOCK_M_COMPUTE / 4][4];

    const auto *baseA = A + baseY * K;
    const auto *baseB = B + baseX;

    auto compute_stage_idx = 0;

    int rowA = baseIdx >> 1, rowB = baseIdx >> 5, colA = (baseIdx & 1) << 2, colB = baseIdx & 31;
    int warpId = baseIdx >> 5, warpBaseId = baseIdx & 31;
    int rowC = ((warpId >> 1 << 3) + ((warpBaseId >> 4) << 1) + (warpBaseId & 1)) << 2, colC = (((warpId & 1) << 4) + ((warpBaseId & 15) >> 1)) << 2;
    auto *baseC = C + (baseY + rowC) * N + baseX + colC;

    int8_t preA[4], preB[4];

    preB[0] = baseB[rowB * N + colB];
    preB[1] = baseB[rowB * N + colB + 32];
    preB[2] = baseB[rowB * N + colB + 32 * 2];
    preB[3] = baseB[rowB * N + colB + 32 * 3];
    preA[0] = baseA[rowA * K + colA];
    preA[1] = baseA[rowA * K + colA + 1];
    preA[2] = baseA[rowA * K + colA + 2];
    preA[3] = baseA[rowA * K + colA + 3];

    subB[0][(baseIdx / 32) * BLOCK_N + (baseIdx & 31)] = preB[0];
    subB[0][(baseIdx / 32) * BLOCK_N + (baseIdx & 31) + 32] = preB[1];
    subB[0][(baseIdx / 32) * BLOCK_N + (baseIdx & 31) + 32 * 2] = preB[2];
    subB[0][(baseIdx / 32) * BLOCK_N + (baseIdx & 31) + 32 * 3] = preB[3];

    subA[0][rowA + colA * subAlda] = preA[0];
    subA[0][rowA + (colA + 1) * subAlda] = preA[1];
    subA[0][rowA + (colA + 2) * subAlda] = preA[2];
    subA[0][rowA + (colA + 3) * subAlda] = preA[3];

    __syncthreads();

    for (int i = BLOCK_K; i < K; i += BLOCK_K)
    {
        preB[0] = baseB[rowB * N + i * N + colB];
        preB[1] = baseB[rowB * N + i * N + colB + 32];
        preB[2] = baseB[rowB * N + i * N + colB + 32 * 2];
        preB[3] = baseB[rowB * N + i * N + colB + 32 * 3];

        preA[0] = baseA[i + rowA * K + colA];
        preA[1] = baseA[i + rowA * K + colA + 1];
        preA[2] = baseA[i + rowA * K + colA + 2];
        preA[3] = baseA[i + rowA * K + colA + 3];

#pragma unroll
        for (int ii = 0; ii < BLOCK_K; ii++)
        {
            regB[0][0] = subB[compute_stage_idx][colC + BLOCK_N * ii];
            regB[0][1] = subB[compute_stage_idx][colC + BLOCK_N * ii + 1];
            regB[0][2] = subB[compute_stage_idx][colC + BLOCK_N * ii + 2];
            regB[0][3] = subB[compute_stage_idx][colC + BLOCK_N * ii + 3];
            regB[1][0] = subB[compute_stage_idx][colC + 32 + BLOCK_N * ii];
            regB[1][1] = subB[compute_stage_idx][colC + 32 + BLOCK_N * ii + 1];
            regB[1][2] = subB[compute_stage_idx][colC + 32 + BLOCK_N * ii + 2];
            regB[1][3] = subB[compute_stage_idx][colC + 32 + BLOCK_N * ii + 3];

            regA[0][0] = subA[compute_stage_idx][rowC + ii * subAlda];
            regA[0][1] = subA[compute_stage_idx][rowC + ii * subAlda + 1];
            regA[0][2] = subA[compute_stage_idx][rowC + ii * subAlda + 2];
            regA[0][3] = subA[compute_stage_idx][rowC + ii * subAlda + 3];
            regA[1][0] = subA[compute_stage_idx][rowC + 16 + ii * subAlda];
            regA[1][1] = subA[compute_stage_idx][rowC + 16 + ii * subAlda + 1];
            regA[1][2] = subA[compute_stage_idx][rowC + 16 + ii * subAlda + 2];
            regA[1][3] = subA[compute_stage_idx][rowC + 16 + ii * subAlda + 3];

#pragma unroll
            for (int cpi = 0; cpi < BLOCK_M_COMPUTE / 4; cpi++)
            {
#pragma unroll
                for (int cpj = 0; cpj < BLOCK_N_COMPUTE / 4; cpj++)
                {
                    c[cpi * 4 * BLOCK_M_COMPUTE + cpj * 4] += regA[cpi][0] * regB[cpj][0];
                    c[cpi * 4 * BLOCK_M_COMPUTE + cpj * 4 + 1] += regA[cpi][0] * regB[cpj][1];
                    c[cpi * 4 * BLOCK_M_COMPUTE + cpj * 4 + 2] += regA[cpi][0] * regB[cpj][2];
                    c[cpi * 4 * BLOCK_M_COMPUTE + cpj * 4 + 3] += regA[cpi][0] * regB[cpj][3];

                    c[(cpi * 4 + 1) * BLOCK_M_COMPUTE + cpj * 4] += regA[cpi][1] * regB[cpj][0];
                    c[(cpi * 4 + 1) * BLOCK_M_COMPUTE + cpj * 4 + 1] += regA[cpi][1] * regB[cpj][1];
                    c[(cpi * 4 + 1) * BLOCK_M_COMPUTE + cpj * 4 + 2] += regA[cpi][1] * regB[cpj][2];
                    c[(cpi * 4 + 1) * BLOCK_M_COMPUTE + cpj * 4 + 3] += regA[cpi][1] * regB[cpj][3];

                    c[(cpi * 4 + 2) * BLOCK_M_COMPUTE + cpj * 4] += regA[cpi][2] * regB[cpj][0];
                    c[(cpi * 4 + 2) * BLOCK_M_COMPUTE + cpj * 4 + 1] += regA[cpi][2] * regB[cpj][1];
                    c[(cpi * 4 + 2) * BLOCK_M_COMPUTE + cpj * 4 + 2] += regA[cpi][2] * regB[cpj][2];
                    c[(cpi * 4 + 2) * BLOCK_M_COMPUTE + cpj * 4 + 3] += regA[cpi][2] * regB[cpj][3];

                    c[(cpi * 4 + 3) * BLOCK_M_COMPUTE + cpj * 4] += regA[cpi][3] * regB[cpj][0];
                    c[(cpi * 4 + 3) * BLOCK_M_COMPUTE + cpj * 4 + 1] += regA[cpi][3] * regB[cpj][1];
                    c[(cpi * 4 + 3) * BLOCK_M_COMPUTE + cpj * 4 + 2] += regA[cpi][3] * regB[cpj][2];
                    c[(cpi * 4 + 3) * BLOCK_M_COMPUTE + cpj * 4 + 3] += regA[cpi][3] * regB[cpj][3];
                }
            }
        }

        compute_stage_idx ^= 1;

        subB[compute_stage_idx][(baseIdx / 32) * BLOCK_N + (baseIdx & 31)] = preB[0];
        subB[compute_stage_idx][(baseIdx / 32) * BLOCK_N + (baseIdx & 31) + 32] = preB[1];
        subB[compute_stage_idx][(baseIdx / 32) * BLOCK_N + (baseIdx & 31) + 32 * 2] = preB[2];
        subB[compute_stage_idx][(baseIdx / 32) * BLOCK_N + (baseIdx & 31) + 32 * 3] = preB[3];

        subA[compute_stage_idx][rowA + colA * subAlda] = preA[0];
        subA[compute_stage_idx][rowA + (colA + 1) * subAlda] = preA[1];
        subA[compute_stage_idx][rowA + (colA + 2) * subAlda] = preA[2];
        subA[compute_stage_idx][rowA + (colA + 3) * subAlda] = preA[3];
        __syncthreads();
    }

#pragma unroll
    for (int ii = 0; ii < BLOCK_K; ii++)
    {
        regB[0][0] = subB[compute_stage_idx][colC + BLOCK_N * ii];
        regB[0][1] = subB[compute_stage_idx][colC + BLOCK_N * ii + 1];
        regB[0][2] = subB[compute_stage_idx][colC + BLOCK_N * ii + 2];
        regB[0][3] = subB[compute_stage_idx][colC + BLOCK_N * ii + 3];
        regB[1][0] = subB[compute_stage_idx][colC + 32 + BLOCK_N * ii];
        regB[1][1] = subB[compute_stage_idx][colC + 32 + BLOCK_N * ii + 1];
        regB[1][2] = subB[compute_stage_idx][colC + 32 + BLOCK_N * ii + 2];
        regB[1][3] = subB[compute_stage_idx][colC + 32 + BLOCK_N * ii + 3];

        regA[0][0] = subA[compute_stage_idx][rowC + ii * subAlda];
        regA[0][1] = subA[compute_stage_idx][rowC + ii * subAlda + 1];
        regA[0][2] = subA[compute_stage_idx][rowC + ii * subAlda + 2];
        regA[0][3] = subA[compute_stage_idx][rowC + ii * subAlda + 3];
        regA[1][0] = subA[compute_stage_idx][rowC + 16 + ii * subAlda];
        regA[1][1] = subA[compute_stage_idx][rowC + 16 + ii * subAlda + 1];
        regA[1][2] = subA[compute_stage_idx][rowC + 16 + ii * subAlda + 2];
        regA[1][3] = subA[compute_stage_idx][rowC + 16 + ii * subAlda + 3];

#pragma unroll
        for (int cpi = 0; cpi < BLOCK_M_COMPUTE / 4; cpi++)
        {
#pragma unroll
            for (int cpj = 0; cpj < BLOCK_N_COMPUTE / 4; cpj++)
            {
                c[cpi * 4 * BLOCK_M_COMPUTE + cpj * 4] += regA[cpi][0] * regB[cpj][0];
                c[cpi * 4 * BLOCK_M_COMPUTE + cpj * 4 + 1] += regA[cpi][0] * regB[cpj][1];
                c[cpi * 4 * BLOCK_M_COMPUTE + cpj * 4 + 2] += regA[cpi][0] * regB[cpj][2];
                c[cpi * 4 * BLOCK_M_COMPUTE + cpj * 4 + 3] += regA[cpi][0] * regB[cpj][3];

                c[(cpi * 4 + 1) * BLOCK_M_COMPUTE + cpj * 4] += regA[cpi][1] * regB[cpj][0];
                c[(cpi * 4 + 1) * BLOCK_M_COMPUTE + cpj * 4 + 1] += regA[cpi][1] * regB[cpj][1];
                c[(cpi * 4 + 1) * BLOCK_M_COMPUTE + cpj * 4 + 2] += regA[cpi][1] * regB[cpj][2];
                c[(cpi * 4 + 1) * BLOCK_M_COMPUTE + cpj * 4 + 3] += regA[cpi][1] * regB[cpj][3];

                c[(cpi * 4 + 2) * BLOCK_M_COMPUTE + cpj * 4] += regA[cpi][2] * regB[cpj][0];
                c[(cpi * 4 + 2) * BLOCK_M_COMPUTE + cpj * 4 + 1] += regA[cpi][2] * regB[cpj][1];
                c[(cpi * 4 + 2) * BLOCK_M_COMPUTE + cpj * 4 + 2] += regA[cpi][2] * regB[cpj][2];
                c[(cpi * 4 + 2) * BLOCK_M_COMPUTE + cpj * 4 + 3] += regA[cpi][2] * regB[cpj][3];

                c[(cpi * 4 + 3) * BLOCK_M_COMPUTE + cpj * 4] += regA[cpi][3] * regB[cpj][0];
                c[(cpi * 4 + 3) * BLOCK_M_COMPUTE + cpj * 4 + 1] += regA[cpi][3] * regB[cpj][1];
                c[(cpi * 4 + 3) * BLOCK_M_COMPUTE + cpj * 4 + 2] += regA[cpi][3] * regB[cpj][2];
                c[(cpi * 4 + 3) * BLOCK_M_COMPUTE + cpj * 4 + 3] += regA[cpi][3] * regB[cpj][3];
            }
        }
    }

    {
        int32_t preA[4];
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            preA[0] = baseC[i * N];
            preA[1] = baseC[i * N + 1];
            preA[2] = baseC[i * N + 2];
            preA[3] = baseC[i * N + 3];
            preA[0] = preA[0] * beta + alpha * c[i * BLOCK_N_COMPUTE];
            preA[1] = preA[1] * beta + alpha * c[1 + i * BLOCK_N_COMPUTE];
            preA[2] = preA[2] * beta + alpha * c[2 + i * BLOCK_N_COMPUTE];
            preA[3] = preA[3] * beta + alpha * c[3 + i * BLOCK_N_COMPUTE];
            baseC[i * N] = preA[0];
            baseC[i * N + 1] = preA[1];
            baseC[i * N + 2] = preA[2];
            baseC[i * N + 3] = preA[3];

            preA[0] = baseC[i * N + 32];
            preA[1] = baseC[i * N + 32 + 1];
            preA[2] = baseC[i * N + 32 + 2];
            preA[3] = baseC[i * N + 32 + 3];
            preA[0] = preA[0] * beta + alpha * c[4 + i * BLOCK_N_COMPUTE];
            preA[1] = preA[1] * beta + alpha * c[5 + i * BLOCK_N_COMPUTE];
            preA[2] = preA[2] * beta + alpha * c[6 + i * BLOCK_N_COMPUTE];
            preA[3] = preA[3] * beta + alpha * c[7 + i * BLOCK_N_COMPUTE];
            baseC[i * N + 32] = preA[0];
            baseC[i * N + 32 + 1] = preA[1];
            baseC[i * N + 32 + 2] = preA[2];
            baseC[i * N + 32 + 3] = preA[3];

            preA[0] = baseC[(i + 16) * N];
            preA[1] = baseC[(i + 16) * N + 1];
            preA[2] = baseC[(i + 16) * N + 2];
            preA[3] = baseC[(i + 16) * N + 3];
            preA[0] = preA[0] * beta + alpha * c[32 + i * BLOCK_N_COMPUTE];
            preA[1] = preA[1] * beta + alpha * c[33 + i * BLOCK_N_COMPUTE];
            preA[2] = preA[2] * beta + alpha * c[34 + i * BLOCK_N_COMPUTE];
            preA[3] = preA[3] * beta + alpha * c[35 + i * BLOCK_N_COMPUTE];
            baseC[(i + 16) * N] = preA[0];
            baseC[(i + 16) * N + 1] = preA[1];
            baseC[(i + 16) * N + 2] = preA[2];
            baseC[(i + 16) * N + 3] = preA[3];

            preA[0] = baseC[(i + 16) * N + 32];
            preA[1] = baseC[(i + 16) * N + 32 + 1];
            preA[2] = baseC[(i + 16) * N + 32 + 2];
            preA[3] = baseC[(i + 16) * N + 32 + 3];
            preA[0] = preA[0] * beta + alpha * c[36 + i * BLOCK_N_COMPUTE];
            preA[1] = preA[1] * beta + alpha * c[37 + i * BLOCK_N_COMPUTE];
            preA[2] = preA[2] * beta + alpha * c[38 + i * BLOCK_N_COMPUTE];
            preA[3] = preA[3] * beta + alpha * c[39 + i * BLOCK_N_COMPUTE];
            baseC[(i + 16) * N + 32] = preA[0];
            baseC[(i + 16) * N + 32 + 1] = preA[1];
            baseC[(i + 16) * N + 32 + 2] = preA[2];
            baseC[(i + 16) * N + 32 + 3] = preA[3];
        }
    }
}

void i8gemm(int M, int N, int K, int8_t *a, int8_t *b, int32_t *c, int32_t alpha, int32_t beta)
{
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((M + BLOCK_M - 1) / BLOCK_M, (N + BLOCK_N - 1) / BLOCK_N);
#ifdef __CUDACC__ // workaround for stupid vscode intellisense
    i8gemm8x8x128<<<numBlocks, threadsPerBlock>>>(a, b, c, M, N, K, alpha, beta);
#endif
}