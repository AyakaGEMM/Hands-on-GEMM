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

__device__ __forceinline__ void stg32(const float &reg, void *ptr, bool guard)
{
    asm volatile(
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
        " @p st.global.f32 [%0], %1;}\n"
        :
        : "l"(ptr), "f"(reg), "r"((int)guard));
}

__device__ __forceinline__ void lds128(float &reg0, float &reg1,
                                       float &reg2, float &reg3,
                                       const uint32_t &addr)
{
    asm volatile(
        "ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];\n"
        : "=f"(reg0), "=f"(reg1), "=f"(reg2), "=f"(reg3)
        : "r"(addr));
}

__device__ __forceinline__ void sts32(const float &reg, const uint32_t &addr)
{
    asm volatile(
        "st.shared.f32 [%0], %1;\n"
        :
        : "r"(addr), "f"(reg));
}

__device__ __forceinline__ void sts128(const float &reg0, const float &reg1,
                                       const float &reg2, const float &reg3,
                                       const uint32_t &addr)
{
    asm volatile(
        "st.shared.v4.f32 [%0], {%1, %2, %3, %4};\n"
        :
        : "r"(addr), "f"(reg0), "f"(reg1), "f"(reg2), "f"(reg3));
}

__device__ __forceinline__ uint32_t smem_u32addr(const void *smem_ptr)
{
    uint32_t addr;
    asm("{.reg .u64 u64addr;\n"
        " cvta.to.shared.u64 u64addr, %1;\n"
        " cvt.u32.u64 %0, u64addr;}\n"
        : "=r"(addr)
        : "l"(smem_ptr));

    return addr;
}

//#define subA reinterpret_cast<float *>(addrA)
//#define subB reinterpret_cast<float *>(addrB)

__global__ void matrixMul(const float *A, const float *B, float *C,
                          int M, int N, int K, float alpha, float beta)
{
    const size_t baseX = blockIdx.x * blockDim.x * BLOCK_M_COMPUTE;
    const size_t baseY = blockIdx.y * blockDim.y * BLOCK_N_COMPUTE;

    const int moveNum = shared_memory_element / (BLOCK_SIZE * BLOCK_SIZE) / 2;
    const size_t baseIdx = threadIdx.y * blockDim.y + threadIdx.x;

    constexpr size_t threadsNum = BLOCK_SIZE * BLOCK_SIZE;

    float c[BLOCK_M_COMPUTE * BLOCK_N_COMPUTE] = {};
    constexpr size_t subAlda = BLOCK_M + 4; // plus 4 here to avoid bank conflict and maintain float4 read

    //__shared__ float subA[2][subAlda * BLOCK_K];
    //__shared__ float subB[2][BLOCK_N * BLOCK_K];

    __shared__ __align__(16 * 1024) char smem[6 * 4 * 1024];
    auto subA = reinterpret_cast<float *>(smem);
    auto subB = reinterpret_cast<float *>(smem + 4 * 4 * 1024);

    float4 regB[2][BLOCK_M_COMPUTE / 4]; // hopefully, these should reside in register.
    float4 regA[2][BLOCK_M_COMPUTE / 4];

    const float *baseA = A + baseY * K;
    const float *baseB = B + baseX;

    auto compute_stage_idx = 0;

    int rowA = baseIdx >> 1, rowB = baseIdx >> 5, colA = (baseIdx & 1) << 2, colB = baseIdx & 31;
    int warpId = baseIdx >> 5, warpBaseId = baseIdx & 31;
    int rowC = ((warpId >> 1 << 3) + ((warpBaseId >> 4) << 1) + (warpBaseId & 1)) << 2, colC = (((warpId & 1) << 4) + ((warpBaseId & 15) >> 1)) << 2;
    float *baseC = C + (baseY + rowC) * N + baseX + colC;

    float4 preA, preB;

    preB.x = baseB[rowB * N + colB];
    preB.y = baseB[rowB * N + colB + 32];
    preB.z = baseB[rowB * N + colB + 32 * 2];
    preB.w = baseB[rowB * N + colB + 32 * 3];
    preA = *reinterpret_cast<const float4 *>(baseA + rowA * K + colA);

    //#pragma unroll
    //    for (int i = 0; i < 4; i++)
    //    {
    //        sts32(preB.x, addrB + ((baseIdx / 32) * BLOCK_N + (baseIdx & 31) * sizeof(float)));
    //    }

    uint32_t stsAddrA = smem_u32addr(subA + rowA + colA * subAlda), stsAddrB = smem_u32addr(subB + (baseIdx / 32) * BLOCK_N + (baseIdx & 31));
    uint32_t ldsAddrA = smem_u32addr(subA + rowC), ldsAddrB = smem_u32addr(subB + colC);

    sts32(preB.x, stsAddrB);
    sts32(preB.y, stsAddrB + 32 * sizeof(float));
    sts32(preB.z, stsAddrB + 32 * 2 * sizeof(float));
    sts32(preB.w, stsAddrB + 32 * 3 * sizeof(float));

    sts32(preA.x, stsAddrA);
    sts32(preA.y, stsAddrA + subAlda * sizeof(float));
    sts32(preA.z, stsAddrA + subAlda * 2 * sizeof(float));
    sts32(preA.w, stsAddrA + subAlda * 3 * sizeof(float));

    __syncthreads();

    lds128(regB[0][0].x, regB[0][0].y, regB[0][0].z, regB[0][0].w, ldsAddrB);
    lds128(regB[0][1].x, regB[0][1].y, regB[0][1].z, regB[0][1].w, ldsAddrB + 32 * sizeof(float));

    lds128(regA[0][0].x, regA[0][0].y, regA[0][0].z, regA[0][0].w, ldsAddrA);
    lds128(regA[0][1].x, regA[0][1].y, regA[0][1].z, regA[0][1].w, ldsAddrA + 16 * sizeof(float));

    for (int i = BLOCK_K; i < K; i += BLOCK_K)
    {
        preB.x = baseB[rowB * N + i * N + colB];
        preB.y = baseB[rowB * N + i * N + colB + 32];
        preB.z = baseB[rowB * N + i * N + colB + 32 * 2];
        preB.w = baseB[rowB * N + i * N + colB + 32 * 3];
        preA = *reinterpret_cast<const float4 *>(baseA + i + rowA * K + colA);

#pragma unroll
        for (int ii = 0; ii < BLOCK_K; ii++)
        {
            if (ii != 7)
            {
                lds128(regB[(ii + 1) % 2][0].x, regB[(ii + 1) % 2][0].y, regB[(ii + 1) % 2][0].z, regB[(ii + 1) % 2][0].w, ldsAddrB + BLOCK_N * (ii + 1) * sizeof(float));
                lds128(regB[(ii + 1) % 2][1].x, regB[(ii + 1) % 2][1].y, regB[(ii + 1) % 2][1].z, regB[(ii + 1) % 2][1].w, ldsAddrB + 32 * sizeof(float) + BLOCK_N * (ii + 1) * sizeof(float));

                lds128(regA[(ii + 1) % 2][0].x, regA[(ii + 1) % 2][0].y, regA[(ii + 1) % 2][0].z, regA[(ii + 1) % 2][0].w, ldsAddrA + (ii + 1) * subAlda * sizeof(float));
                lds128(regA[(ii + 1) % 2][1].x, regA[(ii + 1) % 2][1].y, regA[(ii + 1) % 2][1].z, regA[(ii + 1) % 2][1].w, ldsAddrA + 16 * sizeof(float) + (ii + 1) * subAlda * sizeof(float));
            }

#pragma unroll
            for (int cpi = 0; cpi < BLOCK_M_COMPUTE / 4; cpi++)
            {
#pragma unroll
                for (int cpj = 0; cpj < BLOCK_N_COMPUTE / 4; cpj++)
                {
                    c[cpi * 4 * BLOCK_M_COMPUTE + cpj * 4] += regA[ii & 1][cpi].x * regB[ii & 1][cpj].x;
                    c[cpi * 4 * BLOCK_M_COMPUTE + cpj * 4 + 1] += regA[ii & 1][cpi].x * regB[ii & 1][cpj].y;
                    c[cpi * 4 * BLOCK_M_COMPUTE + cpj * 4 + 2] += regA[ii & 1][cpi].x * regB[ii & 1][cpj].z;
                    c[cpi * 4 * BLOCK_M_COMPUTE + cpj * 4 + 3] += regA[ii & 1][cpi].x * regB[ii & 1][cpj].w;

                    c[(cpi * 4 + 1) * BLOCK_M_COMPUTE + cpj * 4] += regA[ii & 1][cpi].y * regB[ii & 1][cpj].x;
                    c[(cpi * 4 + 1) * BLOCK_M_COMPUTE + cpj * 4 + 1] += regA[ii & 1][cpi].y * regB[ii & 1][cpj].y;
                    c[(cpi * 4 + 1) * BLOCK_M_COMPUTE + cpj * 4 + 2] += regA[ii & 1][cpi].y * regB[ii & 1][cpj].z;
                    c[(cpi * 4 + 1) * BLOCK_M_COMPUTE + cpj * 4 + 3] += regA[ii & 1][cpi].y * regB[ii & 1][cpj].w;

                    c[(cpi * 4 + 2) * BLOCK_M_COMPUTE + cpj * 4] += regA[ii & 1][cpi].z * regB[ii & 1][cpj].x;
                    c[(cpi * 4 + 2) * BLOCK_M_COMPUTE + cpj * 4 + 1] += regA[ii & 1][cpi].z * regB[ii & 1][cpj].y;
                    c[(cpi * 4 + 2) * BLOCK_M_COMPUTE + cpj * 4 + 2] += regA[ii & 1][cpi].z * regB[ii & 1][cpj].z;
                    c[(cpi * 4 + 2) * BLOCK_M_COMPUTE + cpj * 4 + 3] += regA[ii & 1][cpi].z * regB[ii & 1][cpj].w;

                    c[(cpi * 4 + 3) * BLOCK_M_COMPUTE + cpj * 4] += regA[ii & 1][cpi].w * regB[ii & 1][cpj].x;
                    c[(cpi * 4 + 3) * BLOCK_M_COMPUTE + cpj * 4 + 1] += regA[ii & 1][cpi].w * regB[ii & 1][cpj].y;
                    c[(cpi * 4 + 3) * BLOCK_M_COMPUTE + cpj * 4 + 2] += regA[ii & 1][cpi].w * regB[ii & 1][cpj].z;
                    c[(cpi * 4 + 3) * BLOCK_M_COMPUTE + cpj * 4 + 3] += regA[ii & 1][cpi].w * regB[ii & 1][cpj].w;
                }
            }
        }

        stsAddrA ^= 0x2000;
        ldsAddrA ^= 0x2000;

        stsAddrB ^= 0x1000;
        ldsAddrB ^= 0x1000;

        sts32(preB.x, stsAddrB);
        sts32(preB.y, stsAddrB + 32 * sizeof(float));
        sts32(preB.z, stsAddrB + 32 * 2 * sizeof(float));
        sts32(preB.w, stsAddrB + 32 * 3 * sizeof(float));

        sts32(preA.x, stsAddrA);
        sts32(preA.y, stsAddrA + subAlda * sizeof(float));
        sts32(preA.z, stsAddrA + subAlda * 2 * sizeof(float));
        sts32(preA.w, stsAddrA + subAlda * 3 * sizeof(float));

        __syncthreads();

        lds128(regB[0][0].x, regB[0][0].y, regB[0][0].z, regB[0][0].w, ldsAddrB);
        lds128(regB[0][1].x, regB[0][1].y, regB[0][1].z, regB[0][1].w, ldsAddrB + 32 * sizeof(float));

        lds128(regA[0][0].x, regA[0][0].y, regA[0][0].z, regA[0][0].w, ldsAddrA);
        lds128(regA[0][1].x, regA[0][1].y, regA[0][1].z, regA[0][1].w, ldsAddrA + 16 * sizeof(float));
    }

#pragma unroll
    for (int ii = 0; ii < BLOCK_K; ii++)
    {
        if (ii != 7)
        {
            lds128(regB[(ii + 1) % 2][0].x, regB[(ii + 1) % 2][0].y, regB[(ii + 1) % 2][0].z, regB[(ii + 1) % 2][0].w, ldsAddrB + BLOCK_N * (ii + 1) * sizeof(float));
            lds128(regB[(ii + 1) % 2][1].x, regB[(ii + 1) % 2][1].y, regB[(ii + 1) % 2][1].z, regB[(ii + 1) % 2][1].w, ldsAddrB + 32 * sizeof(float) + BLOCK_N * (ii + 1) * sizeof(float));

            lds128(regA[(ii + 1) % 2][0].x, regA[(ii + 1) % 2][0].y, regA[(ii + 1) % 2][0].z, regA[(ii + 1) % 2][0].w, ldsAddrA + (ii + 1) * subAlda * sizeof(float));
            lds128(regA[(ii + 1) % 2][1].x, regA[(ii + 1) % 2][1].y, regA[(ii + 1) % 2][1].z, regA[(ii + 1) % 2][1].w, ldsAddrA + 16 * sizeof(float) + (ii + 1) * subAlda * sizeof(float));
        }
#pragma unroll
        for (int cpi = 0; cpi < BLOCK_M_COMPUTE / 4; cpi++)
        {
#pragma unroll
            for (int cpj = 0; cpj < BLOCK_N_COMPUTE / 4; cpj++)
            {
                c[cpi * 4 * BLOCK_M_COMPUTE + cpj * 4] += regA[ii & 1][cpi].x * regB[ii & 1][cpj].x;
                c[cpi * 4 * BLOCK_M_COMPUTE + cpj * 4 + 1] += regA[ii & 1][cpi].x * regB[ii & 1][cpj].y;
                c[cpi * 4 * BLOCK_M_COMPUTE + cpj * 4 + 2] += regA[ii & 1][cpi].x * regB[ii & 1][cpj].z;
                c[cpi * 4 * BLOCK_M_COMPUTE + cpj * 4 + 3] += regA[ii & 1][cpi].x * regB[ii & 1][cpj].w;

                c[(cpi * 4 + 1) * BLOCK_M_COMPUTE + cpj * 4] += regA[ii & 1][cpi].y * regB[ii & 1][cpj].x;
                c[(cpi * 4 + 1) * BLOCK_M_COMPUTE + cpj * 4 + 1] += regA[ii & 1][cpi].y * regB[ii & 1][cpj].y;
                c[(cpi * 4 + 1) * BLOCK_M_COMPUTE + cpj * 4 + 2] += regA[ii & 1][cpi].y * regB[ii & 1][cpj].z;
                c[(cpi * 4 + 1) * BLOCK_M_COMPUTE + cpj * 4 + 3] += regA[ii & 1][cpi].y * regB[ii & 1][cpj].w;

                c[(cpi * 4 + 2) * BLOCK_M_COMPUTE + cpj * 4] += regA[ii & 1][cpi].z * regB[ii & 1][cpj].x;
                c[(cpi * 4 + 2) * BLOCK_M_COMPUTE + cpj * 4 + 1] += regA[ii & 1][cpi].z * regB[ii & 1][cpj].y;
                c[(cpi * 4 + 2) * BLOCK_M_COMPUTE + cpj * 4 + 2] += regA[ii & 1][cpi].z * regB[ii & 1][cpj].z;
                c[(cpi * 4 + 2) * BLOCK_M_COMPUTE + cpj * 4 + 3] += regA[ii & 1][cpi].z * regB[ii & 1][cpj].w;

                c[(cpi * 4 + 3) * BLOCK_M_COMPUTE + cpj * 4] += regA[ii & 1][cpi].w * regB[ii & 1][cpj].x;
                c[(cpi * 4 + 3) * BLOCK_M_COMPUTE + cpj * 4 + 1] += regA[ii & 1][cpi].w * regB[ii & 1][cpj].y;
                c[(cpi * 4 + 3) * BLOCK_M_COMPUTE + cpj * 4 + 2] += regA[ii & 1][cpi].w * regB[ii & 1][cpj].z;
                c[(cpi * 4 + 3) * BLOCK_M_COMPUTE + cpj * 4 + 3] += regA[ii & 1][cpi].w * regB[ii & 1][cpj].w;
            }
        }
    }

#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        *reinterpret_cast<float4 *>(&preA) = *reinterpret_cast<float4 *>(&baseC[i * N]);
        preA.x = preA.x * beta + alpha * c[i * BLOCK_N_COMPUTE];
        preA.y = preA.y * beta + alpha * c[1 + i * BLOCK_N_COMPUTE];
        preA.z = preA.z * beta + alpha * c[2 + i * BLOCK_N_COMPUTE];
        preA.w = preA.w * beta + alpha * c[3 + i * BLOCK_N_COMPUTE];
        *reinterpret_cast<float4 *>(&baseC[i * N]) = *reinterpret_cast<float4 *>(&preA);

        *reinterpret_cast<float4 *>(&preA) = *reinterpret_cast<float4 *>(&baseC[i * N + 32]);
        preA.x = preA.x * beta + alpha * c[4 + i * BLOCK_N_COMPUTE];
        preA.y = preA.y * beta + alpha * c[5 + i * BLOCK_N_COMPUTE];
        preA.z = preA.z * beta + alpha * c[6 + i * BLOCK_N_COMPUTE];
        preA.w = preA.w * beta + alpha * c[7 + i * BLOCK_N_COMPUTE];
        *reinterpret_cast<float4 *>(&baseC[i * N + 32]) = *reinterpret_cast<float4 *>(&preA);

        *reinterpret_cast<float4 *>(&preA) = *reinterpret_cast<float4 *>(&baseC[(i + 16) * N]);
        preA.x = preA.x * beta + alpha * c[32 + i * BLOCK_N_COMPUTE];
        preA.y = preA.y * beta + alpha * c[33 + i * BLOCK_N_COMPUTE];
        preA.z = preA.z * beta + alpha * c[34 + i * BLOCK_N_COMPUTE];
        preA.w = preA.w * beta + alpha * c[35 + i * BLOCK_N_COMPUTE];
        *reinterpret_cast<float4 *>(&baseC[(i + 16) * N]) = *reinterpret_cast<float4 *>(&preA);

        *reinterpret_cast<float4 *>(&preA) = *reinterpret_cast<float4 *>(&baseC[(i + 16) * N + 32]);
        preA.x = preA.x * beta + alpha * c[36 + i * BLOCK_N_COMPUTE];
        preA.y = preA.y * beta + alpha * c[37 + i * BLOCK_N_COMPUTE];
        preA.z = preA.z * beta + alpha * c[38 + i * BLOCK_N_COMPUTE];
        preA.w = preA.w * beta + alpha * c[39 + i * BLOCK_N_COMPUTE];
        *reinterpret_cast<float4 *>(&baseC[(i + 16) * N + 32]) = *reinterpret_cast<float4 *>(&preA);
    }
}

void sgemm(int M, int N, int K, float *a, float *b, float *c, float alpha = 1, float beta = 0)
{
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((M + BLOCK_M - 1) / BLOCK_M, (N + BLOCK_N - 1) / BLOCK_N);
#ifdef __CUDACC__ // workaround for stupid vscode intellisense
    matrixMul<<<numBlocks, threadsPerBlock>>>(a, b, c, M, N, K, alpha, beta);
#endif
}
