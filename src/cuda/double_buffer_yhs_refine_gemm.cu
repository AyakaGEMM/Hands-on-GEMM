#include <cstdlib>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
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

// #define subA reinterpret_cast<float *>(addrA)
// #define subB reinterpret_cast<float *>(addrB)

__global__ __launch_bounds__(256, 2) void matrixMul(const float *A, const float *B, float *C,
                                                    int M, int N, int K, float alpha, float beta)
{
    const size_t baseX = blockIdx.x * blockDim.x * BLOCK_M_COMPUTE;
    const size_t baseY = blockIdx.y * blockDim.y * BLOCK_N_COMPUTE;

    const int moveNum = shared_memory_element / (BLOCK_SIZE * BLOCK_SIZE) / 2;
    const size_t baseIdx = threadIdx.y * blockDim.y + threadIdx.x;

    constexpr size_t threadsNum = BLOCK_SIZE * BLOCK_SIZE;

    float c[BLOCK_M_COMPUTE * BLOCK_N_COMPUTE] = {};
    constexpr size_t subAlda = BLOCK_M + 4; // plus 4 here to avoid bank conflict and maintain float4 read

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

    // #pragma unroll
    //     for (int i = 0; i < 4; i++)
    //     {
    //         sts32(preB.x, addrB + ((baseIdx / 32) * BLOCK_N + (baseIdx & 31) * sizeof(float)));
    //     }

    subB[(baseIdx / 32) * BLOCK_N + (baseIdx & 31)] = preB.x;
    subB[(baseIdx / 32) * BLOCK_N + (baseIdx & 31) + 32] = preB.y;
    subB[(baseIdx / 32) * BLOCK_N + (baseIdx & 31) + 32 * 2] = preB.z;
    subB[(baseIdx / 32) * BLOCK_N + (baseIdx & 31) + 32 * 3] = preB.w;

    subA[rowA + colA * subAlda] = preA.x;
    subA[rowA + (colA + 1) * subAlda] = preA.y;
    subA[rowA + (colA + 2) * subAlda] = preA.z;
    subA[rowA + (colA + 3) * subAlda] = preA.w;

    __syncthreads();

    regB[0][0] = *reinterpret_cast<float4 *>(&subB[colC]);
    regB[0][1] = *reinterpret_cast<float4 *>(&subB[colC + 32]);

    regA[0][0] = *reinterpret_cast<float4 *>(&subA[rowC]);
    regA[0][1] = *reinterpret_cast<float4 *>(&subA[rowC + 16]);

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
                regB[(ii + 1) % 2][0] = *reinterpret_cast<float4 *>(&subB[colC + BLOCK_N * (ii + 1)]);
                regB[(ii + 1) % 2][1] = *reinterpret_cast<float4 *>(&subB[colC + 32 + BLOCK_N * (ii + 1)]);

                regA[(ii + 1) % 2][0] = *reinterpret_cast<float4 *>(&subA[rowC + (ii + 1) * subAlda]);
                regA[(ii + 1) % 2][1] = *reinterpret_cast<float4 *>(&subA[(rowC + 16) + (ii + 1) * subAlda]);
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

        if ((i / BLOCK_K) & 1)
        {
            subA += 2 * 1024;
            subB += 1024;
        }
        else
        {
            subA -= 2 * 1024;
            subB -= 1024;
        }

        subB[(baseIdx / 32) * BLOCK_N + (baseIdx & 31)] = preB.x;
        subB[(baseIdx / 32) * BLOCK_N + (baseIdx & 31) + 32] = preB.y;
        subB[(baseIdx / 32) * BLOCK_N + (baseIdx & 31) + 32 * 2] = preB.z;
        subB[(baseIdx / 32) * BLOCK_N + (baseIdx & 31) + 32 * 3] = preB.w;

        subA[rowA + colA * subAlda] = preA.x;
        subA[rowA + (colA + 1) * subAlda] = preA.y;
        subA[rowA + (colA + 2) * subAlda] = preA.z;
        subA[rowA + (colA + 3) * subAlda] = preA.w;

        __syncthreads();

        regB[0][0] = *reinterpret_cast<float4 *>(&subB[colC]);
        regB[0][1] = *reinterpret_cast<float4 *>(&subB[colC + 32]);

        regA[0][0] = *reinterpret_cast<float4 *>(&subA[rowC]);
        regA[0][1] = *reinterpret_cast<float4 *>(&subA[rowC + 16]);
    }

#pragma unroll
    for (int ii = 0; ii < BLOCK_K; ii++)
    {
        if (ii != 7)
        {
            regB[(ii + 1) % 2][0] = *reinterpret_cast<float4 *>(&subB[colC + BLOCK_N * (ii + 1)]);
            regB[(ii + 1) % 2][1] = *reinterpret_cast<float4 *>(&subB[colC + 32 + BLOCK_N * (ii + 1)]);

            regA[(ii + 1) % 2][0] = *reinterpret_cast<float4 *>(&subA[rowC + (ii + 1) * subAlda]);
            regA[(ii + 1) % 2][1] = *reinterpret_cast<float4 *>(&subA[(rowC + 16) + (ii + 1) * subAlda]);
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
