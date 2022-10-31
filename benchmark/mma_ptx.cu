#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

const int WARMUP = 100;
const int ROUND = 50;

// Really naive benchmark of wmm instructions

__global__ __launch_bounds__(32, 1) void wmm_m16n16k16_latency_kernel(const uint32_t *addr, uint32_t *ret, uint32_t *clk)
{
    uint32_t start;
    uint32_t stop;
    uint32_t smem_addr;
    int32_t a[2] = {1, 2}, b[2] = {9, 1}, c[8] = {23, 2, 3, 34, 23, 13, 1, 3}, d[8] = {};

    asm volatile(
        "bar.sync 0;\n"
        "mov.u32 %0, %%clock;\n"
        : "=r"(start)
        :
        : "memory");

#pragma unroll
    for (int i = 0; i < ROUND; ++i)
    {
        /*
         * dependent LDS instructions to make sure that
         * LDS latency can not be hidden by parallel LDS.
         */
        asm volatile(
            "wmma.mma.sync.aligned.m16n16k16.row.row.s32.s8.s8.s32 \
            {%0, %1, %2, %3, %4, %5, %6, %7}, \
            {%8, %9}, {%10, %11}, \
            {%12, %13, %14, %15, %16, %17, %18, %19};\n"
            : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]), "=r"(d[4]), "=r"(d[5]), "=r"(d[6]), "=r"(d[7])
            : "r"(a[0]), "r"(a[1]), "r"(b[0]), "r"(b[1]), "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]), "r"(c[4]), "r"(c[5]), "r"(c[6]), "r"(c[7])
            : "memory");

        c[0] = d[0];
    }

    asm volatile(
        "bar.sync 0;\n"
        "mov.u32 %0, %%clock;\n"
        : "=r"(stop)
        :
        : "memory");

    clk[threadIdx.x] = stop - start;

    ret[0] = d[0];
    ret[1] = d[1];
    ret[2] = d[2];
    ret[3] = d[3];
}

__global__ __launch_bounds__(32, 1) void wmm_m8n32k16_latency_kernel(const uint32_t *addr, uint32_t *ret, uint32_t *clk)
{

    uint32_t start;
    uint32_t stop;
    int32_t a[2] = {1, threadIdx.x}, b[4] = {threadIdx.x, 1}, c[8] = {23, 2, threadIdx.x, 34, 23, 13, 1, 3}, d[8] = {};

    asm volatile(
        "bar.sync 0;\n"
        "mov.u32 %0, %%clock;\n"
        : "=r"(start)
        :
        : "memory");

#pragma unroll
    for (int i = 0; i < ROUND; ++i)
    {
        /*
         * dependent LDS instructions to make sure that
         * LDS latency can not be hidden by parallel LDS.
         */
        asm volatile(
            "wmma.mma.sync.aligned.m8n32k16.row.row.s32.s8.s8.s32 \
            {%0, %1, %2, %3, %4, %5, %6, %7}, \
            {%8}, {%9, %10, %11, %12}, \
            {%13, %14, %15, %16, %17, %18, %19, %20};\n"
            : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]), "=r"(d[4]), "=r"(d[5]), "=r"(d[6]), "=r"(d[7])
            : "r"(a[0]), "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]), "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]), "r"(c[4]), "r"(c[5]), "r"(c[6]), "r"(c[7])
            : "memory");

        c[0] = d[0];
    }

    asm volatile(
        "bar.sync 0;\n"
        "mov.u32 %0, %%clock;\n"
        : "=r"(stop)
        :
        : "memory");

    clk[threadIdx.x] = stop - start;

    ret[0] = d[0];
    ret[1] = d[1];
    ret[2] = d[2];
    ret[3] = d[3];
}

__global__ __launch_bounds__(32, 1) void mma_m16n8k32_latency_kernel(const uint32_t *addr, uint32_t *ret, uint32_t *clk)
{

    uint32_t start;
    uint32_t stop;
    int32_t a[4] = {1, threadIdx.x}, b[8] = {threadIdx.x, 1}, c[8] = {23, 2, threadIdx.x, 34, 23, 13, 1, 3}, d[8] = {};

    asm volatile(
        "bar.sync 0;\n"
        "mov.u32 %0, %%clock;\n"
        : "=r"(start)
        :
        : "memory");

#pragma unroll
    for (int i = 0; i < ROUND; ++i)
    {
        /*
         * dependent LDS instructions to make sure that
         * LDS latency can not be hidden by parallel LDS.
         */
        // asm volatile(
        //     "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 \
        //    {%0, %1, %2, %3}, \
        //    {%4, %5, %6, %7}, {%8, %9}, \
        //    {%10, %11, %12, %13};\n"
        //     : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        //     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3])
        //     : "memory");

        c[0] = d[0];
    }

    asm volatile(
        "bar.sync 0;\n"
        "mov.u32 %0, %%clock;\n"
        : "=r"(stop)
        :
        : "memory");

    clk[threadIdx.x] = stop - start;

    ret[0] = d[0];
    ret[1] = d[1];
    ret[2] = d[2];
    ret[3] = d[3];
}

__global__ __launch_bounds__(32, 1) void mma_m8n8k16_latency_kernel(const uint32_t *addr, uint32_t *ret, uint32_t *clk)
{

    uint32_t start;
    uint32_t stop;
    int32_t a[4] = {1, threadIdx.x}, b[8] = {threadIdx.x, 1}, c[8] = {23, 2, threadIdx.x, 34, 23, 13, 1, 3}, d[8] = {};

    asm volatile(
        "bar.sync 0;\n"
        "mov.u32 %0, %%clock;\n"
        : "=r"(start)
        :
        : "memory");

#pragma unroll
    for (int i = 0; i < ROUND; ++i)
    {
        /*
         * dependent LDS instructions to make sure that
         * LDS latency can not be hidden by parallel LDS.
         */
        asm volatile(
            "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 \
            {%0, %1}, \
            {%2}, {%3}, \
            {%4, %5};\n"
            : "=r"(d[0]), "=r"(d[1])
            : "r"(a[0]), "r"(b[0]), "r"(c[0]), "r"(c[1])
            : "memory");

        c[0] = d[0];
    }

    asm volatile(
        "bar.sync 0;\n"
        "mov.u32 %0, %%clock;\n"
        : "=r"(stop)
        :
        : "memory");

    clk[threadIdx.x] = stop - start;

    ret[0] = d[0];
    ret[1] = d[1];
    ret[2] = d[2];
    ret[3] = d[3];
}

int main()
{
    uint32_t *h_addr;
    cudaMallocHost(&h_addr, 16 * sizeof(uint32_t));

    for (int i = 0; i < 16; ++i)
    {
        h_addr[i] = i * sizeof(uint32_t);
    }

    uint32_t *d_addr, *d_ret;
    cudaMalloc(&d_addr, 32 * sizeof(uint32_t));
    cudaMalloc(&d_ret, sizeof(uint32_t));
    cudaMemcpy(d_addr, h_addr, 32 * sizeof(uint32_t), cudaMemcpyHostToDevice);

    uint32_t *d_clk;
    cudaMalloc(&d_clk, 32 * sizeof(uint32_t));

    // shared memory latency benchmark
    wmm_m16n16k16_latency_kernel<<<1, 32>>>(d_addr, d_ret, d_clk);

    uint32_t h_clk[32];
    cudaMemcpy(h_clk, d_clk, 32 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    printf("mma.m16n16k16.s32.s8.s8.s32 latency %u cycles\n", h_clk[0] / ROUND);

    cudaFree(d_addr);
    cudaFree(d_ret);
    cudaFree(d_clk);
    cudaFreeHost(h_addr);

    return 0;
}
