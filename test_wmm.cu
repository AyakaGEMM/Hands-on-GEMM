#include <assert.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <iostream>
using namespace std;

__global__ void convert(half *in, half *out)
{
    __shared__ half smem[64];
    auto addr = __cvta_generic_to_shared(smem + (threadIdx.x % 8) * 8);

    if (threadIdx.x == 0)
    {
        for (int i = 0; i < 64; i++)
            smem[i] = in[i];
    }
    __syncthreads();
    int32_t a = 0;
    asm(
        "ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];"
        : "=r"(a)
        : "l"(addr));
    // a = *reinterpret_cast<int32_t *>(smem + threadIdx.x / 4 * 8 + (threadIdx.x % 4) * 2);
    *reinterpret_cast<int32_t *>(&out[threadIdx.x / 4 * 8 + (threadIdx.x % 4) * 2]) = a;
}

int main()
{
    half a[64], b[64];
    half *d_a, *d_b;
    cudaMalloc(&d_a, sizeof(a));
    cudaMalloc(&d_b, sizeof(b));
    memset(a, 0, sizeof(a));
    for (int i = 0; i < 64; i++)
        b[i] = i;
    for (int i = 0; i < 64; i++)
    {
        if (i % 8 == 0)
            cout << endl;
        cout << float(b[i]) << " ";
    }
    cudaMemcpy(d_a, a, sizeof(a), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(b), cudaMemcpyHostToDevice);
    convert<<<1, 32>>>(d_b, d_a);
    cudaMemcpy(a, d_a, sizeof(a), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 64; i++)
    {
        if (i % 8 == 0)
            cout << endl;
        cout << float(a[i]) << " ";
    }
}