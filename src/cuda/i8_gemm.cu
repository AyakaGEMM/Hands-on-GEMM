#include <cstdlib>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <cub/cub.cuh>
#include <cuda_help_func.hpp>

#ifndef __CUDACC__
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
void __syncthreads(); // workaround __syncthreads warning
void __syncwarp();
#endif

#include <iostream>

enum quantType
{
    MIN_MAX,
    PER_COL,
    PER_ROW
};

class quantMatrixHolder
{
    int8_t *_data;
    size_t _M, _N, _size;
    float *_scales;
    int32_t *_zeroPoints;
    quantType _qt;
    int32_t *_sums;

public:
    quantMatrixHolder() : _data(nullptr), _scales(nullptr), _zeroPoints(nullptr), _sums(nullptr), _M(0), _N(0), _size(0), _qt(MIN_MAX) {}
    ~quantMatrixHolder()
    {
        cudaFree(_data);
        cudaFree(_scales);
        cudaFree(_zeroPoints);
        cudaFree(_sums);
    }

    quantMatrixHolder(size_t M, size_t N, quantType qt) : _qt(qt)
    {
        cudaMalloc(&_data, M * N);
        if (_qt == MIN_MAX)
        {
            cudaMalloc(&_scales, sizeof(float));
            cudaMalloc(&_zeroPoints, sizeof(int32_t));
            cudaMalloc(&_sums, sizeof(int32_t) * N);
        }
        else if (_qt == PER_COL)
        {
            cudaMalloc(&_scales, sizeof(float) * N);
            cudaMalloc(&_zeroPoints, sizeof(int32_t) * N);
            cudaMalloc(&_sums, sizeof(int32_t) * N);
        }
        else if (_qt == PER_ROW)
        {
            cudaMalloc(&_scales, sizeof(float) * M);
            cudaMalloc(&_zeroPoints, sizeof(int32_t) * M);
            cudaMalloc(&_sums, sizeof(int32_t) * M);
        }
        _M = M;
        _N = N;
        _size = M * N;
    }

    void resize(size_t M, size_t N)
    {
        if (M * N >= _size)
        {
            _size = M * N * 2;
            cudaFree(_data);
            cudaMalloc(&_data, _size * sizeof(int8_t));
        }

        if (_qt == PER_COL && N >= _N)
        {
            cudaFree(_scales);
            cudaFree(_zeroPoints);
            cudaFree(_sums);
            cudaMalloc(&_scales, sizeof(float) * N);
            cudaMalloc(&_zeroPoints, sizeof(int32_t) * N);
            cudaMalloc(&_sums, sizeof(int32_t) * N);
        }
        else if (_qt == PER_ROW && M >= _M)
        {
            cudaFree(_scales);
            cudaFree(_zeroPoints);
            cudaFree(_sums);
            cudaMalloc(&_scales, sizeof(float) * M);
            cudaMalloc(&_zeroPoints, sizeof(int32_t) * M);
            cudaMalloc(&_sums, sizeof(int32_t) * M);
        }
        else if (_qt == MIN_MAX)
        {
            if (_scales == nullptr)
            {
                cudaMalloc(&_scales, sizeof(float));
                cudaMalloc(&_zeroPoints, sizeof(int32_t));
            }
            if (N >= _N)
            {
                cudaFree(_sums);
                cudaMalloc(&_sums, sizeof(int32_t) * N);
            }
        }
        _M = M;
        _N = N;
    }

    void showStatus()
    {
        void *hostMem = malloc(_size * sizeof(float));

        checkCudaErrors(cudaMemcpy(hostMem, _data, sizeof(int8_t) * _size, cudaMemcpyDeviceToHost));
        std::cout << "Quant Result: " << std::endl;
        for (int i = 0; i < _M; i++)
        {
            for (int j = 0; j < _N; j++)
            {
                std::cout << int16_t(reinterpret_cast<int8_t *>(hostMem)[i * _N + j]) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
        if (_qt == MIN_MAX)
        {
            checkCudaErrors(cudaMemcpy(hostMem, _zeroPoints, sizeof(int32_t), cudaMemcpyDeviceToHost));
            std::cout << "Zero Point: " << reinterpret_cast<int32_t *>(hostMem)[0] << std::endl;
            checkCudaErrors(cudaMemcpy(hostMem, _scales, sizeof(float), cudaMemcpyDeviceToHost));
            std::cout << "Scale: " << reinterpret_cast<float *>(hostMem)[0] << std::endl;
            checkCudaErrors(cudaMemcpy(hostMem, _sums, sizeof(int32_t) * _M, cudaMemcpyDeviceToHost));
            for (int i = 0; i < _M; i++)
            {
                std::cout << reinterpret_cast<int32_t *>(hostMem)[i] << " ";
            }
            std::cout << std::endl;
        }
        else if (_qt == PER_COL)
        {
            checkCudaErrors(cudaMemcpy(hostMem, _zeroPoints, sizeof(int32_t) * _N, cudaMemcpyDeviceToHost));
            std::cout << "Zero Point: " << std::endl;
            for (int i = 0; i < _N; i++)
            {
                std::cout << reinterpret_cast<int32_t *>(hostMem)[i] << " ";
            }
            std::cout << std::endl;
            checkCudaErrors(cudaMemcpy(hostMem, _scales, sizeof(float) * _N, cudaMemcpyDeviceToHost));
            std::cout << "Scale: " << std::endl;
            for (int i = 0; i < _N; i++)
            {
                std::cout << reinterpret_cast<float *>(hostMem)[i] << " ";
            }
            std::cout << std::endl;
            checkCudaErrors(cudaMemcpy(hostMem, _sums, sizeof(int32_t) * _N, cudaMemcpyDeviceToHost));
            for (int i = 0; i < _N; i++)
            {
                std::cout << reinterpret_cast<int32_t *>(hostMem)[i] << " ";
            }
            std::cout << std::endl;
        }
        else if (_qt == PER_ROW)
        {
            checkCudaErrors(cudaMemcpy(hostMem, _zeroPoints, sizeof(int32_t) * _M, cudaMemcpyDeviceToHost));
            std::cout << "Zero Point: " << std::endl;
            for (int i = 0; i < _M; i++)
            {
                std::cout << reinterpret_cast<int32_t *>(hostMem)[i] << " ";
            }
            std::cout << std::endl;
            checkCudaErrors(cudaMemcpy(hostMem, _scales, sizeof(float) * _M, cudaMemcpyDeviceToHost));
            std::cout << "Scale: " << std::endl;
            for (int i = 0; i < _M; i++)
            {
                std::cout << reinterpret_cast<float *>(hostMem)[i] << " ";
            }
            std::cout << std::endl;
            checkCudaErrors(cudaMemcpy(hostMem, _sums, sizeof(int32_t) * _M, cudaMemcpyDeviceToHost));
            for (int i = 0; i < _M; i++)
            {
                std::cout << reinterpret_cast<int32_t *>(hostMem)[i] << " ";
            }
            std::cout << std::endl;
        }
    }

    size_t size()
    {
        return _size;
    }

    int8_t *dataPtr()
    {
        return _data;
    }

    auto sumsPtr()
    {
        return _sums;
    }

    auto scalesPtr()
    {
        return _scales;
    }

    auto zeroPointsPtr()
    {
        return _zeroPoints;
    }

    void setQuantType(quantType qt)
    {
        cudaFree(_scales);
        cudaFree(_zeroPoints);
        cudaFree(_sums);
        _qt = qt;
        if (_qt == MIN_MAX)
        {
            cudaMalloc(&_scales, sizeof(float));
            cudaMalloc(&_zeroPoints, sizeof(int32_t));
            cudaMalloc(&_sums, sizeof(int32_t) * _M);
        }
        else if (_qt == PER_COL)
        {
            cudaMalloc(&_scales, sizeof(float) * _N);
            cudaMalloc(&_zeroPoints, sizeof(int32_t) * _N);
            cudaMalloc(&_sums, sizeof(int32_t) * _N);
        }
        else if (_qt == PER_ROW)
        {
            cudaMalloc(&_scales, sizeof(float) * _M);
            cudaMalloc(&_zeroPoints, sizeof(int32_t) * _M);
            cudaMalloc(&_sums, sizeof(int32_t) * _M);
        }
    }

    quantMatrixHolder(const quantMatrixHolder &) = delete;
    quantMatrixHolder(const quantMatrixHolder &&) = delete;
    quantMatrixHolder &operator=(const quantMatrixHolder &) = delete;
    quantMatrixHolder &operator=(const quantMatrixHolder &&) = delete;
};

template <typename T, typename S>
__device__ S climp(const T &a)
{
    T min = std::numeric_limits<S>::min(), max = std::numeric_limits<S>::max();
    return std::min(std::max(min, a), max);
}

// develop in progress functions
template <typename T> // naive min max kernel need optimize further
__device__ __forceinline__ void findMinMax(const T *from, T &min, T &max, int32_t N)
{
    const T *to = from + N;
#pragma unroll
    while (from < to)
    {
        T next = *(from++);
        min = (min <= next) ? min : next;
        max = (max <= next) ? next : max;
    }
}

template <typename T> // naive min max kernel need optimize further
__device__ __forceinline__ void getSums(const T *from, T &sum, T &psum, int32_t M, int32_t N)
{
    const T *to = from + M * N;
#pragma unroll
    while (from < to)
    {
        T next = *from;
        from += N;
        sum += next;
        psum += next * next;
    }
}

template <const int BLOCK_THREADS, const int workPerThread>
__global__ void quantInput(
    const float *__restrict__ input,
    const int M,
    const int N,
    const float *max,
    const float *min,
    std::int8_t *__restrict__ output,
    float &scale,
    std::int32_t &zeroPoint,
    std::int32_t *__restrict__ sums)
{
    using namespace cub;

    constexpr float kEpsilon = 1e-8f;
    const int64_t baseIdx = (int)blockIdx.x * blockDim.x + threadIdx.x;

    if (baseIdx * workPerThread < N)
    {
        const float *baseA = input + baseIdx * workPerThread * N;
        auto baseOutput = output + baseIdx * workPerThread * N;

        const float range = *max - *min;

        auto Tscale = range / 255;
        const auto invScale = 255.0f / (range + kEpsilon);
        auto TzeroPoint = int32_t(std::nearbyintf(127 - *max * invScale));

        float sum[workPerThread] = {};

#pragma unroll
        for (std::size_t col = 0; col < workPerThread * N && col + baseIdx * workPerThread * N < M * N; ++col)
        {
            baseOutput[col] = climp<float, int8_t>(std::nearbyintf(baseA[col] * invScale + TzeroPoint));
            sum[col / N] += baseOutput[col];
        }
#pragma unroll
        for (int i = 0; i < workPerThread; i++)
        {
            sums[baseIdx * workPerThread + i] = sum[i];
        }

        if (baseIdx == 0)
        {
            scale = Tscale;
            zeroPoint = TzeroPoint;
        }
    }
}

template <int BLOCK_THREADS, int workPerThread>
__global__ void quantWeight(
    const float *__restrict__ input,
    const int M,
    const int N,
    std::int8_t *__restrict__ output,
    float *__restrict__ scales,
    std::int32_t *__restrict__ zeroPoints,
    std::int32_t *__restrict__ sums)
{
    using namespace cub;

    constexpr float kEpsilon = 1e-8f;

    const int64_t baseIdx = blockIdx.x * blockDim.x + threadIdx.x;
    auto baseOutput = output + baseIdx * workPerThread;
    float sumCols[workPerThread], psumCols[workPerThread];
#pragma unroll
    for (int i = 0; i < workPerThread; i++)
    {
        sumCols[i] = psumCols[i] = 0;
    }

#pragma unroll
    for (int i = 0; i + baseIdx * workPerThread < N && i < workPerThread; i++)
    {
        auto col = i + baseIdx * workPerThread;
        getSums<float>(input + col, sumCols[i], psumCols[i], M, N);
        const float mean = sumCols[i] / M;
        const float stdDevs = sqrtf(psumCols[i] / M - mean * mean);
        // Here 7 is a magic number(a.k.a. hyper-parameter)
        const float min = mean - 7 * stdDevs, max = mean + 7 * stdDevs;
        const float range = max - min;

        scales[col] = range / 255;
        const auto invScale = 255.0f / (range + kEpsilon);
        zeroPoints[col] = int32_t(std::nearbyintf(127 - max * invScale));

        int32_t sum = 0;

#pragma unroll
        for (std::size_t row = 0; row < M; ++row)
        {
            baseOutput[row * N + i] = climp<float, int8_t>(std::nearbyintf(input[col + row * N] * invScale + zeroPoints[col]));
            sum += baseOutput[row * N + i];
        }
        sums[col] = sum;
    }
}

__global__ void dequantFloatMatrix(
    const int32_t *__restrict__ input,
    const int M,
    const int N,
    const int K,
    const int32_t *zeroPointsA, const int32_t *zeroPointsB,
    const float *scalesA, const float *scalesB,
    const int32_t *sumsA, const int32_t *sumsB,
    float *__restrict__ output)
{
    const int64_t baseIdx = (int)blockIdx.x * blockDim.x;
    const auto tid = threadIdx.x;
    const auto tx = tid / 16, ty = tid % 16;
    const size_t baseX = 16 * blockIdx.x, baseY = 16 * blockIdx.y;
    const auto warpId = tid / 32;
    __shared__ int32_t sumA[16], sumB[16], zeroPointB[16];
    __shared__ float scaleB[16];

    if (baseX < M && baseY < N)
    {
        const auto scaleA = *scalesA; // broadcast here
        const auto zeroPointA = *zeroPointsA;
        if (warpId == 0) // warp 0 to copy
        {
            if (tid < 16 && baseX + tid < M)
                sumA[tid] = sumsA[baseX + tid];
            else if (tid >= 16 && baseY + tid - 16 < N)
            {
                zeroPointB[tid - 16] = zeroPointsB[baseY + tid - 16];
                scaleB[tid - 16] = scalesB[baseY + tid - 16];
                sumB[tid - 16] = sumsB[baseY + tid - 16];
            }
        }
        __syncthreads();
        if (baseX + tx < M && baseY + ty < N)
        {
            auto baseA = input[(baseX + tx) * N + baseY + ty];

            output[(baseX + tx) * N + baseY + ty] = scaleA * scaleB[ty] * (baseA - zeroPointA * sumB[ty] - zeroPointB[ty] * sumA[tx] + K * zeroPointA * zeroPointB[ty]);
        }
    }
}

void sgemm(int M, int N, int K, float *a, float *b, float *c, cublasHandle_t handle, float alpha = 1, float beta = 0)
{
    constexpr int workPerThread = 2;
    constexpr int threadsPerBlockSize = 256;
    dim3 threadsPerBlock(threadsPerBlockSize);
    dim3 numInputBlocks((M + threadsPerBlock.x * workPerThread - 1) / threadsPerBlock.x * workPerThread);
    dim3 numWeightBlocks((N + threadsPerBlock.x * workPerThread - 1) / threadsPerBlock.x * workPerThread);
    dim3 numDequantBlocks((M + 15) / 16, (N + 15) / 16);
    static thread_local quantMatrixHolder quantA(M, K, MIN_MAX), quantB(K, N, PER_COL);

    static int32_t *quantC = nullptr;
    static size_t Csize = 0;

    quantA.resize(M, K);
    quantB.resize(K, N);

    if (Csize * 2 < M * N)
    {
        cudaFree(quantC);
        cudaMalloc(&quantC, sizeof(int32_t) * M * N * 2);
        Csize = M * N;
    }

    void *d_temp_storage = nullptr;
    size_t size, prev_size = 0;
    static float *min = nullptr, *max = nullptr;

    if (min == nullptr)
    {
        cudaMalloc(&min, sizeof(float));
        cudaMalloc(&max, sizeof(float));
    }

    cub::DeviceReduce::Max(nullptr, size, a, max, M * K);
    if (size > prev_size * 2)
    {
        cudaFree(d_temp_storage);
        cudaMalloc(&d_temp_storage, size * 2);
        prev_size = size;
    }

    cub::DeviceReduce::Max(d_temp_storage, size, a, max, M * K); // These two can be fused into one.
    cub::DeviceReduce::Min(d_temp_storage, size, a, min, M * K);

#ifdef __CUDACC__ // workaround for stupid vscode intellisense
    quantInput<threadsPerBlockSize, workPerThread><<<numInputBlocks, threadsPerBlock>>>(a, M, K, max, min, quantA.dataPtr(), *quantA.scalesPtr(), *quantA.zeroPointsPtr(), quantA.sumsPtr());
    quantWeight<threadsPerBlockSize, workPerThread><<<numWeightBlocks, threadsPerBlock>>>(b, K, N, quantB.dataPtr(), quantB.scalesPtr(), quantB.zeroPointsPtr(), quantB.sumsPtr());
#endif
    int32_t i32alpha = alpha, i32beta = beta;
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                 N, M, K,
                 &i32alpha,
                 quantB.dataPtr(), CUDA_R_8I, N,
                 quantA.dataPtr(), CUDA_R_8I, K,
                 &i32beta,
                 quantC, CUDA_R_32I, N,
                 CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

#ifdef __CUDACC__ // workaround for stupid vscode intellisense
    dequantFloatMatrix<<<numDequantBlocks, threadsPerBlock>>>(quantC, M, N, K, quantA.zeroPointsPtr(), quantB.zeroPointsPtr(), quantA.scalesPtr(), quantB.scalesPtr(), quantA.sumsPtr(), quantB.sumsPtr(), c);
#endif
}
