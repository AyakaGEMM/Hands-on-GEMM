#include <cstdlib>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>

#ifndef __CUDACC__
#define __CUDACC__
#define __HAHA__
#endif

#include <cooperative_groups/memcpy_async.h>

#ifdef __HAHA__
#undef __CUDACC__
#endif
#include <cuda/pipeline>

#ifndef __CUDACC__
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
void __syncthreads(); // workaround __syncthreads warning
#endif

using namespace cooperative_groups;

template <typename T>
__global__ void example_kernel(T *global1, T *global2, size_t subset_count)
{
    extern __shared__ T s[];
    constexpr unsigned stages_count = 2;
    auto group = cooperative_groups::this_thread_block();
    T *shared[stages_count] = {s, s + 2 * group.size()};

    // Create a synchronization object (cuda::pipeline)
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope::thread_scope_block, stages_count> shared_state;
    auto pipeline = cuda::make_pipeline(group, &shared_state);

    size_t fetch;
    size_t subset;
    for (subset = fetch = 0; subset < subset_count; ++subset)
    {
        // Fetch ahead up to stages_count subsets
        for (; fetch < subset_count && fetch < (subset + stages_count); ++fetch)
        {
            pipeline.producer_acquire();
            cuda::memcpy_async(group, shared[fetch % 2],
                               &global1[fetch * group.size()], sizeof(T) * group.size(), pipeline);
            cuda::memcpy_async(group, shared[fetch % 2] + group.size(),
                               &global2[fetch * group.size()], sizeof(T) * group.size(), pipeline);
            pipeline.producer_commit(); // Commit the fetch-ahead stage
        }
        pipeline.consumer_wait(); // Wait for ‘subset’ stage to be available

        compute(shared[subset % 2]);

        pipeline.consumer_release();
    }
}

int main()
{
    int n = 1 << 24;
    int blockSize = 256;
    int nBlocks = (n + blockSize - 1) / blockSize;
    int sharedBytes = blockSize * sizeof(int);

    int *sum, *data;
    cudaMallocManaged(&sum, sizeof(int));
    cudaMallocManaged(&data, n * sizeof(int));
    std::fill_n(data, n, 1); // initialize data
    cudaMemset(sum, 0, sizeof(int));

    // sum_kernel_block<<<nBlocks, blockSize, sharedBytes>>>(sum, data, n);
}