# CUDA GEMM 注意事项

1. 使用 tiling，将global mem中的东西移动到shared mem中，以减少global mem的访问压力。
2. thread排布依然非常重要，这关系到一个warp的计算访存比，以及一个wave的l2 cache命中率。（一个wave是指当前一个gpu上能够执行多少个thread，这些thread需要从global mem存东西到shared mem，所以影响的是l2访存。）
3. 尝试使用float4 load。因为gpu可以将4次float访问合并为一次访问，这样做不仅可以节省访存次数，同时减少的访存指令还能缓解GPU 访存单元的预测执行压力。
4. float4 load 相比于传统的load会有broadcast limitation（不知道这个limitation在哪）。
5. 使用double buffer来掩盖内存读取的延迟，但是我实际做出来感觉差不多，看到另一个开源实现感觉区别也不是很大。
6. 高occupancy不一定代表高性能，但低occupancy往往意味着低性能。这是一个相对的概念，主要是因为有可能高occupancy但是每一个thread的运行效率不高从而导致高occupancy的性能不一定好。

# Notes

1. 第一版初始的i8 gemm不知道为何速度还比不上sgemm，理论上操作数一致，计算更快能带来更快的计算效率。

# Problems

1. If I use $K_{tile} = 64$, then the l2 cache hit rate will be quite low (at about ~81% compared to ~91% when using $K_{ile}=32$). This may be caused by the HW l2 data exchange strategy. Could try [3.2.3. Device Memory L2 Access Management](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#L2_access_intro) or ptx ld instruction with cache hint in the future.