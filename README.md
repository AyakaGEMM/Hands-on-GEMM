# NLP-Inference-MM

尝试实现一个适用于 NLP 序列生成任务的 GEMM。

# To-Do

+ 更好的 benchmark。
    
    当前 benchmark借鉴于 https://github.com/Cjkkkk/CUDA_gemm 。不足之处在于其无法一次性的测量多个形状的矩阵乘速度，无外部导出机制。

+ cuda gemm的高效实现。
    
    尝试使用 cuda simd 以及 tensor core。
    
+ 更多的 To-Do