# Hands-on-GEMM

A GEMM tutorial.

# Performance

![SGEMM 性能对比](https://user-images.githubusercontent.com/31173671/204142853-b1e45cb0-a8b4-42ff-a207-7eee31712305.png)

# Usage

去 `src/cuda` 文件夹下面，找到你想看性能的 gemm，记住那个名字，然后回到主项目文件夹下，首先`mkdir build`，然后输入 `make benchmark_xxx`。

如你想看 `double_buffer_yhs_refine_gemm.cu` 这个矩阵乘的性能，就输入：

```
make benchmark_double_buffer_yhs_refine
```

然后二进制会出现在 `bin` 文件夹下面。

# Tutorial

知乎链接：[这里](https://zhuanlan.zhihu.com/p/584236348)

公众号链接：[这里](https://mp.weixin.qq.com/s/rWWx0Uf4oin0kmtEjVXBqw)
