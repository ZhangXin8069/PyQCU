/*
// 编译
nvcc gpu_info.cpp
// 执行
./a.ou
// 输出 GPU 信息如下
GPU Name = GeForce GTX 1080 Ti
Compute Capability = 6.1
GPU SMs = 28
GPU SM clock rate = 1.683 GHz
GPU Mem clock rate = 5.505 GHz
*/
// gpu_info.cpp 源代码如下
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#define CHECK_CUDA(x, str) \
  if ((x) != cudaSuccess)  \
  {                        \
    fprintf(stderr, str);  \
    exit(EXIT_FAILURE);    \
  }
int main(void)
{
  int gpu_index = 0;
  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties(&prop, gpu_index), "cudaGetDeviceProperties error");
  printf("GPU Name = %s\n", prop.name);
  printf("Compute Capability = %d.%d\n", prop.major, prop.minor);        // 获得 SM 版本
  printf("GPU SMs = %d\n", prop.multiProcessorCount);                    // 获得 SM 数目
  printf("GPU SM clock rate = %.3f GHz\n", prop.clockRate / 1e6);        // prop.clockRate 单位为 kHz，除以 1e6 之后单位为 GHz
  printf("GPU Mem clock rate = %.3f GHz\n", prop.memoryClockRate / 1e6); // 同上
  if ((prop.major == 8) && (prop.minor == 0)) // SM 8.0，即 A100
  {
    // 根据公式计算峰值吞吐，其中 64、32、256、256 是从表中查到
    printf("-----------CUDA Core Performance------------\n");
    printf("FP32 Peak Performance = %.3f GFLOPS.\n", prop.multiProcessorCount * (prop.clockRate / 1e6) * 64 * 2);
    printf("FP64 Peak Performance = %.3f GFLOPS.\n", prop.multiProcessorCount * (prop.clockRate / 1e6) * 32 * 2);
    printf("FP16 Peak Performance = %.3f GFLOPS.\n", prop.multiProcessorCount * (prop.clockRate / 1e6) * 256 * 2);
    printf("BF16 Peak Performance = %.3f GFLOPS.\n", prop.multiProcessorCount * (prop.clockRate / 1e6) * 128 * 2);
    printf("INT8 Peak Performance = %.3f GOPS.\n", prop.multiProcessorCount * (prop.clockRate / 1e6) * 256 * 2);
    printf("-----------Tensor Core Dense Performance------------\n");
    printf("TF32 Peak Performance = %.3f GFLOPS.\n", prop.multiProcessorCount * (prop.clockRate / 1e6) * 512 * 2);
    printf("FP64 Peak Performance = %.3f GFLOPS.\n", prop.multiProcessorCount * (prop.clockRate / 1e6) * 64 * 2);
    printf("FP16 Peak Performance = %.3f GFLOPS.\n", prop.multiProcessorCount * (prop.clockRate / 1e6) * 1024 * 2);
    printf("BF16 Peak Performance = %.3f GFLOPS.\n", prop.multiProcessorCount * (prop.clockRate / 1e6) * 1024 * 2);
    printf("INT8 Peak Performance = %.3f GOPS.\n", prop.multiProcessorCount * (prop.clockRate / 1e6) * 2048 * 2);
    printf("INT4 Peak Performance = %.3f GOPS.\n", prop.multiProcessorCount * (prop.clockRate / 1e6) * 4096 * 2);
    printf("INT1 Peak Performance = %.3f GOPS.\n", prop.multiProcessorCount * (prop.clockRate / 1e6) * 16384 * 2);
    printf("-----------Tensor Core Sparse Performance------------\n");
    printf("TF32 Peak Performance = %.3f GFLOPS.\n", prop.multiProcessorCount * (prop.clockRate / 1e6) * 1024 * 2);
    printf("FP16 Peak Performance = %.3f GFLOPS.\n", prop.multiProcessorCount * (prop.clockRate / 1e6) * 2048 * 2);
    printf("BF16 Peak Performance = %.3f GFLOPS.\n", prop.multiProcessorCount * (prop.clockRate / 1e6) * 2048 * 2);
    printf("INT8 Peak Performance = %.3f GOPS.\n", prop.multiProcessorCount * (prop.clockRate / 1e6) * 4096 * 2);
    printf("INT4 Peak Performance = %.3f GOPS.\n", prop.multiProcessorCount * (prop.clockRate / 1e6) * 8192 * 2);
  }
  return 0;
}
