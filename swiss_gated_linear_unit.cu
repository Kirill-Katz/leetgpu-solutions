#include <cuda_runtime.h>

__global__ void swiglu_kernel(const float* input, float* output, int halfN) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < halfN) {
        float x1 = input[idx];
        float x2 = input[idx + halfN];
        output[idx] = x1 / (1.0f + __expf(-x1)) * x2;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    swiglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    cudaDeviceSynchronize();
}

