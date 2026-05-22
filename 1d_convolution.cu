#include <cuda_runtime.h>

__global__ void convolution_1d_kernel(
    const float* input,
    const float* kernel,
    float* output,
    int input_size, int kernel_size
) {
    extern __shared__ float tile[];

    int local = threadIdx.x;
    int block_start = blockIdx.x * blockDim.x;
    int idx = block_start + local;
    int output_size = input_size - kernel_size + 1;
    int tile_size = blockDim.x + kernel_size - 1;

    for (int t = local; t < tile_size; t += blockDim.x) {
        int input_idx = block_start + t;

        if (input_idx < input_size) {
            tile[t] = input[input_idx];
        } else {
            tile[t] = 0.0f;
        }
    }

    __syncthreads();

    if (idx < output_size) {
        float val = 0.0f;

        for (int i = 0; i < kernel_size; ++i) {
            val += tile[local + i] * kernel[i];
        }

        output[idx] = val;
    }
}

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, const float* kernel, float* output, int input_size,
                      int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    int sharedMemBytes = (threadsPerBlock + kernel_size - 1) * sizeof(float);
    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemBytes>>>(input, kernel, output, input_size,
                                                              kernel_size);
    cudaDeviceSynchronize();
}
