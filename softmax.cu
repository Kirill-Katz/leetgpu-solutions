#include <cuda_runtime.h>
#include <cfloat>

__global__ void softmax_kernel(const float* input, float* output, int N, float* block_maxes) {
    __device__ static float max_val = -FLT_MAX;
    extern __shared__ float shared[];
    __shared__ float local_max;

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;

    if (tid == 0) {
        local_max = -FLT_MAX;
    }
    __syncthreads();

    if (idx < N) {
        shared[tid] = input[idx];
    } else shared[tid] = -FLT_MAX;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = max(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_maxes[blockIdx.x] = shared[0];
    }



}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    float* d_block_maxes;
    cudaMalloc(&d_block_maxes, blocksPerGrid * sizeof(float));

    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, d_block_maxes);
    cudaDeviceSynchronize();
}

