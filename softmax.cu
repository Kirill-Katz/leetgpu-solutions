#include <cuda_runtime.h>
#include <cfloat>

__global__ void softmax_kernel(float* input, int N, float sum) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < N) {
        input[idx] = input[idx] / sum;
    }
}

__global__ void get_max(const float* input, int N, float* block_maxes) {
    extern __shared__ float shared[];

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;

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

__global__ void get_sum(const float* input, int N, float* block_sums) {
    extern __shared__ float shared[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if(idx < N){
        shared[tid] = input[idx];
    } else shared[tid] = 0.0f;

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_sums[blockIdx.x] = shared[0];
    }
}

__global__ void normalize_vals(const float* input, int N, float* output, float max_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        output[idx] = __expf(input[idx] - max_val);
    }
}

float get_normalized_sum(float* input, int N) {
    int current = N;
    const float* current_input = input;
    float* prev_input = nullptr;

    while (current > 1) {
        int threadsPerBlock = 256;
        int blocksPerGrid = (current + threadsPerBlock - 1) / threadsPerBlock;

        float* d_block_sums = nullptr;

        cudaMalloc(&d_block_sums, blocksPerGrid * sizeof(float));

        size_t shared_mem = threadsPerBlock * sizeof(float);

        get_sum<<<blocksPerGrid, threadsPerBlock, shared_mem>>>(
            current_input, current, d_block_sums
        );

        cudaDeviceSynchronize();

        if (prev_input != nullptr) {
            cudaFree(prev_input);
        }

        current_input = d_block_sums;
        prev_input = d_block_sums;
        current = blocksPerGrid;
    }

    float result;

    cudaMemcpy(&result, current_input, sizeof(float), cudaMemcpyDeviceToHost);

    if (prev_input != nullptr) {
        cudaFree(prev_input);
    }

    return result;
}


float get_max_val(const float* input, int N) {
    int current = N;
    const float* current_input = input;
    float* prev_input = nullptr;

    while (current > 1) {
        int threadsPerBlock = 256;
        int blocksPerGrid = (current + threadsPerBlock - 1) / threadsPerBlock;

        float* d_block_maxes = nullptr;
        cudaMalloc(&d_block_maxes, blocksPerGrid * sizeof(float));

        size_t shared_mem = threadsPerBlock * sizeof(float);

        get_max<<<blocksPerGrid, threadsPerBlock, shared_mem>>>(
            current_input,
            current,
            d_block_maxes
        );

        cudaDeviceSynchronize();

        if (prev_input != nullptr) {
            cudaFree(prev_input);
        }

        current_input = d_block_maxes;
        prev_input = d_block_maxes;
        current = blocksPerGrid;
    }

    float result;

    cudaMemcpy(&result, current_input, sizeof(float), cudaMemcpyDeviceToHost);

    if (prev_input != nullptr) {
        cudaFree(prev_input);
    }

    return result;
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    float max_val = get_max_val(input, N);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    normalize_vals<<<blocksPerGrid, threadsPerBlock>>>(input, N, output, max_val);

    float sum = get_normalized_sum(output, N);

    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(output, N, sum);
    cudaDeviceSynchronize();
}
