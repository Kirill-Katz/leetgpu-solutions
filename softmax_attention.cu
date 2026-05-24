#include <__clang_cuda_builtin_vars.h>
#include <cuda_runtime.h>

__global__ void matrix_multiplication_kernel(
    const float* A,
    const float* B,
    float* C, int M, int N, int K
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int r = idx / K;
    int c = idx % K;

    if (r < M && c < K) {
        float sum = 0;

        for (int i = 0; i < N; ++i) {
             sum += A[r * N + i] * B[i * K + c];
        }

        C[r * K + c] = sum;
    }
}

__global__ void divide_by_scalar(float* A, int m, int n, float d) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n * m) {
        A[idx] = A[idx] / d;
    }
}

__global__ void matrix_transpose_kernel(
    const float* input,
    float* output,
    int rows, int cols
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    int r = idx / rows;
    int c = idx % rows;

    if (r < rows && c < cols) {
        output[c * rows + r] = input[r * cols + c];
    }
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
    float* K_T;
    cudaMalloc(&K_T, sizeof(float) * N * d);

    int threadsPerBlock = 256;
    int blocksPerGrid = ((N * d + threadsPerBlock - 1) / threadsPerBlock);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(K, K_T, N, d);

    float* QK_T;
    cudaMalloc(&QK_T, sizeof(float) * M * N);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(Q, K_T, QK_T, M, d, N);

    divide_by_scalar<<<blocksPerGrid, threadsPerBlock>>>(QK_T, M, N, sqrt(d));




}

