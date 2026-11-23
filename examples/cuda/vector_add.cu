#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << "\n"; \
        return 1; \
    } \
} while(0)

__global__ void vector_add(
    const float* a,
    const float* b,
    float* c,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int N = 1 << 20;
    const size_t bytes = N * sizeof(float);

    std::vector<float> h_a(N, 2.0f);
    std::vector<float> h_b(N, 3.0f);
    std::vector<float> h_c(N);

    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;

    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    bool valid = true;

    for (int i = 0; i < N; ++i) {
        if (std::abs(h_c[i] - 5.0f) > 1e-5f) {
            valid = false;
            break;
        }
    }

    std::cout << "GPU: ";
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << prop.name << "\n";

    std::cout << "Elements: " << N << "\n";
    std::cout << "Expected result: 5.0\n";
    std::cout << "Validation: " << (valid ? "PASSED" : "FAILED") << "\n";

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return valid ? 0 : 1;
}
