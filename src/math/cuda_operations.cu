
#include "cppml/math/cuda_operations.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace cppml::math::cuda {
namespace {

constexpr unsigned int THREADS_PER_BLOCK = 256;

// ============================================================
// CUDA Error Handling
// ============================================================

void check_cuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(error)
        );
    }
}

// ============================================================
// GPU Memory Management
// ============================================================

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t bytes) {
        check_cuda(cudaMalloc(&pointer_, bytes), "cudaMalloc");
    }

    ~DeviceBuffer() {
        if (pointer_ != nullptr) {
            cudaFree(pointer_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    float* get() noexcept {
        return pointer_;
    }

private:
    float* pointer_ = nullptr;
};

// ============================================================
// CUDA Grid Calculation
// ============================================================

unsigned int calculate_blocks(std::size_t count) {
    const std::size_t blocks =
        count / THREADS_PER_BLOCK +
        (count % THREADS_PER_BLOCK != 0);

    if (blocks > std::numeric_limits<unsigned int>::max()) {
        throw std::length_error("CUDA grid dimension is too large");
    }

    return static_cast<unsigned int>(blocks);
}

// ============================================================
// CUDA Kernels
// ============================================================

// X(rows x cols) * w(cols) -> result(rows)

__global__ void matvec_kernel(
    const float* X,
    const float* w,
    float* result,
    std::size_t rows,
    std::size_t cols
) {
    const std::size_t row =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (row >= rows) {
        return;
    }

    float accumulator = 0.0f;

    for (std::size_t col = 0; col < cols; ++col) {
        accumulator += X[row * cols + col] * w[col];
    }

    result[row] = accumulator;
}

// X^T(cols x rows) * r(rows) -> result(cols)

__global__ void transpose_matvec_kernel(
    const float* X,
    const float* r,
    float* result,
    std::size_t rows,
    std::size_t cols
) {
    const std::size_t col =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (col >= cols) {
        return;
    }

    float accumulator = 0.0f;

    for (std::size_t row = 0; row < rows; ++row) {
        accumulator += X[row * cols + col] * r[row];
    }

    result[col] = accumulator;
}

// a(n) - b(n) -> result(n)

__global__ void subtract_kernel(
    const float* a,
    const float* b,
    float* result,
    std::size_t size
) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index < size) {
        result[index] = a[index] - b[index];
    }
}

// a(n) + scalar -> result(n)

__global__ void add_scalar_kernel(
    const float* input,
    float* result,
    float value,
    std::size_t size
) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index < size) {
        result[index] = input[index] + value;
    }
}

// ============================================================
// Dot Product Reduction Kernels
// ============================================================

__global__ void dot_partial_kernel(
    const float* a,
    const float* b,
    float* partial_sums,
    std::size_t size
) {
    __shared__ float shared_sums[THREADS_PER_BLOCK];

    const unsigned int tid = threadIdx.x;

    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + tid;

    float value = 0.0f;

    if (index < size) {
        value = a[index] * b[index];
    }

    shared_sums[tid] = value;

    __syncthreads();

    for (unsigned int stride = blockDim.x / 2;
         stride > 0;
         stride /= 2) {

        if (tid < stride) {
            shared_sums[tid] += shared_sums[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = shared_sums[0];
    }
}

__global__ void dot_final_kernel(
    const float* partial_sums,
    float* result,
    std::size_t partial_count
) {
    __shared__ float shared_sums[THREADS_PER_BLOCK];

    const unsigned int tid = threadIdx.x;

    float value = 0.0f;

    for (std::size_t i = tid;
         i < partial_count;
         i += blockDim.x) {

        value += partial_sums[i];
    }

    shared_sums[tid] = value;

    __syncthreads();

    for (unsigned int stride = blockDim.x / 2;
         stride > 0;
         stride /= 2) {

        if (tid < stride) {
            shared_sums[tid] += shared_sums[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0) {
        result[0] = shared_sums[0];
    }
}

// ============================================================
// Sum Reduction Kernels
// ============================================================

__global__ void sum_partial_kernel(
    const float* input,
    float* partial_sums,
    std::size_t size
) {
    __shared__ float shared_sums[THREADS_PER_BLOCK];

    const unsigned int tid = threadIdx.x;

    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + tid;

    float value = 0.0f;

    if (index < size) {
        value = input[index];
    }

    shared_sums[tid] = value;

    __syncthreads();

    for (unsigned int stride = blockDim.x / 2;
         stride > 0;
         stride /= 2) {

        if (tid < stride) {
            shared_sums[tid] += shared_sums[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = shared_sums[0];
    }
}

__global__ void sum_final_kernel(
    const float* partial_sums,
    float* result,
    std::size_t partial_count
) {
    __shared__ float shared_sums[THREADS_PER_BLOCK];

    const unsigned int tid = threadIdx.x;

    float value = 0.0f;

    for (std::size_t i = tid;
         i < partial_count;
         i += blockDim.x) {

        value += partial_sums[i];
    }

    shared_sums[tid] = value;

    __syncthreads();

    for (unsigned int stride = blockDim.x / 2;
         stride > 0;
         stride /= 2) {

        if (tid < stride) {
            shared_sums[tid] += shared_sums[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0) {
        result[0] = shared_sums[0];
    }
}

} // namespace

// ============================================================
// Matrix-Vector Multiplication
// ============================================================

Vector matvec(const Matrix& X, const Vector& w) {
    if (X.cols() != w.size()) {
        throw std::invalid_argument(
            "cuda::matvec: matrix columns must equal vector size"
        );
    }

    const std::size_t rows = X.rows();
    const std::size_t cols = X.cols();

    if (rows == 0) {
        return Vector{};
    }

    if (cols == 0) {
        return Vector(rows, 0.0f);
    }

    const std::size_t matrix_bytes = X.size() * sizeof(float);
    const std::size_t vector_bytes = w.size() * sizeof(float);
    const std::size_t result_bytes = rows * sizeof(float);

    Vector result(rows);

    DeviceBuffer device_X(matrix_bytes);
    DeviceBuffer device_w(vector_bytes);
    DeviceBuffer device_result(result_bytes);

    check_cuda(
        cudaMemcpy(device_X.get(), X.data(), matrix_bytes,
                   cudaMemcpyHostToDevice),
        "Copy matrix to GPU"
    );

    check_cuda(
        cudaMemcpy(device_w.get(), w.data(), vector_bytes,
                   cudaMemcpyHostToDevice),
        "Copy vector to GPU"
    );

    const unsigned int blocks = calculate_blocks(rows);

    matvec_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        device_X.get(), device_w.get(), device_result.get(), rows, cols
    );

    check_cuda(cudaGetLastError(), "Launch matvec kernel");
    check_cuda(cudaDeviceSynchronize(), "Synchronize matvec kernel");

    check_cuda(
        cudaMemcpy(result.data(), device_result.get(), result_bytes,
                   cudaMemcpyDeviceToHost),
        "Copy matvec result from GPU"
    );

    return result;
}

// ============================================================
// Transpose Matrix-Vector Multiplication
// ============================================================

Vector transpose_matvec(const Matrix& X, const Vector& r) {
    if (X.rows() != r.size()) {
        throw std::invalid_argument(
            "cuda::transpose_matvec: matrix rows must equal vector size"
        );
    }

    const std::size_t rows = X.rows();
    const std::size_t cols = X.cols();

    if (cols == 0) {
        return Vector{};
    }

    if (rows == 0) {
        return Vector(cols, 0.0f);
    }

    const std::size_t matrix_bytes = X.size() * sizeof(float);
    const std::size_t vector_bytes = r.size() * sizeof(float);
    const std::size_t result_bytes = cols * sizeof(float);

    Vector result(cols);

    DeviceBuffer device_X(matrix_bytes);
    DeviceBuffer device_r(vector_bytes);
    DeviceBuffer device_result(result_bytes);

    check_cuda(
        cudaMemcpy(device_X.get(), X.data(), matrix_bytes,
                   cudaMemcpyHostToDevice),
        "Copy matrix to GPU"
    );

    check_cuda(
        cudaMemcpy(device_r.get(), r.data(), vector_bytes,
                   cudaMemcpyHostToDevice),
        "Copy vector to GPU"
    );

    const unsigned int blocks = calculate_blocks(cols);

    transpose_matvec_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        device_X.get(), device_r.get(), device_result.get(), rows, cols
    );

    check_cuda(cudaGetLastError(), "Launch transpose_matvec kernel");
    check_cuda(cudaDeviceSynchronize(), "Synchronize transpose_matvec kernel");

    check_cuda(
        cudaMemcpy(result.data(), device_result.get(), result_bytes,
                   cudaMemcpyDeviceToHost),
        "Copy transpose_matvec result from GPU"
    );

    return result;
}

// ============================================================
// Element-wise Vector Subtraction
// ============================================================

Vector subtract(const Vector& a, const Vector& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument(
            "cuda::subtract: vectors must have the same size"
        );
    }

    const std::size_t size = a.size();

    if (size == 0) {
        return Vector{};
    }

    const std::size_t vector_bytes = size * sizeof(float);

    Vector result(size);

    DeviceBuffer device_a(vector_bytes);
    DeviceBuffer device_b(vector_bytes);
    DeviceBuffer device_result(vector_bytes);

    check_cuda(
        cudaMemcpy(device_a.get(), a.data(), vector_bytes,
                   cudaMemcpyHostToDevice),
        "Copy first vector to GPU"
    );

    check_cuda(
        cudaMemcpy(device_b.get(), b.data(), vector_bytes,
                   cudaMemcpyHostToDevice),
        "Copy second vector to GPU"
    );

    const unsigned int blocks = calculate_blocks(size);

    subtract_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        device_a.get(), device_b.get(), device_result.get(), size
    );

    check_cuda(cudaGetLastError(), "Launch subtract kernel");
    check_cuda(cudaDeviceSynchronize(), "Synchronize subtract kernel");

    check_cuda(
        cudaMemcpy(result.data(), device_result.get(), vector_bytes,
                   cudaMemcpyDeviceToHost),
        "Copy subtract result from GPU"
    );

    return result;
}

// ============================================================
// Vector Dot Product
// ============================================================

float dot(const Vector& a, const Vector& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument(
            "cuda::dot: vectors must have the same size"
        );
    }

    const std::size_t size = a.size();

    if (size == 0) {
        return 0.0f;
    }

    const unsigned int blocks = calculate_blocks(size);

    const std::size_t vector_bytes = size * sizeof(float);
    const std::size_t partial_bytes =
        static_cast<std::size_t>(blocks) * sizeof(float);

    Vector result(1);

    DeviceBuffer device_a(vector_bytes);
    DeviceBuffer device_b(vector_bytes);
    DeviceBuffer device_partial(partial_bytes);
    DeviceBuffer device_result(sizeof(float));

    check_cuda(
        cudaMemcpy(device_a.get(), a.data(), vector_bytes,
                   cudaMemcpyHostToDevice),
        "Copy first dot vector to GPU"
    );

    check_cuda(
        cudaMemcpy(device_b.get(), b.data(), vector_bytes,
                   cudaMemcpyHostToDevice),
        "Copy second dot vector to GPU"
    );

    dot_partial_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        device_a.get(), device_b.get(), device_partial.get(), size
    );

    check_cuda(cudaGetLastError(), "Launch dot partial reduction kernel");

    dot_final_kernel<<<1, THREADS_PER_BLOCK>>>(
        device_partial.get(), device_result.get(), blocks
    );

    check_cuda(cudaGetLastError(), "Launch dot final reduction kernel");
    check_cuda(cudaDeviceSynchronize(), "Synchronize dot reduction kernels");

    check_cuda(
        cudaMemcpy(result.data(), device_result.get(), sizeof(float),
                   cudaMemcpyDeviceToHost),
        "Copy dot result from GPU"
    );

    return result[0];
}

// ============================================================
// Vector Sum
// ============================================================

float sum(const Vector& v) {
    const std::size_t size = v.size();

    if (size == 0) {
        return 0.0f;
    }

    const unsigned int blocks = calculate_blocks(size);

    const std::size_t vector_bytes = size * sizeof(float);
    const std::size_t partial_bytes =
        static_cast<std::size_t>(blocks) * sizeof(float);

    Vector result(1);

    DeviceBuffer device_input(vector_bytes);
    DeviceBuffer device_partial(partial_bytes);
    DeviceBuffer device_result(sizeof(float));

    check_cuda(
        cudaMemcpy(device_input.get(), v.data(), vector_bytes,
                   cudaMemcpyHostToDevice),
        "Copy input vector to GPU"
    );

    sum_partial_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        device_input.get(), device_partial.get(), size
    );

    check_cuda(cudaGetLastError(), "Launch sum partial reduction kernel");

    sum_final_kernel<<<1, THREADS_PER_BLOCK>>>(
        device_partial.get(), device_result.get(), blocks
    );

    check_cuda(cudaGetLastError(), "Launch sum final reduction kernel");
    check_cuda(cudaDeviceSynchronize(), "Synchronize sum reduction kernels");

    check_cuda(
        cudaMemcpy(result.data(), device_result.get(), sizeof(float),
                   cudaMemcpyDeviceToHost),
        "Copy sum result from GPU"
    );

    return result[0];
}

// ============================================================
// Vector Mean
// ============================================================

float mean(const Vector& v) {
    if (v.empty()) {
        throw std::invalid_argument(
            "cuda::mean: cannot calculate mean of an empty vector"
        );
    }

    return sum(v) / static_cast<float>(v.size());
}

// ============================================================
// Add Scalar to Vector
// ============================================================

Vector add_scalar(const Vector& v, float value) {
    const std::size_t size = v.size();

    if (size == 0) {
        return Vector{};
    }

    const std::size_t vector_bytes = size * sizeof(float);

    Vector result(size);

    DeviceBuffer device_input(vector_bytes);
    DeviceBuffer device_result(vector_bytes);

    check_cuda(
        cudaMemcpy(device_input.get(), v.data(), vector_bytes,
                   cudaMemcpyHostToDevice),
        "Copy input vector to GPU"
    );

    const unsigned int blocks = calculate_blocks(size);

    add_scalar_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        device_input.get(),
        device_result.get(),
        value,
        size
    );

    check_cuda(cudaGetLastError(), "Launch add_scalar kernel");
    check_cuda(cudaDeviceSynchronize(), "Synchronize add_scalar kernel");

    check_cuda(
        cudaMemcpy(result.data(), device_result.get(), vector_bytes,
                   cudaMemcpyDeviceToHost),
        "Copy add_scalar result from GPU"
    );

    return result;
}

} // namespace cppml::math::cuda
