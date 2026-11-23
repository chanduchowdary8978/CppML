#include "cppml/models/cuda_linear_regression.hpp"

#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace cppml::models {
namespace {

constexpr unsigned int THREADS_PER_BLOCK = 256;
constexpr unsigned int FEATURE_TILE = 32;
constexpr unsigned int ROW_TILE = 8;

void check_cuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(error)
        );
    }
}

float checked_float(double value, const char* operation) {
    if (!std::isfinite(value) ||
        std::abs(value) >
            static_cast<double>(std::numeric_limits<float>::max())) {
        throw std::overflow_error(
            std::string(operation) + ": result exceeds float range"
        );
    }

    return static_cast<float>(value);
}

std::size_t checked_product(
    std::size_t a,
    std::size_t b,
    const char* operation
) {
    if (b != 0 &&
        a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::length_error(
            std::string(operation) + ": size overflow"
        );
    }

    return a * b;
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) {
        if (count == 0 ||
            count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::invalid_argument(
                "DeviceBuffer: invalid allocation size"
            );
        }

        check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&pointer_),
                count * sizeof(T)
            ),
            "cudaMalloc"
        );
    }

    ~DeviceBuffer() {
        if (pointer_ != nullptr) {
            cudaFree(pointer_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    [[nodiscard]] T* get() noexcept {
        return pointer_;
    }

    [[nodiscard]] const T* get() const noexcept {
        return pointer_;
    }

private:
    T* pointer_ = nullptr;
};

unsigned int calculate_blocks(std::size_t count) {
    const std::size_t blocks =
        count / THREADS_PER_BLOCK +
        (count % THREADS_PER_BLOCK != 0);

    if (blocks > std::numeric_limits<unsigned int>::max()) {
        throw std::length_error("CUDA grid dimension is too large");
    }

    return static_cast<unsigned int>(blocks);
}

// ------------------------------------------------------------
// Prediction
// ------------------------------------------------------------

__global__ void predict_kernel(
    const float* X,
    const float* weights,
    const float* bias,
    double* predictions,
    std::size_t rows,
    std::size_t cols
) {
    const std::size_t row =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (row >= rows) {
        return;
    }

    double prediction = static_cast<double>(bias[0]);

    for (std::size_t col = 0; col < cols; ++col) {
        prediction +=
            static_cast<double>(X[row * cols + col]) *
            static_cast<double>(weights[col]);
    }

    predictions[row] = prediction;
}

// ------------------------------------------------------------
// Residual calculation
// ------------------------------------------------------------

__global__ void residual_kernel(
    const double* predictions,
    const float* y,
    double* residuals,
    std::size_t rows
) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index < rows) {
        residuals[index] =
            predictions[index] - static_cast<double>(y[index]);
    }
}

// ------------------------------------------------------------
// Parallel partial gradient reduction
//
// Each block processes:
//   ROW_TILE rows x FEATURE_TILE features.
//
// Thread layout:
//   threadIdx.x -> feature
//   threadIdx.y -> row
//
// Shared memory reduces partial sums over rows.
// ------------------------------------------------------------

__global__ void partial_gradient_kernel(
    const float* X,
    const double* residuals,
    double* partial_gradients,
    double* partial_bias,
    std::size_t rows,
    std::size_t cols
) {
    __shared__ double shared_gradients[ROW_TILE][FEATURE_TILE];
    __shared__ double shared_bias[ROW_TILE];

    const unsigned int local_col = threadIdx.x;
    const unsigned int local_row = threadIdx.y;

    const std::size_t col =
        static_cast<std::size_t>(blockIdx.x) * FEATURE_TILE + local_col;

    const std::size_t row =
        static_cast<std::size_t>(blockIdx.y) * ROW_TILE + local_row;

    double value = 0.0;

    if (row < rows && col < cols) {
        value =
            static_cast<double>(X[row * cols + col]) * residuals[row];
    }

    shared_gradients[local_row][local_col] = value;

    if (local_col == 0) {
        shared_bias[local_row] =
            row < rows ? residuals[row] : 0.0;
    }

    __syncthreads();

    // Reduce eight rows into one partial sum.
    for (unsigned int stride = ROW_TILE / 2;
         stride > 0;
         stride >>= 1) {

        if (local_row < stride) {
            shared_gradients[local_row][local_col] +=
                shared_gradients[local_row + stride][local_col];

            if (local_col == 0) {
                shared_bias[local_row] +=
                    shared_bias[local_row + stride];
            }
        }

        __syncthreads();
    }

    if (local_row == 0 && col < cols) {
        partial_gradients[
            static_cast<std::size_t>(blockIdx.y) * cols + col
        ] = shared_gradients[0][local_col];
    }

    // Bias partial is written only by the first feature tile.
    if (local_row == 0 &&
        local_col == 0 &&
        blockIdx.x == 0) {

        partial_bias[blockIdx.y] = shared_bias[0];
    }
}

// ------------------------------------------------------------
// Final gradient reduction
//
// Each thread reduces partial gradients for one feature.
// Thread zero also reduces the bias partial sums.
// ------------------------------------------------------------

__global__ void final_gradient_kernel(
    const double* partial_gradients,
    const double* partial_bias,
    double* gradients,
    double* bias_gradient,
    std::size_t cols,
    std::size_t row_tiles,
    std::size_t rows
) {
    const std::size_t col =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (col < cols) {
        double gradient = 0.0;

        for (std::size_t tile = 0; tile < row_tiles; ++tile) {
            gradient += partial_gradients[tile * cols + col];
        }

        gradients[col] = gradient / static_cast<double>(rows);
    }

    if (col == 0) {
        double gradient = 0.0;

        for (std::size_t tile = 0; tile < row_tiles; ++tile) {
            gradient += partial_bias[tile];
        }

        bias_gradient[0] =
            gradient / static_cast<double>(rows);
    }
}

// ------------------------------------------------------------
// Fused parameter update
//
// Thread zero updates bias.
// Other active threads update weights.
// ------------------------------------------------------------

__global__ void update_parameters_kernel(
    float* weights,
    float* bias,
    const double* gradients,
    const double* bias_gradient,
    float learning_rate,
    std::size_t cols,
    int* numerical_error
) {
    const std::size_t col =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (*numerical_error != 0) {
        return;
    }

    if (col < cols) {
        const double updated =
            static_cast<double>(weights[col]) -
            static_cast<double>(learning_rate) * gradients[col];

        if (!isfinite(updated) || fabs(updated) > static_cast<double>(FLT_MAX)) {
            atomicExch(numerical_error, 1);
        } else {
            weights[col] = static_cast<float>(updated);
        }
    }

    if (col == 0) {
        const double updated =
            static_cast<double>(bias[0]) -
            static_cast<double>(learning_rate) * bias_gradient[0];

        if (!isfinite(updated) || fabs(updated) > static_cast<double>(FLT_MAX)) {
            atomicExch(numerical_error, 1);
        } else {
            bias[0] = static_cast<float>(updated);
        }
    }
}

// ------------------------------------------------------------
// Input validation
// ------------------------------------------------------------

void validate_training_data(
    const Matrix& X,
    const Vector& y,
    float learning_rate,
    std::size_t epochs
) {
    if (X.rows() == 0 || X.cols() == 0) {
        throw std::invalid_argument(
            "CudaLinearRegression: training matrix must not be empty"
        );
    }

    if (X.rows() != y.size()) {
        throw std::invalid_argument(
            "CudaLinearRegression: feature and target row counts must match"
        );
    }

    if (y.empty()) {
        throw std::invalid_argument(
            "CudaLinearRegression: target vector must not be empty"
        );
    }

    if (!std::isfinite(learning_rate) || learning_rate <= 0.0f) {
        throw std::invalid_argument(
            "CudaLinearRegression: learning rate must be finite and positive"
        );
    }

    if (epochs == 0) {
        throw std::invalid_argument(
            "CudaLinearRegression: epochs must be greater than zero"
        );
    }

    for (std::size_t i = 0; i < X.size(); ++i) {
        if (!std::isfinite(X.data()[i])) {
            throw std::invalid_argument(
                "CudaLinearRegression: features must contain finite values"
            );
        }
    }

    for (std::size_t i = 0; i < y.size(); ++i) {
        if (!std::isfinite(y[i])) {
            throw std::invalid_argument(
                "CudaLinearRegression: targets must contain finite values"
            );
        }
    }
}

} // namespace

// ============================================================
// CUDA batch gradient descent
// ============================================================

void CudaLinearRegression::fit(
    const Matrix& X,
    const Vector& y,
    float learning_rate,
    std::size_t epochs
) {
    fitted_ = false;

    validate_training_data(X, y, learning_rate, epochs);

    const std::size_t rows = X.rows();
    const std::size_t cols = X.cols();

    const std::size_t row_tiles =
        rows / ROW_TILE + (rows % ROW_TILE != 0);

    const std::size_t feature_tiles =
        cols / FEATURE_TILE + (cols % FEATURE_TILE != 0);

    // CUDA grid limits for the chosen 2D grid.
    if (row_tiles > 65535 ||
        feature_tiles > 2147483647ULL) {
        throw std::length_error(
            "CudaLinearRegression: training dimensions exceed CUDA grid limits"
        );
    }

    const std::size_t matrix_bytes =
        checked_product(X.size(), sizeof(float), "Feature matrix");

    const std::size_t target_bytes =
        checked_product(y.size(), sizeof(float), "Target vector");

    const std::size_t weights_bytes =
        checked_product(cols, sizeof(float), "Model weights");

    const std::size_t partial_count =
        checked_product(row_tiles, cols, "Partial gradient matrix");

    weights_ = Vector(cols, 0.0f);
    bias_ = 0.0f;

    DeviceBuffer<float> device_X(X.size());
    DeviceBuffer<float> device_y(y.size());
    DeviceBuffer<float> device_weights(cols);
    DeviceBuffer<float> device_bias(1);

    DeviceBuffer<double> device_predictions(rows);
    DeviceBuffer<double> device_residuals(rows);

    DeviceBuffer<double> device_partial_gradients(partial_count);
    DeviceBuffer<double> device_partial_bias(row_tiles);

    DeviceBuffer<double> device_gradients(cols);
    DeviceBuffer<double> device_bias_gradient(1);

    DeviceBuffer<int> device_numerical_error(1);

    check_cuda(
        cudaMemcpy(
            device_X.get(),
            X.data(),
            matrix_bytes,
            cudaMemcpyHostToDevice
        ),
        "Copy training features to GPU"
    );

    check_cuda(
        cudaMemcpy(
            device_y.get(),
            y.data(),
            target_bytes,
            cudaMemcpyHostToDevice
        ),
        "Copy training targets to GPU"
    );

    check_cuda(
        cudaMemset(device_weights.get(), 0, weights_bytes),
        "Initialize GPU weights"
    );

    check_cuda(
        cudaMemset(device_bias.get(), 0, sizeof(float)),
        "Initialize GPU bias"
    );

    check_cuda(
        cudaMemset(device_numerical_error.get(), 0, sizeof(int)),
        "Initialize numerical error flag"
    );

    const unsigned int row_blocks = calculate_blocks(rows);
    const unsigned int col_blocks = calculate_blocks(cols);

    const dim3 partial_grid(
        static_cast<unsigned int>(feature_tiles),
        static_cast<unsigned int>(row_tiles),
        1
    );

    const dim3 partial_block(
        FEATURE_TILE,
        ROW_TILE,
        1
    );

    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {

        // 1. Predictions.
        predict_kernel<<<row_blocks, THREADS_PER_BLOCK>>>(
            device_X.get(),
            device_weights.get(),
            device_bias.get(),
            device_predictions.get(),
            rows,
            cols
        );

        check_cuda(
            cudaGetLastError(),
            "Launch prediction kernel"
        );

        // 2. Residuals.
        residual_kernel<<<row_blocks, THREADS_PER_BLOCK>>>(
            device_predictions.get(),
            device_y.get(),
            device_residuals.get(),
            rows
        );

        check_cuda(
            cudaGetLastError(),
            "Launch residual kernel"
        );

        // 3. Parallel partial gradients.
        partial_gradient_kernel<<<partial_grid, partial_block>>>(
            device_X.get(),
            device_residuals.get(),
            device_partial_gradients.get(),
            device_partial_bias.get(),
            rows,
            cols
        );

        check_cuda(
            cudaGetLastError(),
            "Launch partial gradient kernel"
        );

        // 4. Final gradient reduction.
        final_gradient_kernel<<<col_blocks, THREADS_PER_BLOCK>>>(
            device_partial_gradients.get(),
            device_partial_bias.get(),
            device_gradients.get(),
            device_bias_gradient.get(),
            cols,
            row_tiles,
            rows
        );

        check_cuda(
            cudaGetLastError(),
            "Launch final gradient kernel"
        );

        // 5. Fused weight and bias update.
        update_parameters_kernel<<<col_blocks, THREADS_PER_BLOCK>>>(
            device_weights.get(),
            device_bias.get(),
            device_gradients.get(),
            device_bias_gradient.get(),
            learning_rate,
            cols,
            device_numerical_error.get()
        );

        check_cuda(
            cudaGetLastError(),
            "Launch parameter update kernel"
        );
    }

    check_cuda(
        cudaDeviceSynchronize(),
        "Synchronize CUDA gradient descent"
    );

    int numerical_error = 0;

    check_cuda(
        cudaMemcpy(
            &numerical_error,
            device_numerical_error.get(),
            sizeof(int),
            cudaMemcpyDeviceToHost
        ),
        "Read numerical error flag"
    );

    if (numerical_error != 0) {
        throw std::overflow_error(
            "CudaLinearRegression: parameter update exceeds float range"
        );
    }

    check_cuda(
        cudaMemcpy(
            weights_.data(),
            device_weights.get(),
            weights_bytes,
            cudaMemcpyDeviceToHost
        ),
        "Copy trained weights from GPU"
    );

    check_cuda(
        cudaMemcpy(
            &bias_,
            device_bias.get(),
            sizeof(float),
            cudaMemcpyDeviceToHost
        ),
        "Copy trained bias from GPU"
    );

    for (std::size_t i = 0; i < weights_.size(); ++i) {
        if (!std::isfinite(weights_[i])) {
            throw std::runtime_error(
                "CudaLinearRegression: training produced non-finite weights"
            );
        }
    }

    if (!std::isfinite(bias_)) {
        throw std::runtime_error(
            "CudaLinearRegression: training produced non-finite bias"
        );
    }

    fitted_ = true;
}

// ============================================================
// Prediction
// ============================================================

Vector CudaLinearRegression::predict(const Matrix& X) const {
    if (!fitted_) {
        throw std::logic_error(
            "CudaLinearRegression: model must be fitted before prediction"
        );
    }

    if (X.cols() != weights_.size()) {
        throw std::invalid_argument(
            "CudaLinearRegression: prediction feature count does not match"
        );
    }

    if (X.rows() == 0) {
        return Vector{};
    }

    for (std::size_t i = 0; i < X.size(); ++i) {
        if (!std::isfinite(X.data()[i])) {
            throw std::invalid_argument(
                "CudaLinearRegression: prediction features must be finite"
            );
        }
    }

    const std::size_t rows = X.rows();
    const std::size_t cols = X.cols();

    const std::size_t matrix_bytes =
        checked_product(X.size(), sizeof(float), "Prediction matrix");

    const std::size_t weights_bytes =
        checked_product(cols, sizeof(float), "Prediction weights");

    DeviceBuffer<float> device_X(X.size());
    DeviceBuffer<float> device_weights(cols);
    DeviceBuffer<float> device_bias(1);
    DeviceBuffer<double> device_predictions(rows);

    check_cuda(
        cudaMemcpy(
            device_X.get(),
            X.data(),
            matrix_bytes,
            cudaMemcpyHostToDevice
        ),
        "Copy prediction features to GPU"
    );

    check_cuda(
        cudaMemcpy(
            device_weights.get(),
            weights_.data(),
            weights_bytes,
            cudaMemcpyHostToDevice
        ),
        "Copy model weights to GPU"
    );

    check_cuda(
        cudaMemcpy(
            device_bias.get(),
            &bias_,
            sizeof(float),
            cudaMemcpyHostToDevice
        ),
        "Copy model bias to GPU"
    );

    const unsigned int blocks = calculate_blocks(rows);

    predict_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        device_X.get(),
        device_weights.get(),
        device_bias.get(),
        device_predictions.get(),
        rows,
        cols
    );

    check_cuda(
        cudaGetLastError(),
        "Launch prediction kernel"
    );

    check_cuda(
        cudaDeviceSynchronize(),
        "Synchronize prediction kernel"
    );

    std::vector<double> host_predictions(rows);

    check_cuda(
        cudaMemcpy(
            host_predictions.data(),
            device_predictions.get(),
            checked_product(rows, sizeof(double), "Prediction output"),
            cudaMemcpyDeviceToHost
        ),
        "Copy predictions from GPU"
    );

    Vector predictions(rows);

    for (std::size_t i = 0; i < rows; ++i) {
        predictions[i] = checked_float(
            host_predictions[i],
            "CudaLinearRegression prediction"
        );
    }

    return predictions;
}

// ============================================================
// Model parameters
// ============================================================

const Vector& CudaLinearRegression::coefficients() const noexcept {
    return weights_;
}

float CudaLinearRegression::intercept() const noexcept {
    return bias_;
}

} // namespace cppml::models