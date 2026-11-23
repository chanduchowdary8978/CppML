
#include "cppml/models/cuda_linear_regression.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace cppml::models {
namespace {

constexpr unsigned int THREADS_PER_BLOCK = 256;

void check_cuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(error)
        );
    }
}

float checked_float(double value, const char* operation) {
    if (!std::isfinite(value) ||
        std::abs(value) > static_cast<double>(
            std::numeric_limits<float>::max()
        )) {
        throw std::overflow_error(
            std::string(operation) + ": result exceeds float range"
        );
    }

    return static_cast<float>(value);
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
// Prediction: double accumulation and double output
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
// Residuals: double precision
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
// Weight gradients: double accumulation
// ------------------------------------------------------------

__global__ void weight_gradient_kernel(
    const float* X,
    const double* residuals,
    double* gradients,
    std::size_t rows,
    std::size_t cols
) {
    const std::size_t col =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (col >= cols) {
        return;
    }

    double gradient = 0.0;

    for (std::size_t row = 0; row < rows; ++row) {
        gradient +=
            static_cast<double>(X[row * cols + col]) * residuals[row];
    }

    gradients[col] = gradient / static_cast<double>(rows);
}

// ------------------------------------------------------------
// Bias gradient: double accumulation
// ------------------------------------------------------------

__global__ void bias_gradient_kernel(
    const double* residuals,
    double* bias_gradient,
    std::size_t rows
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    double gradient = 0.0;

    for (std::size_t row = 0; row < rows; ++row) {
        gradient += residuals[row];
    }

    bias_gradient[0] = gradient / static_cast<double>(rows);
}

// ------------------------------------------------------------
// Weight update: checked float conversion
// ------------------------------------------------------------

__global__ void update_weights_kernel(
    float* weights,
    const double* gradients,
    float learning_rate,
    std::size_t cols,
    int* numerical_error
) {
    const std::size_t col =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (col >= cols) {
        return;
    }

    const double updated =
        static_cast<double>(weights[col]) -
        static_cast<double>(learning_rate) * gradients[col];

    if (!isfinite(updated) ||
        fabs(updated) > static_cast<double>(FLT_MAX)) {
        atomicExch(numerical_error, 1);
        return;
    }

    weights[col] = static_cast<float>(updated);
}

// ------------------------------------------------------------
// Bias update: checked float conversion
// ------------------------------------------------------------

__global__ void update_bias_kernel(
    float* bias,
    const double* bias_gradient,
    float learning_rate,
    int* numerical_error
) {
    if (blockIdx.x != 0 || threadIdx.x != 0 ||
        *numerical_error != 0) {
        return;
    }

    const double updated =
        static_cast<double>(bias[0]) -
        static_cast<double>(learning_rate) * bias_gradient[0];

    if (!isfinite(updated) ||
        fabs(updated) > static_cast<double>(FLT_MAX)) {
        atomicExch(numerical_error, 1);
        return;
    }

    bias[0] = static_cast<float>(updated);
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

    const std::size_t matrix_bytes = X.size() * sizeof(float);
    const std::size_t target_bytes = y.size() * sizeof(float);
    const std::size_t weights_bytes = cols * sizeof(float);

    weights_ = Vector(cols, 0.0f);
    bias_ = 0.0f;

    DeviceBuffer<float> device_X(X.size());
    DeviceBuffer<float> device_y(y.size());
    DeviceBuffer<float> device_weights(cols);
    DeviceBuffer<float> device_bias(1);

    DeviceBuffer<double> device_predictions(rows);
    DeviceBuffer<double> device_residuals(rows);
    DeviceBuffer<double> device_weight_gradients(cols);
    DeviceBuffer<double> device_bias_gradient(1);

    DeviceBuffer<int> device_numerical_error(1);

    check_cuda(
        cudaMemcpy(
            device_X.get(), X.data(), matrix_bytes,
            cudaMemcpyHostToDevice
        ),
        "Copy training features to GPU"
    );

    check_cuda(
        cudaMemcpy(
            device_y.get(), y.data(), target_bytes,
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

    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
        predict_kernel<<<row_blocks, THREADS_PER_BLOCK>>>(
            device_X.get(),
            device_weights.get(),
            device_bias.get(),
            device_predictions.get(),
            rows,
            cols
        );

        check_cuda(cudaGetLastError(), "Launch prediction kernel");

        residual_kernel<<<row_blocks, THREADS_PER_BLOCK>>>(
            device_predictions.get(),
            device_y.get(),
            device_residuals.get(),
            rows
        );

        check_cuda(cudaGetLastError(), "Launch residual kernel");

        weight_gradient_kernel<<<col_blocks, THREADS_PER_BLOCK>>>(
            device_X.get(),
            device_residuals.get(),
            device_weight_gradients.get(),
            rows,
            cols
        );

        check_cuda(cudaGetLastError(), "Launch weight gradient kernel");

        bias_gradient_kernel<<<1, 1>>>(
            device_residuals.get(),
            device_bias_gradient.get(),
            rows
        );

        check_cuda(cudaGetLastError(), "Launch bias gradient kernel");

        update_weights_kernel<<<col_blocks, THREADS_PER_BLOCK>>>(
            device_weights.get(),
            device_weight_gradients.get(),
            learning_rate,
            cols,
            device_numerical_error.get()
        );

        check_cuda(cudaGetLastError(), "Launch weight update kernel");

        update_bias_kernel<<<1, 1>>>(
            device_bias.get(),
            device_bias_gradient.get(),
            learning_rate,
            device_numerical_error.get()
        );

        check_cuda(cudaGetLastError(), "Launch bias update kernel");
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

    const std::size_t matrix_bytes = X.size() * sizeof(float);
    const std::size_t weights_bytes = cols * sizeof(float);

    DeviceBuffer<float> device_X(X.size());
    DeviceBuffer<float> device_weights(cols);
    DeviceBuffer<float> device_bias(1);
    DeviceBuffer<double> device_predictions(rows);

    check_cuda(
        cudaMemcpy(
            device_X.get(), X.data(), matrix_bytes,
            cudaMemcpyHostToDevice
        ),
        "Copy prediction features to GPU"
    );

    check_cuda(
        cudaMemcpy(
            device_weights.get(), weights_.data(), weights_bytes,
            cudaMemcpyHostToDevice
        ),
        "Copy model weights to GPU"
    );

    check_cuda(
        cudaMemcpy(
            device_bias.get(), &bias_, sizeof(float),
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

    check_cuda(cudaGetLastError(), "Launch prediction kernel");

    check_cuda(
        cudaDeviceSynchronize(),
        "Synchronize prediction kernel"
    );

    std::vector<double> host_predictions(rows);

    check_cuda(
        cudaMemcpy(
            host_predictions.data(),
            device_predictions.get(),
            rows * sizeof(double),
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
