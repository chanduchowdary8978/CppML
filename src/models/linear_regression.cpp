
#include "cppml/models/linear_regression.hpp"

#include <cmath>
#include <cstddef>
#include <algorithm>
#include <vector>
#include <limits>
#include <stdexcept>

#ifdef CPPML_HAS_CUDA
#include <cuda_runtime_api.h>
#endif

namespace cppml::models {
namespace {

// Provisional threshold based on RTX 3050 benchmark measurements.
constexpr std::size_t AUTO_CUDA_WORK_THRESHOLD = 1'000'000;

std::size_t saturating_multiply(
    std::size_t a,
    std::size_t b
) noexcept {
    if (a != 0 &&
        b > std::numeric_limits<std::size_t>::max() / a) {
        return std::numeric_limits<std::size_t>::max();
    }

    return a * b;
}

float checked_float(double value, const char* operation) {
    if (!std::isfinite(value) ||
        std::abs(value) > static_cast<double>(
            std::numeric_limits<float>::max()
        )) {
        throw std::overflow_error(operation);
    }

    return static_cast<float>(value);
}

void validate_training_data(
    const Matrix& X,
    const Vector& y,
    float learning_rate,
    std::size_t epochs
) {
    if (X.rows() == 0 || X.cols() == 0) {
        throw std::invalid_argument(
            "LinearRegression: training matrix must not be empty"
        );
    }

    if (X.rows() != y.size()) {
        throw std::invalid_argument(
            "LinearRegression: feature and target row counts must match"
        );
    }

    if (y.empty()) {
        throw std::invalid_argument(
            "LinearRegression: target vector must not be empty"
        );
    }

    if (!std::isfinite(learning_rate) || learning_rate <= 0.0f) {
        throw std::invalid_argument(
            "LinearRegression: learning rate must be finite and positive"
        );
    }

    if (epochs == 0) {
        throw std::invalid_argument(
            "LinearRegression: epochs must be greater than zero"
        );
    }

    for (std::size_t i = 0; i < X.size(); ++i) {
        if (!std::isfinite(X.data()[i])) {
            throw std::invalid_argument(
                "LinearRegression: features must contain finite values"
            );
        }
    }

    for (std::size_t i = 0; i < y.size(); ++i) {
        if (!std::isfinite(y[i])) {
            throw std::invalid_argument(
                "LinearRegression: targets must contain finite values"
            );
        }
    }
}

bool cuda_device_available() noexcept {
#ifdef CPPML_HAS_CUDA
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);

    if (status != cudaSuccess) {
        cudaGetLastError();
        return false;
    }

    return device_count > 0;
#else
    return false;
#endif
}

} // namespace

LinearRegression::LinearRegression(ExecutionBackend backend)
    : backend_(backend) {

    if (backend != ExecutionBackend::CPU &&
        backend != ExecutionBackend::CUDA &&
        backend != ExecutionBackend::AUTO) {
        throw std::invalid_argument(
            "LinearRegression: invalid execution backend"
        );
    }

#ifndef CPPML_HAS_CUDA
    if (backend_ == ExecutionBackend::CUDA) {
        throw std::runtime_error(
            "LinearRegression: CUDA backend is not enabled in this build"
        );
    }
#endif
}

bool LinearRegression::should_use_cuda(
    std::size_t rows,
    std::size_t cols,
    std::size_t epochs
) const noexcept {

    if (backend_ == ExecutionBackend::CPU) {
        return false;
    }

    if (backend_ == ExecutionBackend::CUDA) {
        return true;
    }

    if (!cuda_device_available()) {
        return false;
    }

    const std::size_t operations_per_epoch =
        saturating_multiply(rows, cols);

    const std::size_t estimated_work =
        saturating_multiply(operations_per_epoch, epochs);

    return estimated_work >= AUTO_CUDA_WORK_THRESHOLD;
}

void LinearRegression::fit(
    const Matrix& X,
    const Vector& y,
    float learning_rate,
    std::size_t epochs
) {
    fitted_ = false;

    validate_training_data(X, y, learning_rate, epochs);

    const bool use_cuda = should_use_cuda(
        X.rows(),
        X.cols(),
        epochs
    );

    if (use_cuda) {
#ifdef CPPML_HAS_CUDA
        cuda_model_.fit(X, y, learning_rate, epochs);

        weights_ = cuda_model_.coefficients();
        bias_ = cuda_model_.intercept();

        active_backend_ = ExecutionBackend::CUDA;
        fitted_ = true;
        return;
#else
        throw std::runtime_error(
            "LinearRegression: CUDA backend is unavailable"
        );
#endif
    }

    fit_cpu(X, y, learning_rate, epochs);

    active_backend_ = ExecutionBackend::CPU;
    fitted_ = true;
}

void LinearRegression::fit_cpu(
    const Matrix& X,
    const Vector& y,
    float learning_rate,
    std::size_t epochs
) {
    const std::size_t rows = X.rows();
    const std::size_t cols = X.cols();

    weights_ = Vector(cols, 0.0f);
    bias_ = 0.0f;

    const double inverse_rows =
        1.0 / static_cast<double>(rows);

    const double learning_rate_double =
        static_cast<double>(learning_rate);

    std::vector<double> gradients(cols, 0.0);

    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
        std::fill(gradients.begin(), gradients.end(), 0.0);

        double bias_gradient = 0.0;

        for (std::size_t i = 0; i < rows; ++i) {
            double prediction = static_cast<double>(bias_);

            for (std::size_t j = 0; j < cols; ++j) {
                prediction +=
                    static_cast<double>(X(i, j)) *
                    static_cast<double>(weights_[j]);
            }

            if (!std::isfinite(prediction)) {
                throw std::overflow_error(
                    "LinearRegression: prediction overflow during training"
                );
            }

            const double residual =
                prediction - static_cast<double>(y[i]);

            if (!std::isfinite(residual)) {
                throw std::overflow_error(
                    "LinearRegression: residual overflow during training"
                );
            }

            bias_gradient += residual;

            for (std::size_t j = 0; j < cols; ++j) {
                gradients[j] +=
                    static_cast<double>(X(i, j)) * residual;

                if (!std::isfinite(gradients[j])) {
                    throw std::overflow_error(
                        "LinearRegression: gradient overflow during training"
                    );
                }
            }

            if (!std::isfinite(bias_gradient)) {
                throw std::overflow_error(
                    "LinearRegression: bias gradient overflow during training"
                );
            }
        }

        bias_gradient *= inverse_rows;

        for (std::size_t j = 0; j < cols; ++j) {
            gradients[j] *= inverse_rows;

            const double updated_weight =
                static_cast<double>(weights_[j]) -
                learning_rate_double * gradients[j];

            weights_[j] = checked_float(
                updated_weight,
                "LinearRegression: weight update exceeds float range"
            );
        }

        const double updated_bias =
            static_cast<double>(bias_) -
            learning_rate_double * bias_gradient;

        bias_ = checked_float(
            updated_bias,
            "LinearRegression: bias update exceeds float range"
        );
    }
}

Vector LinearRegression::predict(const Matrix& X) const {
    if (!fitted_) {
        throw std::logic_error(
            "LinearRegression: model must be fitted before prediction"
        );
    }

    if (X.cols() != weights_.size()) {
        throw std::invalid_argument(
            "LinearRegression: prediction feature count does not match"
        );
    }

    if (X.rows() == 0) {
        return Vector{};
    }

    for (std::size_t i = 0; i < X.size(); ++i) {
        if (!std::isfinite(X.data()[i])) {
            throw std::invalid_argument(
                "LinearRegression: prediction features must be finite"
            );
        }
    }

#ifdef CPPML_HAS_CUDA
    if (active_backend_ == ExecutionBackend::CUDA) {
        return cuda_model_.predict(X);
    }
#endif

    Vector predictions(X.rows(), 0.0f);

    for (std::size_t i = 0; i < X.rows(); ++i) {
        double prediction = static_cast<double>(bias_);

        for (std::size_t j = 0; j < X.cols(); ++j) {
            prediction +=
                static_cast<double>(X(i, j)) *
                static_cast<double>(weights_[j]);
        }

        predictions[i] = checked_float(
            prediction,
            "LinearRegression: prediction exceeds float range"
        );
    }

    return predictions;
}

const Vector& LinearRegression::coefficients() const noexcept {
    return weights_;
}

float LinearRegression::intercept() const noexcept {
    return bias_;
}

ExecutionBackend LinearRegression::backend() const noexcept {
    return backend_;
}

ExecutionBackend LinearRegression::active_backend() const noexcept {
    return active_backend_;
}

} // namespace cppml::models
