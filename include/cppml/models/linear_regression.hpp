
#pragma once

#include "cppml/core/matrix.hpp"
#include "cppml/core/vector.hpp"
#include "cppml/models/execution_backend.hpp"

#ifdef CPPML_HAS_CUDA
#include "cppml/models/cuda_linear_regression.hpp"
#endif

#include <cstddef>

namespace cppml::models {

class LinearRegression {
public:
    explicit LinearRegression(
        ExecutionBackend backend = ExecutionBackend::CPU
    );

    void fit(
        const Matrix& X,
        const Vector& y,
        float learning_rate,
        std::size_t epochs
    );

    [[nodiscard]] Vector predict(const Matrix& X) const;

    [[nodiscard]] const Vector& coefficients() const noexcept;

    [[nodiscard]] float intercept() const noexcept;

    [[nodiscard]] ExecutionBackend backend() const noexcept;

    [[nodiscard]] ExecutionBackend active_backend() const noexcept;

private:
    void fit_cpu(
        const Matrix& X,
        const Vector& y,
        float learning_rate,
        std::size_t epochs
    );

    [[nodiscard]] bool should_use_cuda(
        std::size_t rows,
        std::size_t cols,
        std::size_t epochs
    ) const noexcept;

    ExecutionBackend backend_;
    ExecutionBackend active_backend_ = ExecutionBackend::CPU;

    Vector weights_;
    float bias_ = 0.0f;
    bool fitted_ = false;

#ifdef CPPML_HAS_CUDA
    CudaLinearRegression cuda_model_;
#endif
};

} // namespace cppml::models
