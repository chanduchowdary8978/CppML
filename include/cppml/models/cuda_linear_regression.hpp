
#pragma once

#include "cppml/core/matrix.hpp"
#include "cppml/core/vector.hpp"

#include <cstddef>

namespace cppml::models {

class CudaLinearRegression {
public:
    void fit(
        const Matrix& X,
        const Vector& y,
        float learning_rate,
        std::size_t epochs
    );

    [[nodiscard]] Vector predict(const Matrix& X) const;

    [[nodiscard]] const Vector& coefficients() const noexcept;

    [[nodiscard]] float intercept() const noexcept;

private:
    Vector weights_;
    float bias_ = 0.0f;
    bool fitted_ = false;
};

} // namespace cppml::models
