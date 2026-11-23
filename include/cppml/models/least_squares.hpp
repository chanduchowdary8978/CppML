
#pragma once

#include "cppml/core/matrix.hpp"
#include "cppml/core/vector.hpp"

namespace cppml::models {

struct LeastSquaresResult {
    Vector coefficients;
    float intercept = 0.0f;
};

class LeastSquares {
public:
    // Solves min ||Xw + b - y||^2 using Householder QR.
    // Requires at least as many samples as parameters, including intercept,
    // and a full-column-rank augmented design matrix.
    [[nodiscard]] static LeastSquaresResult solve_qr(
        const Matrix& X,
        const Vector& y
    );
};

} // namespace cppml::models
