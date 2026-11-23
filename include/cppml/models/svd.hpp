
#pragma once

#include "cppml/core/matrix.hpp"
#include "cppml/core/vector.hpp"
#include "cppml/models/least_squares.hpp"

namespace cppml::models {

class SVD {
public:
    [[nodiscard]] static LeastSquaresResult solve_pseudoinverse(
        const Matrix& X,
        const Vector& y
    );
};

} // namespace cppml::models
