
#pragma once

#include "cppml/core/matrix.hpp"
#include "cppml/core/vector.hpp"

namespace cppml::math {

// Matrix-vector multiplication: X(m x n) * w(n) -> Vector(m)
Vector matvec(const Matrix& X, const Vector& w);

// Transpose matrix-vector multiplication: X^T(n x m) * r(m) -> Vector(n)
Vector transpose_matvec(const Matrix& X, const Vector& r);

// Element-wise vector subtraction: a - b
Vector subtract(const Vector& a, const Vector& b);

// Dot product: a^T * b
float dot(const Vector& a, const Vector& b);

// Sum of vector elements
float sum(const Vector& v);

// Arithmetic mean of vector elements
float mean(const Vector& v);

// Add a scalar to every vector element
Vector add_scalar(const Vector& v, float value);

} // namespace cppml::math
