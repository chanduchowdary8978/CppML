
#pragma once

#include "cppml/core/matrix.hpp"
#include "cppml/core/vector.hpp"

namespace cppml::math::cuda {

Vector matvec(const Matrix& X, const Vector& w);

Vector transpose_matvec(const Matrix& X, const Vector& r);

Vector subtract(const Vector& a, const Vector& b);

float dot(const Vector& a, const Vector& b);

float sum(const Vector& v);

float mean(const Vector& v);

Vector add_scalar(const Vector& v, float value);

} // namespace cppml::math::cuda
