
#include "cppml/math/operations.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
#include <string>

namespace cppml::math {
namespace {

void validate_finite(const Vector& v, const char* operation) {
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (!std::isfinite(v[i])) {
            throw std::invalid_argument(
                std::string(operation) + ": input contains NaN or infinity"
            );
        }
    }
}

void validate_finite(const Matrix& X, const char* operation) {
    for (std::size_t i = 0; i < X.size(); ++i) {
        if (!std::isfinite(X.data()[i])) {
            throw std::invalid_argument(
                std::string(operation) + ": input contains NaN or infinity"
            );
        }
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

} // namespace

Vector matvec(const Matrix& X, const Vector& w) {
    if (X.cols() != w.size()) {
        throw std::invalid_argument(
            "matvec: matrix columns must equal vector size"
        );
    }

    validate_finite(X, "matvec");
    validate_finite(w, "matvec");

    const std::size_t m = X.rows();
    const std::size_t n = X.cols();

    Vector result(m, 0.0f);

    for (std::size_t i = 0; i < m; ++i) {
        double accumulator = 0.0;

        for (std::size_t j = 0; j < n; ++j) {
            accumulator +=
                static_cast<double>(X(i, j)) *
                static_cast<double>(w[j]);
        }

        result[i] = checked_float(accumulator, "matvec");
    }

    return result;
}

Vector transpose_matvec(const Matrix& X, const Vector& r) {
    if (X.rows() != r.size()) {
        throw std::invalid_argument(
            "transpose_matvec: matrix rows must equal vector size"
        );
    }

    validate_finite(X, "transpose_matvec");
    validate_finite(r, "transpose_matvec");

    const std::size_t m = X.rows();
    const std::size_t n = X.cols();

    Vector result(n, 0.0f);
    std::vector<double> accumulators(n, 0.0);

    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            accumulators[j] +=
                static_cast<double>(X(i, j)) *
                static_cast<double>(r[i]);
        }
    }

    for (std::size_t j = 0; j < n; ++j) {
        result[j] = checked_float(
            accumulators[j],
            "transpose_matvec"
        );
    }

    return result;
}

Vector subtract(const Vector& a, const Vector& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument(
            "subtract: vector dimensions must match"
        );
    }

    validate_finite(a, "subtract");
    validate_finite(b, "subtract");

    Vector result(a.size());

    for (std::size_t i = 0; i < a.size(); ++i) {
        const double value =
            static_cast<double>(a[i]) -
            static_cast<double>(b[i]);

        result[i] = checked_float(value, "subtract");
    }

    return result;
}

float dot(const Vector& a, const Vector& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument(
            "dot: vector dimensions must match"
        );
    }

    validate_finite(a, "dot");
    validate_finite(b, "dot");

    double accumulator = 0.0;

    for (std::size_t i = 0; i < a.size(); ++i) {
        accumulator +=
            static_cast<double>(a[i]) *
            static_cast<double>(b[i]);
    }

    return checked_float(accumulator, "dot");
}

float sum(const Vector& v) {
    validate_finite(v, "sum");

    double accumulator = 0.0;

    for (std::size_t i = 0; i < v.size(); ++i) {
        accumulator += static_cast<double>(v[i]);
    }

    return checked_float(accumulator, "sum");
}

float mean(const Vector& v) {
    if (v.empty()) {
        throw std::invalid_argument(
            "mean: cannot calculate mean of an empty vector"
        );
    }

    validate_finite(v, "mean");

    double accumulator = 0.0;

    for (std::size_t i = 0; i < v.size(); ++i) {
        accumulator += static_cast<double>(v[i]);
    }

    return checked_float(
        accumulator / static_cast<double>(v.size()),
        "mean"
    );
}

Vector add_scalar(const Vector& v, float value) {
    validate_finite(v, "add_scalar");

    if (!std::isfinite(value)) {
        throw std::invalid_argument(
            "add_scalar: scalar must be finite"
        );
    }

    Vector result(v.size());

    for (std::size_t i = 0; i < v.size(); ++i) {
        const double sum =
            static_cast<double>(v[i]) +
            static_cast<double>(value);

        result[i] = checked_float(sum, "add_scalar");
    }

    return result;
}

} // namespace cppml::math
