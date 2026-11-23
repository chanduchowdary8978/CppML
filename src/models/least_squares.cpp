
#include "cppml/models/least_squares.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace cppml::models {
namespace {

void validate_input(const Matrix& X, const Vector& y) {
    if (X.rows() == 0 || X.cols() == 0) {
        throw std::invalid_argument(
            "LeastSquares: feature matrix must not be empty"
        );
    }

    if (X.rows() != y.size()) {
        throw std::invalid_argument(
            "LeastSquares: feature and target row counts must match"
        );
    }

    if (y.empty()) {
        throw std::invalid_argument(
            "LeastSquares: target vector must not be empty"
        );
    }

    for (std::size_t i = 0; i < X.size(); ++i) {
        if (!std::isfinite(X.data()[i])) {
            throw std::invalid_argument(
                "LeastSquares: features must contain finite values"
            );
        }
    }

    for (std::size_t i = 0; i < y.size(); ++i) {
        if (!std::isfinite(y[i])) {
            throw std::invalid_argument(
                "LeastSquares: targets must contain finite values"
            );
        }
    }
}

} // namespace

LeastSquaresResult LeastSquares::solve_qr(
    const Matrix& X,
    const Vector& y
) {
    validate_input(X, y);

    const std::size_t rows = X.rows();
    const std::size_t features = X.cols();
    const std::size_t parameters = features + 1;

    if (rows < parameters) {
        throw std::invalid_argument(
            "LeastSquares: insufficient samples for a unique QR solution"
        );
    }

    // Construct augmented design matrix [X | 1].
    // Internal calculations use double precision.
    std::vector<double> A(rows * parameters, 0.0);
    std::vector<double> transformed_y(rows, 0.0);
    std::vector<double> column_scales(parameters, 0.0);

    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < features; ++j) {
            A[i * parameters + j] =
                static_cast<double>(X(i, j));
        }

        A[i * parameters + features] = 1.0;
        transformed_y[i] = static_cast<double>(y[i]);
    }

    // Column equilibration:
    // Normalize each column to unit L2 norm.
    // This prevents large-scale features from dominating
    // the numerical rank tolerance of small-scale features.
    for (std::size_t j = 0; j < parameters; ++j) {
        double norm_squared = 0.0;

        for (std::size_t i = 0; i < rows; ++i) {
            const double value = A[i * parameters + j];
            norm_squared += value * value;
        }

        const double column_norm = std::sqrt(norm_squared);

        if (!std::isfinite(column_norm) || column_norm == 0.0) {
            throw std::domain_error(
                "LeastSquares: augmented design matrix is rank deficient"
            );
        }

        column_scales[j] = column_norm;

        for (std::size_t i = 0; i < rows; ++i) {
            A[i * parameters + j] /= column_norm;
        }
    }

    // The columns are normalized, so the tolerance no longer
    // depends on the largest original feature magnitude.
    const double tolerance =
        static_cast<double>(std::numeric_limits<float>::epsilon()) *
        static_cast<double>(std::max(rows, parameters));

    // Householder QR factorization.
    std::vector<double> reflector(rows, 0.0);

    for (std::size_t k = 0; k < parameters; ++k) {
        double norm_squared = 0.0;

        for (std::size_t i = k; i < rows; ++i) {
            const double value = A[i * parameters + k];
            norm_squared += value * value;
        }

        const double column_norm = std::sqrt(norm_squared);

        if (!std::isfinite(column_norm) ||
            column_norm <= tolerance) {
            throw std::domain_error(
                "LeastSquares: augmented design matrix is rank deficient"
            );
        }

        const double diagonal = A[k * parameters + k];
        const double alpha = -std::copysign(column_norm, diagonal);

        std::fill(reflector.begin(), reflector.end(), 0.0);

        for (std::size_t i = k; i < rows; ++i) {
            reflector[i] = A[i * parameters + k];
        }

        reflector[k] -= alpha;

        double reflector_norm_squared = 0.0;

        for (std::size_t i = k; i < rows; ++i) {
            reflector_norm_squared +=
                reflector[i] * reflector[i];
        }

        if (!std::isfinite(reflector_norm_squared) ||
            reflector_norm_squared <= 0.0) {
            throw std::domain_error(
                "LeastSquares: invalid Householder reflector"
            );
        }

        const double beta = 2.0 / reflector_norm_squared;

        // Apply reflector to remaining matrix columns.
        for (std::size_t j = k; j < parameters; ++j) {
            double projection = 0.0;

            for (std::size_t i = k; i < rows; ++i) {
                projection +=
                    reflector[i] * A[i * parameters + j];
            }

            projection *= beta;

            for (std::size_t i = k; i < rows; ++i) {
                A[i * parameters + j] -=
                    projection * reflector[i];
            }
        }

        // Apply the same reflector to the target vector.
        double y_projection = 0.0;

        for (std::size_t i = k; i < rows; ++i) {
            y_projection += reflector[i] * transformed_y[i];
        }

        y_projection *= beta;

        for (std::size_t i = k; i < rows; ++i) {
            transformed_y[i] -= y_projection * reflector[i];
        }

        // Maintain upper-triangular structure.
        A[k * parameters + k] = alpha;

        for (std::size_t i = k + 1; i < rows; ++i) {
            A[i * parameters + k] = 0.0;
        }
    }

    // Back substitution: R * scaled_theta = Q^T * y.
    std::vector<double> scaled_theta(parameters, 0.0);

    for (std::size_t reverse = 0; reverse < parameters; ++reverse) {
        const std::size_t i = parameters - 1 - reverse;

        double value = transformed_y[i];

        for (std::size_t j = i + 1; j < parameters; ++j) {
            value -= A[i * parameters + j] * scaled_theta[j];
        }

        const double diagonal = A[i * parameters + i];

        if (!std::isfinite(diagonal) ||
            std::abs(diagonal) <= tolerance) {
            throw std::domain_error(
                "LeastSquares: matrix is numerically rank deficient"
            );
        }

        scaled_theta[i] = value / diagonal;

        if (!std::isfinite(scaled_theta[i])) {
            throw std::runtime_error(
                "LeastSquares: QR solution produced non-finite parameters"
            );
        }
    }

    // Undo column scaling to recover original coefficients.
    LeastSquaresResult result;
    result.coefficients = Vector(features);

    for (std::size_t j = 0; j < features; ++j) {
        const double coefficient =
            scaled_theta[j] / column_scales[j];

        result.coefficients[j] = static_cast<float>(coefficient);

        if (!std::isfinite(coefficient) ||
            !std::isfinite(result.coefficients[j])) {
            throw std::runtime_error(
                "LeastSquares: coefficient is outside float range"
            );
        }
    }

    const double intercept =
        scaled_theta[features] / column_scales[features];

    result.intercept = static_cast<float>(intercept);

    if (!std::isfinite(intercept) ||
        !std::isfinite(result.intercept)) {
        throw std::runtime_error(
            "LeastSquares: intercept is outside float range"
        );
    }

    return result;
}

} // namespace cppml::models
