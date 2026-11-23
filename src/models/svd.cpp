
#include "cppml/models/svd.hpp"

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
            "SVD: feature matrix must not be empty"
        );
    }

    if (X.rows() != y.size()) {
        throw std::invalid_argument(
            "SVD: feature and target row counts must match"
        );
    }

    if (y.empty()) {
        throw std::invalid_argument(
            "SVD: target vector must not be empty"
        );
    }

    for (std::size_t i = 0; i < X.size(); ++i) {
        if (!std::isfinite(X.data()[i])) {
            throw std::invalid_argument(
                "SVD: features must contain finite values"
            );
        }
    }

    for (std::size_t i = 0; i < y.size(); ++i) {
        if (!std::isfinite(y[i])) {
            throw std::invalid_argument(
                "SVD: targets must contain finite values"
            );
        }
    }
}

} // namespace

LeastSquaresResult SVD::solve_pseudoinverse(
    const Matrix& X,
    const Vector& y
) {
    validate_input(X, y);

    const std::size_t rows = X.rows();
    const std::size_t features = X.cols();
    const std::size_t columns = features + 1;

    if (columns > std::numeric_limits<std::size_t>::max() / rows) {
        throw std::length_error(
            "SVD: augmented matrix dimensions overflow"
        );
    }

    if (columns > std::numeric_limits<std::size_t>::max() / columns) {
        throw std::length_error(
            "SVD: right singular vector matrix dimensions overflow"
        );
    }

    // Build augmented design matrix A = [X | 1].
    // B initially stores A and is transformed into A * V.
    std::vector<double> B(rows * columns, 0.0);
    std::vector<double> V(columns * columns, 0.0);
    std::vector<double> target(rows, 0.0);

    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < features; ++j) {
            B[i * columns + j] = static_cast<double>(X(i, j));
        }

        B[i * columns + features] = 1.0;
        target[i] = static_cast<double>(y[i]);
    }

    // Initialize V as the identity matrix.
    for (std::size_t i = 0; i < columns; ++i) {
        V[i * columns + i] = 1.0;
    }

    // One-sided Jacobi SVD.
    //
    // Repeatedly orthogonalize pairs of columns of B.
    // The same rotations are applied to V.
    //
    // At convergence:
    // B = A * V, and columns of B are mutually orthogonal.
    constexpr std::size_t max_sweeps = 100;
    constexpr double orthogonality_tolerance =
        8.0 * std::numeric_limits<double>::epsilon();

    bool converged = false;

    for (std::size_t sweep = 0; sweep < max_sweeps; ++sweep) {
        bool changed = false;

        for (std::size_t p = 0; p < columns; ++p) {
            for (std::size_t q = p + 1; q < columns; ++q) {
                double alpha = 0.0;
                double beta = 0.0;
                double gamma = 0.0;

                for (std::size_t i = 0; i < rows; ++i) {
                    const double bp = B[i * columns + p];
                    const double bq = B[i * columns + q];

                    alpha += bp * bp;
                    beta += bq * bq;
                    gamma += bp * bq;
                }

                if (!std::isfinite(alpha) ||
                    !std::isfinite(beta) ||
                    !std::isfinite(gamma)) {
                    throw std::runtime_error(
                        "SVD: non-finite values during Jacobi iteration"
                    );
                }

                if (alpha == 0.0 || beta == 0.0) {
                    continue;
                }

                const double scale = std::sqrt(alpha) * std::sqrt(beta);

                if (std::abs(gamma) <=
                    orthogonality_tolerance * scale) {
                    continue;
                }

                // Stable Jacobi rotation.
                const double zeta = (beta - alpha) / (2.0 * gamma);

                const double t =
                    std::copysign(1.0, zeta) /
                    (std::abs(zeta) + std::hypot(1.0, zeta));

                const double c = 1.0 / std::sqrt(1.0 + t * t);
                const double s = c * t;

                // Rotate columns p and q of B.
                for (std::size_t i = 0; i < rows; ++i) {
                    const double bp = B[i * columns + p];
                    const double bq = B[i * columns + q];

                    B[i * columns + p] = c * bp - s * bq;
                    B[i * columns + q] = s * bp + c * bq;
                }

                // Apply identical rotation to V.
                for (std::size_t i = 0; i < columns; ++i) {
                    const double vp = V[i * columns + p];
                    const double vq = V[i * columns + q];

                    V[i * columns + p] = c * vp - s * vq;
                    V[i * columns + q] = s * vp + c * vq;
                }

                changed = true;
            }
        }

        if (!changed) {
            converged = true;
            break;
        }
    }

    if (!converged) {
        throw std::runtime_error(
            "SVD: one-sided Jacobi iteration did not converge"
        );
    }

    // Singular values are the norms of the orthogonalized columns.
    std::vector<double> singular_values(columns, 0.0);
    double maximum_singular_value = 0.0;

    for (std::size_t j = 0; j < columns; ++j) {
        double norm_squared = 0.0;

        for (std::size_t i = 0; i < rows; ++i) {
            const double value = B[i * columns + j];
            norm_squared += value * value;
        }

        singular_values[j] = std::sqrt(norm_squared);

        if (!std::isfinite(singular_values[j])) {
            throw std::runtime_error(
                "SVD: non-finite singular value"
            );
        }

        maximum_singular_value =
            std::max(maximum_singular_value, singular_values[j]);
    }

    // Numerical rank threshold based on float input precision.
    const double rank_tolerance =
        static_cast<double>(std::numeric_limits<float>::epsilon()) *
        static_cast<double>(std::max(rows, columns)) *
        maximum_singular_value;

    // Pseudoinverse solution:
    //
    // theta = V * diag(1 / sigma_j^2) * B^T * y
    //
    // Singular directions below the threshold are discarded.
    std::vector<double> theta(columns, 0.0);

    for (std::size_t j = 0; j < columns; ++j) {
        const double sigma = singular_values[j];

        if (sigma <= rank_tolerance) {
            continue;
        }

        double projection = 0.0;

        for (std::size_t i = 0; i < rows; ++i) {
            projection += B[i * columns + j] * target[i];
        }

        const double factor = projection / (sigma * sigma);

        for (std::size_t k = 0; k < columns; ++k) {
            theta[k] += V[k * columns + j] * factor;
        }
    }

    LeastSquaresResult result;
    result.coefficients = Vector(features);

    for (std::size_t j = 0; j < features; ++j) {
        result.coefficients[j] = static_cast<float>(theta[j]);

        if (!std::isfinite(theta[j]) ||
            !std::isfinite(result.coefficients[j])) {
            throw std::runtime_error(
                "SVD: coefficient is outside float range"
            );
        }
    }

    result.intercept = static_cast<float>(theta[features]);

    if (!std::isfinite(theta[features]) ||
        !std::isfinite(result.intercept)) {
        throw std::runtime_error(
            "SVD: intercept is outside float range"
        );
    }

    return result;
}

} // namespace cppml::models
