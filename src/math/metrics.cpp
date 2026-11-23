
#include "cppml/math/metrics.hpp"

#include <cmath>
#include <stdexcept>

namespace cppml::math {

void Metrics::validate_inputs(
    const Vector& y_true,
    const Vector& y_pred
) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument(
            "Metrics: target and prediction dimensions must match"
        );
    }

    if (y_true.empty()) {
        throw std::invalid_argument(
            "Metrics: cannot calculate metric for empty vectors"
        );
    }

    for (std::size_t i = 0; i < y_true.size(); ++i) {
        if (!std::isfinite(y_true[i]) || !std::isfinite(y_pred[i])) {
            throw std::invalid_argument(
                "Metrics: inputs must contain only finite values"
            );
        }
    }
}

float Metrics::mean_squared_error(
    const Vector& y_true,
    const Vector& y_pred
) {
    validate_inputs(y_true, y_pred);

    double squared_error_sum = 0.0;

    for (std::size_t i = 0; i < y_true.size(); ++i) {
        const double error =
            static_cast<double>(y_true[i]) -
            static_cast<double>(y_pred[i]);

        squared_error_sum += error * error;
    }

    return static_cast<float>(
        squared_error_sum / static_cast<double>(y_true.size())
    );
}

float Metrics::root_mean_squared_error(
    const Vector& y_true,
    const Vector& y_pred
) {
    return std::sqrt(mean_squared_error(y_true, y_pred));
}

float Metrics::mean_absolute_error(
    const Vector& y_true,
    const Vector& y_pred
) {
    validate_inputs(y_true, y_pred);

    double absolute_error_sum = 0.0;

    for (std::size_t i = 0; i < y_true.size(); ++i) {
        const double error =
            static_cast<double>(y_true[i]) -
            static_cast<double>(y_pred[i]);

        absolute_error_sum += std::abs(error);
    }

    return static_cast<float>(
        absolute_error_sum / static_cast<double>(y_true.size())
    );
}

float Metrics::r_squared(
    const Vector& y_true,
    const Vector& y_pred
) {
    validate_inputs(y_true, y_pred);

    double target_sum = 0.0;

    for (std::size_t i = 0; i < y_true.size(); ++i) {
        target_sum += static_cast<double>(y_true[i]);
    }

    const double target_mean =
        target_sum / static_cast<double>(y_true.size());

    double residual_sum_squares = 0.0;
    double total_sum_squares = 0.0;

    for (std::size_t i = 0; i < y_true.size(); ++i) {
        const double actual = static_cast<double>(y_true[i]);
        const double predicted = static_cast<double>(y_pred[i]);

        const double residual = actual - predicted;
        const double deviation = actual - target_mean;

        residual_sum_squares += residual * residual;
        total_sum_squares += deviation * deviation;
    }

    if (total_sum_squares == 0.0) {
        throw std::domain_error(
            "R-squared is undefined for constant target values"
        );
    }

    return static_cast<float>(
        1.0 - residual_sum_squares / total_sum_squares
    );
}

} // namespace cppml::math
