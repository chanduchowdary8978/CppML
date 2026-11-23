
#pragma once

#include "cppml/core/vector.hpp"

namespace cppml::math {

class Metrics {
public:
    // Mean Squared Error
    [[nodiscard]] static float mean_squared_error(
        const Vector& y_true,
        const Vector& y_pred
    );

    // Root Mean Squared Error
    [[nodiscard]] static float root_mean_squared_error(
        const Vector& y_true,
        const Vector& y_pred
    );

    // Mean Absolute Error
    [[nodiscard]] static float mean_absolute_error(
        const Vector& y_true,
        const Vector& y_pred
    );

    // Coefficient of Determination
    [[nodiscard]] static float r_squared(
        const Vector& y_true,
        const Vector& y_pred
    );

private:
    static void validate_inputs(
        const Vector& y_true,
        const Vector& y_pred
    );
};

} // namespace cppml::math
