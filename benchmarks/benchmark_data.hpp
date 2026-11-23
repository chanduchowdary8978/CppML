
#pragma once

#include "cppml/core/matrix.hpp"
#include "cppml/core/vector.hpp"

#include <cmath>
#include <cstddef>
#include <utility>

namespace cppml::benchmarks {

struct Dataset {
    Matrix X;
    Vector y;
};

inline Dataset make_dataset(std::size_t rows, std::size_t cols) {
    Matrix X(rows, cols);
    Vector y(rows);

    for (std::size_t i = 0; i < rows; ++i) {
        double target = 0.75;

        for (std::size_t j = 0; j < cols; ++j) {
            const double a =
                static_cast<double>((i + 1) * (j + 1));

            const double b =
                static_cast<double>((i + 1) * (j + 2));

            double value =
                std::sin(a * 0.013) +
                std::cos(b * 0.007);

            if (i == j) {
                value += 0.1;
            }

            X(i, j) = static_cast<float>(value);

            const double coefficient =
                0.05 * static_cast<double>(j + 1);

            target += value * coefficient;
        }

        y[i] = static_cast<float>(target);
    }

    return {std::move(X), std::move(y)};
}

inline Vector make_vector(std::size_t size, float offset = 0.0f) {
    Vector result(size);

    for (std::size_t i = 0; i < size; ++i) {
        result[i] =
            static_cast<float>(
                std::sin(static_cast<double>(i + 1) * 0.01) + offset
            );
    }

    return result;
}

} // namespace cppml::benchmarks
