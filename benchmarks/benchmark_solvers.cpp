
#include <benchmark/benchmark.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "cppml/core/matrix.hpp"
#include "cppml/core/vector.hpp"
#include "cppml/models/least_squares.hpp"
#include "cppml/models/svd.hpp"

namespace {

struct Dataset {
    cppml::Matrix X;
    cppml::Vector y;
    std::vector<double> true_coefficients;
    double true_intercept = 0.75;
};

struct Accuracy {
    double mse = 0.0;
    double max_prediction_error = 0.0;
    double coefficient_rmse = 0.0;
    double max_coefficient_error = 0.0;
    double intercept_error = 0.0;
};

Dataset make_dataset(std::size_t rows, std::size_t cols) {
    Dataset data{
        cppml::Matrix(rows, cols),
        cppml::Vector(rows),
        std::vector<double>(cols)
    };

    for (std::size_t j = 0; j < cols; ++j) {
        data.true_coefficients[j] = 0.05 * static_cast<double>(j + 1);
    }

    for (std::size_t i = 0; i < rows; ++i) {
        double target = data.true_intercept;

        for (std::size_t j = 0; j < cols; ++j) {
            const double row = static_cast<double>(i + 1);
            const double col = static_cast<double>(j + 1);

            const double value =
                std::sin(row * col * 0.013) +
                std::cos(row * (col + 1.0) * 0.007) +
                (i == j ? 0.1 : 0.0);

            data.X(i, j) = static_cast<float>(value);
            target += value * data.true_coefficients[j];
        }

        data.y[i] = static_cast<float>(target);
    }

    return data;
}

Accuracy evaluate(
    const Dataset& data,
    const cppml::Vector& coefficients,
    float intercept
) {
    if (coefficients.size() != data.X.cols()) {
        throw std::invalid_argument("Coefficient count mismatch");
    }

    Accuracy result;
    double squared_coefficient_error = 0.0;

    for (std::size_t j = 0; j < coefficients.size(); ++j) {
        const double error =
            static_cast<double>(coefficients[j]) -
            data.true_coefficients[j];

        squared_coefficient_error += error * error;
        result.max_coefficient_error =
            std::max(result.max_coefficient_error, std::abs(error));
    }

    result.coefficient_rmse = std::sqrt(
        squared_coefficient_error /
        static_cast<double>(coefficients.size())
    );

    result.intercept_error = std::abs(
        static_cast<double>(intercept) - data.true_intercept
    );

    double squared_prediction_error = 0.0;

    for (std::size_t i = 0; i < data.X.rows(); ++i) {
        double prediction = static_cast<double>(intercept);

        for (std::size_t j = 0; j < data.X.cols(); ++j) {
            prediction +=
                static_cast<double>(data.X(i, j)) *
                static_cast<double>(coefficients[j]);
        }

        const double error =
            prediction - static_cast<double>(data.y[i]);

        squared_prediction_error += error * error;
        result.max_prediction_error =
            std::max(result.max_prediction_error, std::abs(error));
    }

    result.mse =
        squared_prediction_error /
        static_cast<double>(data.X.rows());

    return result;
}

void add_accuracy_counters(
    benchmark::State& state,
    const Accuracy& accuracy
) {
    state.counters["prediction_mse"] = accuracy.mse;
    state.counters["prediction_max_abs_error"] =
        accuracy.max_prediction_error;
    state.counters["coefficient_rmse"] = accuracy.coefficient_rmse;
    state.counters["coefficient_max_abs_error"] =
        accuracy.max_coefficient_error;
    state.counters["intercept_abs_error"] = accuracy.intercept_error;
}

void BM_LeastSquares_QR(benchmark::State& state) {
    const auto rows = static_cast<std::size_t>(state.range(0));
    const auto cols = static_cast<std::size_t>(state.range(1));
    const auto data = make_dataset(rows, cols);

    const auto validation_result =
        cppml::models::LeastSquares::solve_qr(data.X, data.y);

    const Accuracy accuracy = evaluate(
        data,
        validation_result.coefficients,
        validation_result.intercept
    );

    for (auto _ : state) {
        const auto result =
            cppml::models::LeastSquares::solve_qr(data.X, data.y);

        benchmark::DoNotOptimize(result.coefficients.data());
        benchmark::DoNotOptimize(result.intercept);
    }

    add_accuracy_counters(state, accuracy);

    state.SetItemsProcessed(
        state.iterations() * static_cast<int64_t>(rows * cols)
    );
    state.SetLabel("CPU QR");
}

void BM_LeastSquares_SVD(benchmark::State& state) {
    const auto rows = static_cast<std::size_t>(state.range(0));
    const auto cols = static_cast<std::size_t>(state.range(1));
    const auto data = make_dataset(rows, cols);

    const auto validation_result =
        cppml::models::SVD::solve_pseudoinverse(data.X, data.y);

    const Accuracy accuracy = evaluate(
        data,
        validation_result.coefficients,
        validation_result.intercept
    );

    for (auto _ : state) {
        const auto result =
            cppml::models::SVD::solve_pseudoinverse(data.X, data.y);

        benchmark::DoNotOptimize(result.coefficients.data());
        benchmark::DoNotOptimize(result.intercept);
    }

    add_accuracy_counters(state, accuracy);

    state.SetItemsProcessed(
        state.iterations() * static_cast<int64_t>(rows * cols)
    );
    state.SetLabel("CPU Jacobi SVD");
}

} // namespace

BENCHMARK(BM_LeastSquares_QR)
    ->Args({256, 4})
    ->Args({512, 8})
    ->Args({1024, 16})
    ->Args({2048, 16})
    ->UseRealTime();

BENCHMARK(BM_LeastSquares_SVD)
    ->Args({256, 4})
    ->Args({512, 8})
    ->Args({1024, 16})
    ->Args({2048, 16})
    ->UseRealTime();

BENCHMARK_MAIN();
