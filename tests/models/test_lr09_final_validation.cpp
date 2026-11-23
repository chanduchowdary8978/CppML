
#include "cppml/models/least_squares.hpp"
#include "cppml/models/svd.hpp"
#include "cppml/models/linear_regression.hpp"
#include "cppml/math/metrics.hpp"

#ifdef CPPML_HAS_CUDA
#include "cppml/models/cuda_linear_regression.hpp"
#endif

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

using cppml::Matrix;
using cppml::Vector;
using cppml::models::LeastSquares;
using cppml::models::SVD;
using cppml::models::LinearRegression;

Vector predict_solver(
    const Matrix& X,
    const Vector& coefficients,
    float intercept
) {
    Vector predictions(X.rows());

    for (std::size_t i = 0; i < X.rows(); ++i) {
        double value = static_cast<double>(intercept);

        for (std::size_t j = 0; j < X.cols(); ++j) {
            value +=
                static_cast<double>(X(i, j)) *
                static_cast<double>(coefficients[j]);
        }

        predictions[i] = static_cast<float>(value);
    }

    return predictions;
}

double calculate_max_error(
    const Vector& expected,
    const Vector& actual
) {
    if (expected.size() != actual.size()) {
        throw std::invalid_argument("Vector sizes do not match");
    }

    double maximum_error = 0.0;

    for (std::size_t i = 0; i < expected.size(); ++i) {
        maximum_error = std::max(
            maximum_error,
            std::abs(
                static_cast<double>(expected[i]) -
                static_cast<double>(actual[i])
            )
        );
    }

    return maximum_error;
}

TEST(LR09RandomizedTest, QRAndSVDRecoverWellConditionedRandomDatasets) {
    std::mt19937 generator(20261123);
    std::uniform_real_distribution<float> distribution(-2.0f, 2.0f);

    constexpr std::size_t trials = 20;
    constexpr std::size_t rows = 30;
    constexpr std::size_t features = 3;

    for (std::size_t trial = 0; trial < trials; ++trial) {
        Matrix X(rows, features);
        Vector y(rows);

        const float true_weights[features] = {
            1.25f, -2.0f, 0.75f
        };
        constexpr float true_bias = 3.5f;

        for (std::size_t i = 0; i < rows; ++i) {
            double target = true_bias;

            for (std::size_t j = 0; j < features; ++j) {
                X(i, j) = distribution(generator);
                target +=
                    static_cast<double>(X(i, j)) *
                    static_cast<double>(true_weights[j]);
            }

            y[i] = static_cast<float>(target);
        }

        const auto qr = LeastSquares::solve_qr(X, y);
        const auto svd = SVD::solve_pseudoinverse(X, y);

        const Vector qr_predictions = predict_solver(
            X, qr.coefficients, qr.intercept
        );

        const Vector svd_predictions = predict_solver(
            X, svd.coefficients, svd.intercept
        );

        EXPECT_LT(calculate_max_error(y, qr_predictions), 2e-3)
            << "QR failed on randomized trial " << trial;

        EXPECT_LT(calculate_max_error(y, svd_predictions), 2e-3)
            << "SVD failed on randomized trial " << trial;

        EXPECT_LT(
            calculate_max_error(qr_predictions, svd_predictions),
            2e-3
        ) << "QR/SVD disagreement on randomized trial " << trial;
    }
}

TEST(LR09RandomizedTest, LinearRegressionProducesFinitePredictions) {
    std::mt19937 generator(20261124);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

    constexpr std::size_t rows = 40;
    constexpr std::size_t features = 2;

    Matrix X(rows, features);
    Vector y(rows);

    for (std::size_t i = 0; i < rows; ++i) {
        X(i, 0) = distribution(generator);
        X(i, 1) = distribution(generator);

        y[i] =
            2.0f * X(i, 0) -
            1.5f * X(i, 1) +
            0.75f;
    }

    LinearRegression model(cppml::models::ExecutionBackend::CPU);
    model.fit(X, y, 0.05f, 2000);

    const Vector predictions = model.predict(X);

    ASSERT_EQ(predictions.size(), rows);

    for (std::size_t i = 0; i < predictions.size(); ++i) {
        EXPECT_TRUE(std::isfinite(predictions[i]));
    }

    EXPECT_LT(
        cppml::math::Metrics::mean_squared_error(y, predictions),
        1e-3f
    );
}

TEST(LR09EdgeCaseTest, QRRejectsConstantFeatureAlongsideIntercept) {
    Matrix X(6, 1, 5.0f);
    Vector y{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    EXPECT_THROW(
        static_cast<void>(LeastSquares::solve_qr(X, y)),
        std::domain_error
    );
}

TEST(LR09EdgeCaseTest, SVDHandlesConstantFeatureWithMinimumNormSolution) {
    Matrix X(6, 1, 5.0f);
    Vector y(6, 11.0f);

    const auto result = SVD::solve_pseudoinverse(X, y);

    ASSERT_EQ(result.coefficients.size(), 1);

    const Vector predictions = predict_solver(
        X, result.coefficients, result.intercept
    );

    EXPECT_LT(calculate_max_error(y, predictions), 1e-3);
}

TEST(LR09EdgeCaseTest, QRAndSVDRejectNonFiniteInputs) {
    Matrix X(4, 2);
    Vector y{1.0f, 2.0f, 3.0f, 4.0f};

    X(0, 0) = 1.0f;
    X(1, 0) = 2.0f;
    X(2, 0) = std::numeric_limits<float>::quiet_NaN();
    X(3, 0) = 4.0f;

    EXPECT_THROW(
        static_cast<void>(LeastSquares::solve_qr(X, y)),
        std::invalid_argument
    );

    EXPECT_THROW(
        static_cast<void>(SVD::solve_pseudoinverse(X, y)),
        std::invalid_argument
    );
}

#ifdef CPPML_HAS_CUDA

TEST(LR09CudaConsistencyTest, LargeFeatureScaleMatchesCPU) {
    Matrix X(4, 1);
    X(0, 0) = -100.0f;
    X(1, 0) = -50.0f;
    X(2, 0) = 50.0f;
    X(3, 0) = 100.0f;

    Vector y{-2.0f, -1.0f, 1.0f, 2.0f};

    LinearRegression cpu(cppml::models::ExecutionBackend::CPU);
    cppml::models::CudaLinearRegression gpu;

    cpu.fit(X, y, 0.0001f, 100);
    gpu.fit(X, y, 0.0001f, 100);

    EXPECT_NEAR(
        cpu.coefficients()[0],
        gpu.coefficients()[0],
        1e-6f
    );

    EXPECT_NEAR(cpu.intercept(), gpu.intercept(), 1e-6f);

    const Vector cpu_predictions = cpu.predict(X);
    const Vector gpu_predictions = gpu.predict(X);

    EXPECT_LT(
        calculate_max_error(cpu_predictions, gpu_predictions),
        1e-5
    );
}

TEST(LR09CudaConsistencyTest, SmallFeatureScaleMatchesCPU) {
    Matrix X(4, 1);
    X(0, 0) = -0.02f;
    X(1, 0) = -0.01f;
    X(2, 0) = 0.01f;
    X(3, 0) = 0.02f;

    Vector y{-0.06f, -0.03f, 0.03f, 0.06f};

    LinearRegression cpu(cppml::models::ExecutionBackend::CPU);
    cppml::models::CudaLinearRegression gpu;

    cpu.fit(X, y, 0.1f, 1000);
    gpu.fit(X, y, 0.1f, 1000);

    EXPECT_NEAR(
        cpu.coefficients()[0],
        gpu.coefficients()[0],
        1e-5f
    );

    EXPECT_NEAR(cpu.intercept(), gpu.intercept(), 1e-5f);

    EXPECT_LT(
        calculate_max_error(cpu.predict(X), gpu.predict(X)),
        1e-5
    );
}

TEST(LR09CudaConsistencyTest, NearCollinearFeaturesMatchCPUPredictions) {
    constexpr std::size_t rows = 5;
    constexpr float epsilon = 0.001f;

    const float first[rows] = {
        -2.0f, -1.0f, 0.0f, 1.0f, 2.0f
    };

    const float perturbation[rows] = {
        1.0f, -2.0f, 0.0f, 2.0f, -1.0f
    };

    Matrix X(rows, 2);
    Vector y(rows);

    for (std::size_t i = 0; i < rows; ++i) {
        X(i, 0) = first[i];
        X(i, 1) = first[i] + epsilon * perturbation[i];
        y[i] = 2.0f * X(i, 0) + 3.0f * X(i, 1) + 1.0f;
    }

    LinearRegression cpu(cppml::models::ExecutionBackend::CPU);
    cppml::models::CudaLinearRegression gpu;

    cpu.fit(X, y, 0.1f, 1000);
    gpu.fit(X, y, 0.1f, 1000);

    EXPECT_LT(
        calculate_max_error(cpu.predict(X), gpu.predict(X)),
        1e-3
    );
}

TEST(LR09CudaConsistencyTest, ExcessiveLearningRateFailsOnBothBackends) {
    Matrix X(4, 1);
    X(0, 0) = -100.0f;
    X(1, 0) = -50.0f;
    X(2, 0) = 50.0f;
    X(3, 0) = 100.0f;

    Vector y{-2.0f, -1.0f, 1.0f, 2.0f};

    LinearRegression cpu(cppml::models::ExecutionBackend::CPU);
    cppml::models::CudaLinearRegression gpu;

    EXPECT_THROW(cpu.fit(X, y, 0.001f, 100), std::overflow_error);
    EXPECT_THROW(gpu.fit(X, y, 0.001f, 100), std::overflow_error);
}

#endif

} // namespace
