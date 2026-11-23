
#include "cppml/models/least_squares.hpp"
#include "cppml/models/svd.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

using cppml::Matrix;
using cppml::Vector;
using cppml::models::LeastSquares;
using cppml::models::SVD;

namespace {

struct Dataset {
    Matrix X;
    Vector y;
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

Dataset make_benchmark_dataset(std::size_t rows, std::size_t cols) {
    Dataset data{
        Matrix(rows, cols),
        Vector(rows),
        std::vector<double>(cols)
    };

    for (std::size_t j = 0; j < cols; ++j) {
        data.true_coefficients[j] =
            0.05 * static_cast<double>(j + 1);
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

Vector predict(
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

double residual_sum_of_squares(
    const Vector& actual,
    const Vector& predicted
) {
    double rss = 0.0;

    for (std::size_t i = 0; i < actual.size(); ++i) {
        const double residual =
            static_cast<double>(actual[i]) -
            static_cast<double>(predicted[i]);

        rss += residual * residual;
    }

    return rss;
}

Accuracy evaluate(
    const Dataset& data,
    const Vector& coefficients,
    float intercept
) {
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

    const Vector predictions = predict(
        data.X,
        coefficients,
        intercept
    );

    double squared_prediction_error = 0.0;

    for (std::size_t i = 0; i < data.y.size(); ++i) {
        const double error =
            static_cast<double>(predictions[i]) -
            static_cast<double>(data.y[i]);

        squared_prediction_error += error * error;

        result.max_prediction_error =
            std::max(result.max_prediction_error, std::abs(error));
    }

    result.mse =
        squared_prediction_error /
        static_cast<double>(data.y.size());

    return result;
}

void expect_benchmark_accuracy(const Accuracy& accuracy) {
    EXPECT_LT(accuracy.mse, 1e-12);
    EXPECT_LT(accuracy.max_prediction_error, 1e-5);
    EXPECT_LT(accuracy.coefficient_rmse, 1e-6);
    EXPECT_LT(accuracy.max_coefficient_error, 1e-6);
    EXPECT_LT(accuracy.intercept_error, 1e-6);
}

} // namespace

TEST(SolverComparisonTest, QRAndSVDProduceSimilarPredictions) {
    Matrix X(8, 2);
    Vector y(8);

    for (std::size_t i = 0; i < 8; ++i) {
        const float x1 = static_cast<float>(i + 1);
        const float x2 = static_cast<float>((i + 1) * (i + 1));

        X(i, 0) = x1;
        X(i, 1) = x2;

        y[i] = 2.0f * x1 - 0.5f * x2 + 4.0f;
    }

    const auto qr_result = LeastSquares::solve_qr(X, y);
    const auto svd_result = SVD::solve_pseudoinverse(X, y);

    const Vector qr_predictions = predict(
        X,
        qr_result.coefficients,
        qr_result.intercept
    );

    const Vector svd_predictions = predict(
        X,
        svd_result.coefficients,
        svd_result.intercept
    );

    for (std::size_t i = 0; i < y.size(); ++i) {
        EXPECT_NEAR(
            qr_predictions[i],
            svd_predictions[i],
            1e-3f
        );
    }
}

TEST(SolverComparisonTest, QRAndSVDProduceSimilarResidualNorms) {
    Matrix X(8, 2);
    Vector y(8);

    const float noise[] = {
        0.1f, -0.2f, 0.15f, -0.1f,
        0.05f, 0.2f, -0.15f, 0.1f
    };

    for (std::size_t i = 0; i < 8; ++i) {
        const float x1 = static_cast<float>(i + 1);
        const float x2 = static_cast<float>((i + 1) * (i + 1));

        X(i, 0) = x1;
        X(i, 1) = x2;

        y[i] = 2.0f * x1 - 0.5f * x2 + 4.0f + noise[i];
    }

    const auto qr_result = LeastSquares::solve_qr(X, y);
    const auto svd_result = SVD::solve_pseudoinverse(X, y);

    const Vector qr_predictions = predict(
        X,
        qr_result.coefficients,
        qr_result.intercept
    );

    const Vector svd_predictions = predict(
        X,
        svd_result.coefficients,
        svd_result.intercept
    );

    const double qr_rss = residual_sum_of_squares(y, qr_predictions);
    const double svd_rss = residual_sum_of_squares(y, svd_predictions);

    EXPECT_NEAR(qr_rss, svd_rss, 1e-5);
}

TEST(SolverComparisonTest, SVDHandlesRankDeficiencyWhereQRRejects) {
    Matrix X(6, 2);
    Vector y(6);

    for (std::size_t i = 0; i < 6; ++i) {
        const float value = static_cast<float>(i + 1);

        X(i, 0) = value;
        X(i, 1) = 2.0f * value;

        y[i] = 3.0f * value + 1.0f;
    }

    EXPECT_THROW(
        static_cast<void>(LeastSquares::solve_qr(X, y)),
        std::domain_error
    );

    const auto svd_result = SVD::solve_pseudoinverse(X, y);

    const Vector predictions = predict(
        X,
        svd_result.coefficients,
        svd_result.intercept
    );

    for (std::size_t i = 0; i < y.size(); ++i) {
        EXPECT_NEAR(predictions[i], y[i], 1e-3f);
    }
}

TEST(SolverComparisonTest, QRAndSVDHandleDifferentFeatureScales) {
    Matrix X(8, 2);
    Vector y(8);

    for (std::size_t i = 0; i < 8; ++i) {
        const float x1 = static_cast<float>(i + 1) * 0.001f;
        const float x2 =
            static_cast<float>((i + 1) * (i + 1)) * 1000.0f;

        X(i, 0) = x1;
        X(i, 1) = x2;

        y[i] = -0.5f * x2 + 4.0f;
    }

    const auto qr_result = LeastSquares::solve_qr(X, y);
    const auto svd_result = SVD::solve_pseudoinverse(X, y);

    const Vector qr_predictions = predict(
        X,
        qr_result.coefficients,
        qr_result.intercept
    );

    const Vector svd_predictions = predict(
        X,
        svd_result.coefficients,
        svd_result.intercept
    );

    for (std::size_t i = 0; i < y.size(); ++i) {
        EXPECT_NEAR(
            qr_predictions[i],
            svd_predictions[i],
            1e-2f
        );
    }
}

TEST(SolverAccuracyTest, QRRecoversBenchmarkDatasets) {
    const std::size_t configurations[][2] = {
        {256, 4},
        {512, 8},
        {1024, 16},
        {2048, 16}
    };

    for (const auto& configuration : configurations) {
        SCOPED_TRACE(
            "QR dataset: " +
            std::to_string(configuration[0]) + "x" +
            std::to_string(configuration[1])
        );

        const auto data = make_benchmark_dataset(
            configuration[0],
            configuration[1]
        );

        const auto result = LeastSquares::solve_qr(data.X, data.y);

        const Accuracy accuracy = evaluate(
            data,
            result.coefficients,
            result.intercept
        );

        expect_benchmark_accuracy(accuracy);
    }
}

TEST(SolverAccuracyTest, SVDRecoversBenchmarkDatasets) {
    const std::size_t configurations[][2] = {
        {256, 4},
        {512, 8},
        {1024, 16},
        {2048, 16}
    };

    for (const auto& configuration : configurations) {
        SCOPED_TRACE(
            "SVD dataset: " +
            std::to_string(configuration[0]) + "x" +
            std::to_string(configuration[1])
        );

        const auto data = make_benchmark_dataset(
            configuration[0],
            configuration[1]
        );

        const auto result = SVD::solve_pseudoinverse(data.X, data.y);

        const Accuracy accuracy = evaluate(
            data,
            result.coefficients,
            result.intercept
        );

        expect_benchmark_accuracy(accuracy);
    }
}

TEST(SolverAccuracyTest, QRAndSVDProduceConsistentBenchmarkPredictions) {
    const std::size_t configurations[][2] = {
        {256, 4},
        {512, 8},
        {1024, 16},
        {2048, 16}
    };

    for (const auto& configuration : configurations) {
        SCOPED_TRACE(
            "Agreement dataset: " +
            std::to_string(configuration[0]) + "x" +
            std::to_string(configuration[1])
        );

        const auto data = make_benchmark_dataset(
            configuration[0],
            configuration[1]
        );

        const auto qr_result = LeastSquares::solve_qr(data.X, data.y);
        const auto svd_result = SVD::solve_pseudoinverse(data.X, data.y);

        const Vector qr_predictions = predict(
            data.X,
            qr_result.coefficients,
            qr_result.intercept
        );

        const Vector svd_predictions = predict(
            data.X,
            svd_result.coefficients,
            svd_result.intercept
        );

        for (std::size_t i = 0; i < data.y.size(); ++i) {
            EXPECT_NEAR(
                qr_predictions[i],
                svd_predictions[i],
                1e-5f
            );
        }
    }
}
