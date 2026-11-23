
#include "cppml/models/linear_regression.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace cppml::models {
namespace {

TEST(LinearRegressionConditioningTest, LargeFeatureScaleNeedsSmallLearningRate) {
    Matrix X(4, 1);
    X(0, 0) = -100.0f;
    X(1, 0) = -50.0f;
    X(2, 0) = 50.0f;
    X(3, 0) = 100.0f;

    Vector y{
        -2.0f,
        -1.0f,
        1.0f,
        2.0f
    };

    LinearRegression model(ExecutionBackend::CPU);
    model.fit(X, y, 0.0001f, 100);

    EXPECT_NEAR(model.coefficients()[0], 0.02f, 1e-5f);
    EXPECT_NEAR(model.intercept(), 0.0f, 1e-5f);
}

TEST(LinearRegressionConditioningTest, SmallFeatureScaleProducesSlowWeightConvergence) {
    Matrix X(4, 1);
    X(0, 0) = -0.02f;
    X(1, 0) = -0.01f;
    X(2, 0) = 0.01f;
    X(3, 0) = 0.02f;

    Vector y{
        -0.06f,
        -0.03f,
        0.03f,
        0.06f
    };

    LinearRegression model(ExecutionBackend::CPU);
    model.fit(X, y, 0.1f, 1000);

    // The optimal coefficient is 3.0.
    // With small feature magnitudes, the gradient is also small.
    EXPECT_GT(model.coefficients()[0], 0.0f);
    EXPECT_LT(model.coefficients()[0], 0.1f);
    EXPECT_NEAR(model.intercept(), 0.0f, 1e-5f);
}

TEST(LinearRegressionConditioningTest, ExcessiveLearningRateCanCauseOverflow) {
    Matrix X(4, 1);
    X(0, 0) = -100.0f;
    X(1, 0) = -50.0f;
    X(2, 0) = 50.0f;
    X(3, 0) = 100.0f;

    Vector y{
        -2.0f,
        -1.0f,
        1.0f,
        2.0f
    };

    LinearRegression model(ExecutionBackend::CPU);

    EXPECT_THROW(
        model.fit(X, y, 0.001f, 100),
        std::overflow_error
    );
}

TEST(LinearRegressionConditioningTest, NearCollinearFeaturesCanGiveAccuratePredictionsButUnstableCoefficients) {
    constexpr std::size_t rows = 5;
    constexpr float epsilon = 0.001f;

    const float feature_one[rows] = {
        -2.0f, -1.0f, 0.0f, 1.0f, 2.0f
    };

    const float perturbation[rows] = {
        1.0f, -2.0f, 0.0f, 2.0f, -1.0f
    };

    Matrix X(rows, 2);
    Vector y(rows);

    for (std::size_t i = 0; i < rows; ++i) {
        X(i, 0) = feature_one[i];
        X(i, 1) = feature_one[i] + epsilon * perturbation[i];

        y[i] = 2.0f * X(i, 0) + 3.0f * X(i, 1) + 1.0f;
    }

    LinearRegression model(ExecutionBackend::CPU);
    model.fit(X, y, 0.1f, 1000);

    const Vector predictions = model.predict(X);

    float maximum_absolute_error = 0.0f;

    for (std::size_t i = 0; i < rows; ++i) {
        maximum_absolute_error = std::max(
            maximum_absolute_error,
            std::abs(predictions[i] - y[i])
        );
    }

    // Predictions can be accurate even though the coefficients
    // are not individually recovered reliably.
    EXPECT_LT(maximum_absolute_error, 0.01f);

    EXPECT_GT(
        std::abs(model.coefficients()[0] - 2.0f),
        0.1f
    );

    EXPECT_GT(
        std::abs(model.coefficients()[1] - 3.0f),
        0.1f
    );
}

} // namespace
} // namespace cppml::models
