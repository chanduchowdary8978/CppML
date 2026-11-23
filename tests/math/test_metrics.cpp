#include <cmath>

#include "cppml/math/metrics.hpp"

#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>

using cppml::Vector;
using cppml::math::Metrics;

TEST(MetricsTest, CalculatesMeanSquaredError) {
    const Vector y_true{3.0f, -0.5f, 2.0f, 7.0f};
    const Vector y_pred{2.5f, 0.0f, 2.0f, 8.0f};

    EXPECT_NEAR(
        Metrics::mean_squared_error(y_true, y_pred),
        0.375f,
        1e-6f
    );
}

TEST(MetricsTest, CalculatesRootMeanSquaredError) {
    const Vector y_true{3.0f, -0.5f, 2.0f, 7.0f};
    const Vector y_pred{2.5f, 0.0f, 2.0f, 8.0f};

    EXPECT_NEAR(
        Metrics::root_mean_squared_error(y_true, y_pred),
        std::sqrt(0.375f),
        1e-6f
    );
}

TEST(MetricsTest, CalculatesMeanAbsoluteError) {
    const Vector y_true{3.0f, -0.5f, 2.0f, 7.0f};
    const Vector y_pred{2.5f, 0.0f, 2.0f, 8.0f};

    EXPECT_NEAR(
        Metrics::mean_absolute_error(y_true, y_pred),
        0.5f,
        1e-6f
    );
}

TEST(MetricsTest, PerfectPredictionsHaveZeroErrors) {
    const Vector values{1.0f, 2.0f, 3.0f, 4.0f};

    EXPECT_FLOAT_EQ(
        Metrics::mean_squared_error(values, values),
        0.0f
    );

    EXPECT_FLOAT_EQ(
        Metrics::root_mean_squared_error(values, values),
        0.0f
    );

    EXPECT_FLOAT_EQ(
        Metrics::mean_absolute_error(values, values),
        0.0f
    );
}

TEST(MetricsTest, PerfectPredictionsHaveRSquaredOne) {
    const Vector values{1.0f, 2.0f, 3.0f, 4.0f};

    EXPECT_NEAR(
        Metrics::r_squared(values, values),
        1.0f,
        1e-6f
    );
}

TEST(MetricsTest, ProducesNegativeRSquaredForPoorPredictions) {
    const Vector y_true{1.0f, 2.0f, 3.0f};
    const Vector y_pred{10.0f, 10.0f, 10.0f};

    EXPECT_LT(
        Metrics::r_squared(y_true, y_pred),
        0.0f
    );
}

TEST(MetricsTest, RejectsConstantTargetForRSquared) {
    const Vector y_true{5.0f, 5.0f, 5.0f};
    const Vector y_pred{4.0f, 5.0f, 6.0f};

    EXPECT_THROW(
        Metrics::r_squared(y_true, y_pred),
        std::domain_error
    );
}

TEST(MetricsTest, RejectsDimensionMismatch) {
    const Vector y_true{1.0f, 2.0f, 3.0f};
    const Vector y_pred{1.0f, 2.0f};

    EXPECT_THROW(
        Metrics::mean_squared_error(y_true, y_pred),
        std::invalid_argument
    );
}

TEST(MetricsTest, RejectsEmptyVectors) {
    const Vector y_true;
    const Vector y_pred;

    EXPECT_THROW(
        Metrics::mean_squared_error(y_true, y_pred),
        std::invalid_argument
    );
}

TEST(MetricsTest, RejectsNaNInTargets) {
    const float nan = std::numeric_limits<float>::quiet_NaN();

    const Vector y_true{1.0f, nan, 3.0f};
    const Vector y_pred{1.0f, 2.0f, 3.0f};

    EXPECT_THROW(
        Metrics::mean_squared_error(y_true, y_pred),
        std::invalid_argument
    );
}

TEST(MetricsTest, RejectsInfinityInTargets) {
    const float inf = std::numeric_limits<float>::infinity();

    const Vector y_true{1.0f, inf, 3.0f};
    const Vector y_pred{1.0f, 2.0f, 3.0f};

    EXPECT_THROW(
        Metrics::mean_absolute_error(y_true, y_pred),
        std::invalid_argument
    );
}

TEST(MetricsTest, RejectsNonFinitePredictions) {
    const float nan = std::numeric_limits<float>::quiet_NaN();

    const Vector y_true{1.0f, 2.0f, 3.0f};
    const Vector y_pred{1.0f, nan, 3.0f};

    EXPECT_THROW(
        Metrics::r_squared(y_true, y_pred),
        std::invalid_argument
    );
}

