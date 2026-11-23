#include "cppml/math/metrics.hpp"
#include "cppml/math/operations.hpp"

#include <gtest/gtest.h>
#include <stdexcept>
#include <cmath>
using cppml::Matrix;
using cppml::Vector;

// ---------- MATRIX OPERATIONS ----------

TEST(MathTest, MatrixVectorMultiplication) {
    Matrix X(3, 2);

    X(0, 0) = 1; X(0, 1) = 2;
    X(1, 0) = 3; X(1, 1) = 4;
    X(2, 0) = 5; X(2, 1) = 6;

    Vector w{2, 3};

    Vector result = cppml::math::matvec(X, w);

    ASSERT_EQ(result.size(), 3);

    EXPECT_FLOAT_EQ(result[0], 8);
    EXPECT_FLOAT_EQ(result[1], 18);
    EXPECT_FLOAT_EQ(result[2], 28);
}

TEST(MathTest, TransposeMatrixVectorMultiplication) {
    Matrix X(3, 2);

    X(0, 0) = 1; X(0, 1) = 2;
    X(1, 0) = 3; X(1, 1) = 4;
    X(2, 0) = 5; X(2, 1) = 6;

    Vector r{1, 2, 3};

    Vector result = cppml::math::transpose_matvec(X, r);

    ASSERT_EQ(result.size(), 2);

    EXPECT_FLOAT_EQ(result[0], 22);
    EXPECT_FLOAT_EQ(result[1], 28);
}

TEST(MathTest, MatvecDimensionMismatch) {
    Matrix X(3, 2);
    Vector w(3);

    EXPECT_THROW(
        cppml::math::matvec(X, w),
        std::invalid_argument
    );
}

TEST(MathTest, TransposeMatvecDimensionMismatch) {
    Matrix X(3, 2);
    Vector r(2);

    EXPECT_THROW(
        cppml::math::transpose_matvec(X, r),
        std::invalid_argument
    );
}

// ---------- VECTOR OPERATIONS ----------

TEST(MathTest, VectorSubtraction) {
    Vector a{10, 20, 30};
    Vector b{1, 2, 3};

    Vector result = cppml::math::subtract(a, b);

    EXPECT_FLOAT_EQ(result[0], 9);
    EXPECT_FLOAT_EQ(result[1], 18);
    EXPECT_FLOAT_EQ(result[2], 27);
}

TEST(MathTest, SubtractionDimensionMismatch) {
    Vector a{1, 2};
    Vector b{1, 2, 3};

    EXPECT_THROW(
        cppml::math::subtract(a, b),
        std::invalid_argument
    );
}

TEST(MathTest, DotProduct) {
    Vector a{1, 2, 3};
    Vector b{4, 5, 6};

    EXPECT_FLOAT_EQ(cppml::math::dot(a, b), 32);
}

TEST(MathTest, DotProductDimensionMismatch) {
    Vector a{1, 2};
    Vector b{1, 2, 3};

    EXPECT_THROW(
        cppml::math::dot(a, b),
        std::invalid_argument
    );
}

TEST(MathTest, Sum) {
    Vector v{1, 2, 3, 4};

    EXPECT_FLOAT_EQ(cppml::math::sum(v), 10);
}

TEST(MathTest, Mean) {
    Vector v{2, 4, 6, 8};

    EXPECT_FLOAT_EQ(cppml::math::mean(v), 5);
}

TEST(MathTest, MeanOfEmptyVector) {
    Vector v;

    EXPECT_THROW(
        cppml::math::mean(v),
        std::invalid_argument
    );
}

TEST(MathTest, AddScalar) {
    Vector v{1, 2, 3};

    Vector result = cppml::math::add_scalar(v, 5);

    EXPECT_FLOAT_EQ(result[0], 6);
    EXPECT_FLOAT_EQ(result[1], 7);
    EXPECT_FLOAT_EQ(result[2], 8);
}

// ---------- METRICS ----------

TEST(MetricsTest, MeanSquaredError) {
    Vector actual{3, -0.5f, 2, 7};
    Vector predicted{2.5f, 0.0f, 2, 8};

    EXPECT_FLOAT_EQ(
        cppml::math::Metrics::mean_squared_error(actual, predicted),
        0.375f
    );
}

TEST(MetricsTest, PerfectPrediction) {
    Vector actual{1, 2, 3};
    Vector predicted{1, 2, 3};

    EXPECT_FLOAT_EQ(
        cppml::math::Metrics::mean_squared_error(actual, predicted),
        0.0f
    );
}

TEST(MetricsTest, DimensionMismatch) {
    Vector actual{1, 2, 3};
    Vector predicted{1, 2};

    EXPECT_THROW(
        cppml::math::Metrics::mean_squared_error(actual, predicted),
        std::invalid_argument
    );
}

TEST(MetricsTest, EmptyVectors) {
    Vector actual;
    Vector predicted;

    EXPECT_THROW(
        cppml::math::Metrics::mean_squared_error(actual, predicted),
        std::invalid_argument
    );
}

// ---------- METRICS V1 ----------

TEST(MetricsTest, RootMeanSquaredError) {
    Vector actual{3, -0.5f, 2, 7};
    Vector predicted{2.5f, 0.0f, 2, 8};

    EXPECT_NEAR(
        cppml::math::Metrics::root_mean_squared_error(actual, predicted),
        std::sqrt(0.375f),
        1e-6f
    );
}

TEST(MetricsTest, MeanAbsoluteError) {
    Vector actual{3, -0.5f, 2, 7};
    Vector predicted{2.5f, 0.0f, 2, 8};

    EXPECT_FLOAT_EQ(
        cppml::math::Metrics::mean_absolute_error(actual, predicted),
        0.5f
    );
}

TEST(MetricsTest, RSquared) {
    Vector actual{1, 2, 3};
    Vector predicted{1, 2, 2};

    EXPECT_FLOAT_EQ(
        cppml::math::Metrics::r_squared(actual, predicted),
        0.5f
    );
}

TEST(MetricsTest, RSquaredPerfectPrediction) {
    Vector actual{1, 2, 3};
    Vector predicted{1, 2, 3};

    EXPECT_FLOAT_EQ(
        cppml::math::Metrics::r_squared(actual, predicted),
        1.0f
    );
}

TEST(MetricsTest, RSquaredCanBeNegative) {
    Vector actual{1, 2, 3};
    Vector predicted{3, 2, 1};

    EXPECT_FLOAT_EQ(
        cppml::math::Metrics::r_squared(actual, predicted),
        -3.0f
    );
}

TEST(MetricsTest, RSquaredConstantTarget) {
    Vector actual{5, 5, 5};
    Vector predicted{5, 5, 5};

    EXPECT_THROW(
        cppml::math::Metrics::r_squared(actual, predicted),
        std::domain_error
    );
}


