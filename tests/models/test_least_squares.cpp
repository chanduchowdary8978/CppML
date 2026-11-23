
#include "cppml/models/least_squares.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <limits>
#include <stdexcept>

using cppml::Matrix;
using cppml::Vector;
using cppml::models::LeastSquares;
using cppml::models::LeastSquaresResult;

TEST(LeastSquaresTest, SolvesSimpleLinearRelationship) {
    Matrix X(5, 1);
    Vector y(5);

    for (std::size_t i = 0; i < 5; ++i) {
        X(i, 0) = static_cast<float>(i + 1);
        y[i] = 3.0f * X(i, 0) + 2.0f;
    }

    const LeastSquaresResult result = LeastSquares::solve_qr(X, y);

    ASSERT_EQ(result.coefficients.size(), 1);
    EXPECT_NEAR(result.coefficients[0], 3.0f, 1e-5f);
    EXPECT_NEAR(result.intercept, 2.0f, 1e-5f);
}

TEST(LeastSquaresTest, SolvesMultipleLinearRelationship) {
    Matrix X(6, 2);
    Vector y(6);

    for (std::size_t i = 0; i < 6; ++i) {
        const float x1 = static_cast<float>(i + 1);
        const float x2 = static_cast<float>((i + 1) * (i + 1));

        X(i, 0) = x1;
        X(i, 1) = x2;
        y[i] = 2.0f * x1 - 0.5f * x2 + 4.0f;
    }

    const LeastSquaresResult result = LeastSquares::solve_qr(X, y);

    ASSERT_EQ(result.coefficients.size(), 2);
    EXPECT_NEAR(result.coefficients[0], 2.0f, 1e-5f);
    EXPECT_NEAR(result.coefficients[1], -0.5f, 1e-5f);
    EXPECT_NEAR(result.intercept, 4.0f, 1e-5f);
}

TEST(LeastSquaresTest, SolvesOverdeterminedNoisyDataset) {
    Matrix X(5, 1);
    Vector y{5.1f, 7.0f, 9.2f, 10.9f, 13.1f};

    for (std::size_t i = 0; i < 5; ++i) {
        X(i, 0) = static_cast<float>(i + 1);
    }

    const LeastSquaresResult result = LeastSquares::solve_qr(X, y);

    ASSERT_EQ(result.coefficients.size(), 1);
    EXPECT_NEAR(result.coefficients[0], 1.99f, 0.03f);
    EXPECT_NEAR(result.intercept, 3.12f, 0.08f);
}

TEST(LeastSquaresTest, SolvesFeaturesWithDifferentScales) {
    Matrix X(6, 2);
    Vector y(6);

    for (std::size_t i = 0; i < 6; ++i) {
        const float x1 = static_cast<float>(i + 1) * 0.001f;
        const float x2 =
            static_cast<float>((i + 1) * (i + 1)) * 1000.0f;

        X(i, 0) = x1;
        X(i, 1) = x2;

        // The small-scale feature is independent but has zero
        // target coefficient. This avoids losing its contribution
        // to float target rounding.
        y[i] = -0.5f * x2 + 4.0f;
    }

    const LeastSquaresResult result = LeastSquares::solve_qr(X, y);

    ASSERT_EQ(result.coefficients.size(), 2);
    EXPECT_NEAR(result.coefficients[0], 0.0f, 1e-3f);
    EXPECT_NEAR(result.coefficients[1], -0.5f, 1e-5f);
    EXPECT_NEAR(result.intercept, 4.0f, 1e-3f);
}

TEST(LeastSquaresTest, RejectsTargetSizeMismatch) {
    Matrix X(3, 1);
    Vector y(2);

    EXPECT_THROW(
        static_cast<void>(LeastSquares::solve_qr(X, y)),
        std::invalid_argument
    );
}

TEST(LeastSquaresTest, RejectsEmptyMatrix) {
    Matrix X;
    Vector y;

    EXPECT_THROW(
        static_cast<void>(LeastSquares::solve_qr(X, y)),
        std::invalid_argument
    );
}

TEST(LeastSquaresTest, RejectsUnderdeterminedSystem) {
    Matrix X(2, 2);
    Vector y{1.0f, 2.0f};

    EXPECT_THROW(
        static_cast<void>(LeastSquares::solve_qr(X, y)),
        std::invalid_argument
    );
}

TEST(LeastSquaresTest, RejectsRankDeficientMatrix) {
    Matrix X(4, 2);
    Vector y(4);

    for (std::size_t i = 0; i < 4; ++i) {
        const float value = static_cast<float>(i + 1);

        X(i, 0) = value;
        X(i, 1) = 2.0f * value;
        y[i] = 3.0f * value + 1.0f;
    }

    EXPECT_THROW(
        static_cast<void>(LeastSquares::solve_qr(X, y)),
        std::domain_error
    );
}

TEST(LeastSquaresTest, RejectsNonFiniteFeatures) {
    Matrix X(3, 1);
    Vector y{1.0f, 2.0f, 3.0f};

    X(0, 0) = 1.0f;
    X(1, 0) = std::numeric_limits<float>::quiet_NaN();
    X(2, 0) = 3.0f;

    EXPECT_THROW(
        static_cast<void>(LeastSquares::solve_qr(X, y)),
        std::invalid_argument
    );
}

TEST(LeastSquaresTest, RejectsNonFiniteTargets) {
    Matrix X(3, 1);
    Vector y{
        1.0f,
        2.0f,
        std::numeric_limits<float>::infinity()
    };

    X(0, 0) = 1.0f;
    X(1, 0) = 2.0f;
    X(2, 0) = 3.0f;

    EXPECT_THROW(
        static_cast<void>(LeastSquares::solve_qr(X, y)),
        std::invalid_argument
    );
}
