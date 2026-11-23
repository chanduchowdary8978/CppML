
#include "cppml/models/svd.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <limits>
#include <stdexcept>

using cppml::Matrix;
using cppml::Vector;
using cppml::models::SVD;

TEST(SVDTest, SolvesSimpleLinearRelationship) {
    Matrix X(5, 1);
    Vector y(5);

    for (std::size_t i = 0; i < 5; ++i) {
        X(i, 0) = static_cast<float>(i + 1);
        y[i] = 3.0f * X(i, 0) + 2.0f;
    }

    const auto result = SVD::solve_pseudoinverse(X, y);

    ASSERT_EQ(result.coefficients.size(), 1);
    EXPECT_NEAR(result.coefficients[0], 3.0f, 1e-4f);
    EXPECT_NEAR(result.intercept, 2.0f, 1e-4f);
}

TEST(SVDTest, SolvesMultipleLinearRelationship) {
    Matrix X(6, 2);
    Vector y(6);

    for (std::size_t i = 0; i < 6; ++i) {
        const float x1 = static_cast<float>(i + 1);
        const float x2 = static_cast<float>((i + 1) * (i + 1));

        X(i, 0) = x1;
        X(i, 1) = x2;
        y[i] = 2.0f * x1 - 0.5f * x2 + 4.0f;
    }

    const auto result = SVD::solve_pseudoinverse(X, y);

    ASSERT_EQ(result.coefficients.size(), 2);
    EXPECT_NEAR(result.coefficients[0], 2.0f, 1e-4f);
    EXPECT_NEAR(result.coefficients[1], -0.5f, 1e-4f);
    EXPECT_NEAR(result.intercept, 4.0f, 1e-4f);
}

TEST(SVDTest, HandlesRankDeficientFeaturesWithMinimumNormSolution) {
    Matrix X(4, 2);
    Vector y(4);

    for (std::size_t i = 0; i < 4; ++i) {
        const float value = static_cast<float>(i + 1);

        X(i, 0) = value;
        X(i, 1) = 2.0f * value;
        y[i] = 3.0f * value + 1.0f;
    }

    const auto result = SVD::solve_pseudoinverse(X, y);

    ASSERT_EQ(result.coefficients.size(), 2);

    // Minimum-norm solution subject to:
    // w1 + 2*w2 = 3.
    EXPECT_NEAR(result.coefficients[0], 0.6f, 1e-4f);
    EXPECT_NEAR(result.coefficients[1], 1.2f, 1e-4f);
    EXPECT_NEAR(result.intercept, 1.0f, 1e-4f);
}

TEST(SVDTest, HandlesUnderdeterminedSystem) {
    Matrix X(1, 2);
    X(0, 0) = 1.0f;
    X(0, 1) = 1.0f;

    Vector y{5.0f};

    const auto result = SVD::solve_pseudoinverse(X, y);

    ASSERT_EQ(result.coefficients.size(), 2);

    // Minimum-norm solution of:
    // w1 + w2 + b = 5.
    EXPECT_NEAR(result.coefficients[0], 5.0f / 3.0f, 1e-4f);
    EXPECT_NEAR(result.coefficients[1], 5.0f / 3.0f, 1e-4f);
    EXPECT_NEAR(result.intercept, 5.0f / 3.0f, 1e-4f);
}

TEST(SVDTest, SolvesOverdeterminedNoisyDataset) {
    Matrix X(5, 1);
    Vector y{5.1f, 7.0f, 9.2f, 10.9f, 13.1f};

    for (std::size_t i = 0; i < 5; ++i) {
        X(i, 0) = static_cast<float>(i + 1);
    }

    const auto result = SVD::solve_pseudoinverse(X, y);

    ASSERT_EQ(result.coefficients.size(), 1);
    EXPECT_NEAR(result.coefficients[0], 1.99f, 0.03f);
    EXPECT_NEAR(result.intercept, 3.12f, 0.08f);
}

TEST(SVDTest, RejectsEmptyMatrix) {
    Matrix X;
    Vector y;

    EXPECT_THROW(
        static_cast<void>(SVD::solve_pseudoinverse(X, y)),
        std::invalid_argument
    );
}

TEST(SVDTest, RejectsTargetSizeMismatch) {
    Matrix X(3, 1);
    Vector y(2);

    EXPECT_THROW(
        static_cast<void>(SVD::solve_pseudoinverse(X, y)),
        std::invalid_argument
    );
}

TEST(SVDTest, RejectsNonFiniteFeatures) {
    Matrix X(3, 1);
    Vector y{1.0f, 2.0f, 3.0f};

    X(0, 0) = 1.0f;
    X(1, 0) = std::numeric_limits<float>::quiet_NaN();
    X(2, 0) = 3.0f;

    EXPECT_THROW(
        static_cast<void>(SVD::solve_pseudoinverse(X, y)),
        std::invalid_argument
    );
}

TEST(SVDTest, RejectsNonFiniteTargets) {
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
        static_cast<void>(SVD::solve_pseudoinverse(X, y)),
        std::invalid_argument
    );
}
