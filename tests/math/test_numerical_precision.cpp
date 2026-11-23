
#include "cppml/core/matrix.hpp"
#include "cppml/core/vector.hpp"
#include "cppml/math/operations.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>

using cppml::Matrix;
using cppml::Vector;

TEST(NumericalPrecisionTest, SumHandlesCatastrophicCancellation) {
    Vector values{1.0e8f, 1.0f, -1.0e8f};
    EXPECT_FLOAT_EQ(cppml::math::sum(values), 1.0f);
}

TEST(NumericalPrecisionTest, MeanHandlesCatastrophicCancellation) {
    Vector values{1.0e8f, 1.0f, -1.0e8f};
    EXPECT_FLOAT_EQ(cppml::math::mean(values), 1.0f / 3.0f);
}

TEST(NumericalPrecisionTest, DotHandlesCatastrophicCancellation) {
    Vector a{1.0e8f, 1.0f, -1.0e8f};
    Vector b{1.0f, 1.0f, 1.0f};
    EXPECT_FLOAT_EQ(cppml::math::dot(a, b), 1.0f);
}

TEST(NumericalPrecisionTest, MatvecHandlesCatastrophicCancellation) {
    Matrix X(1, 3);
    X(0, 0) = 1.0e8f;
    X(0, 1) = 1.0f;
    X(0, 2) = -1.0e8f;

    Vector w{1.0f, 1.0f, 1.0f};
    const Vector result = cppml::math::matvec(X, w);

    ASSERT_EQ(result.size(), 1);
    EXPECT_FLOAT_EQ(result[0], 1.0f);
}

TEST(NumericalPrecisionTest, TransposeMatvecHandlesCatastrophicCancellation) {
    Matrix X(3, 1);
    X(0, 0) = 1.0e8f;
    X(1, 0) = 1.0f;
    X(2, 0) = -1.0e8f;

    Vector r{1.0f, 1.0f, 1.0f};
    const Vector result = cppml::math::transpose_matvec(X, r);

    ASSERT_EQ(result.size(), 1);
    EXPECT_FLOAT_EQ(result[0], 1.0f);
}

TEST(NumericalPrecisionTest, SumOfEmptyVectorIsZero) {
    Vector values;
    EXPECT_FLOAT_EQ(cppml::math::sum(values), 0.0f);
}

TEST(NumericalPrecisionTest, DotOfEmptyVectorsIsZero) {
    Vector a;
    Vector b;
    EXPECT_FLOAT_EQ(cppml::math::dot(a, b), 0.0f);
}

TEST(NumericalPrecisionTest, MatvecWithZeroColumnsProducesZeros) {
    Matrix X(3, 0);
    Vector w;

    const Vector result = cppml::math::matvec(X, w);

    ASSERT_EQ(result.size(), 3);
    for (std::size_t i = 0; i < result.size(); ++i) {
        EXPECT_FLOAT_EQ(result[i], 0.0f);
    }
}

TEST(NumericalPrecisionTest, TransposeMatvecWithZeroRowsProducesZeros) {
    Matrix X(0, 3);
    Vector r;

    const Vector result = cppml::math::transpose_matvec(X, r);

    ASSERT_EQ(result.size(), 3);
    for (std::size_t i = 0; i < result.size(); ++i) {
        EXPECT_FLOAT_EQ(result[i], 0.0f);
    }
}

TEST(NumericalPrecisionTest, DoubleReferenceConfirmsDotProductAccuracy) {
    Vector a{1.0e8f, 1.0f, -1.0e8f, 3.0f, 4.0f};
    Vector b{1.0f, 1.0f, 1.0f, 2.0f, 3.0f};

    double reference = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        reference +=
            static_cast<double>(a[i]) *
            static_cast<double>(b[i]);
    }

    EXPECT_NEAR(
        static_cast<double>(cppml::math::dot(a, b)),
        reference,
        1.0e-6
    );
}

// ---------- NON-FINITE INPUT VALIDATION ----------

TEST(NumericalPrecisionTest, SumRejectsNaN) {
    Vector v{1.0f, std::numeric_limits<float>::quiet_NaN()};
    EXPECT_THROW(cppml::math::sum(v), std::invalid_argument);
}

TEST(NumericalPrecisionTest, SumRejectsInfinity) {
    Vector v{1.0f, std::numeric_limits<float>::infinity()};
    EXPECT_THROW(cppml::math::sum(v), std::invalid_argument);
}

TEST(NumericalPrecisionTest, MeanRejectsNaN) {
    Vector v{std::numeric_limits<float>::quiet_NaN()};
    EXPECT_THROW(cppml::math::mean(v), std::invalid_argument);
}

TEST(NumericalPrecisionTest, DotRejectsNonFiniteInput) {
    Vector a{1.0f, 2.0f};
    Vector b{1.0f, std::numeric_limits<float>::infinity()};
    EXPECT_THROW(cppml::math::dot(a, b), std::invalid_argument);
}

TEST(NumericalPrecisionTest, MatvecRejectsNonFiniteMatrixInput) {
    Matrix X(1, 1);
    X(0, 0) = std::numeric_limits<float>::quiet_NaN();

    Vector w{1.0f};
    EXPECT_THROW(cppml::math::matvec(X, w), std::invalid_argument);
}

TEST(NumericalPrecisionTest, MatvecRejectsNonFiniteVectorInput) {
    Matrix X(1, 1);
    X(0, 0) = 1.0f;

    Vector w{std::numeric_limits<float>::infinity()};
    EXPECT_THROW(cppml::math::matvec(X, w), std::invalid_argument);
}

TEST(NumericalPrecisionTest, TransposeMatvecRejectsNonFiniteInput) {
    Matrix X(1, 1);
    X(0, 0) = 1.0f;

    Vector r{std::numeric_limits<float>::quiet_NaN()};
    EXPECT_THROW(cppml::math::transpose_matvec(X, r), std::invalid_argument);
}

TEST(NumericalPrecisionTest, SubtractRejectsNonFiniteInput) {
    Vector a{1.0f};
    Vector b{std::numeric_limits<float>::infinity()};
    EXPECT_THROW(cppml::math::subtract(a, b), std::invalid_argument);
}

TEST(NumericalPrecisionTest, AddScalarRejectsNonFiniteVectorInput) {
    Vector v{std::numeric_limits<float>::quiet_NaN()};
    EXPECT_THROW(cppml::math::add_scalar(v, 1.0f), std::invalid_argument);
}

TEST(NumericalPrecisionTest, AddScalarRejectsNonFiniteScalar) {
    Vector v{1.0f};
    EXPECT_THROW(
        cppml::math::add_scalar(v, std::numeric_limits<float>::infinity()),
        std::invalid_argument
    );
}

// ---------- FLOAT OUTPUT OVERFLOW ----------

TEST(NumericalPrecisionTest, SumThrowsWhenResultExceedsFloatRange) {
    const float maximum = std::numeric_limits<float>::max();
    Vector v{maximum, maximum};

    EXPECT_THROW(cppml::math::sum(v), std::overflow_error);
}

TEST(NumericalPrecisionTest, DotThrowsWhenResultExceedsFloatRange) {
    const float maximum = std::numeric_limits<float>::max();
    Vector a{maximum};
    Vector b{2.0f};

    EXPECT_THROW(cppml::math::dot(a, b), std::overflow_error);
}

TEST(NumericalPrecisionTest, MatvecThrowsWhenResultExceedsFloatRange) {
    const float maximum = std::numeric_limits<float>::max();

    Matrix X(1, 1);
    X(0, 0) = maximum;

    Vector w{2.0f};

    EXPECT_THROW(cppml::math::matvec(X, w), std::overflow_error);
}

TEST(NumericalPrecisionTest, TransposeMatvecThrowsWhenResultExceedsFloatRange) {
    const float maximum = std::numeric_limits<float>::max();

    Matrix X(1, 1);
    X(0, 0) = maximum;

    Vector r{2.0f};

    EXPECT_THROW(
        cppml::math::transpose_matvec(X, r),
        std::overflow_error
    );
}

TEST(NumericalPrecisionTest, SubtractThrowsWhenResultExceedsFloatRange) {
    const float maximum = std::numeric_limits<float>::max();

    Vector a{maximum};
    Vector b{-maximum};

    EXPECT_THROW(cppml::math::subtract(a, b), std::overflow_error);
}

TEST(NumericalPrecisionTest, AddScalarThrowsWhenResultExceedsFloatRange) {
    const float maximum = std::numeric_limits<float>::max();

    Vector v{maximum};

    EXPECT_THROW(
        cppml::math::add_scalar(v, maximum),
        std::overflow_error
    );
}

// ---------- DOUBLE INTERMEDIATE RANGE ----------

TEST(NumericalPrecisionTest, DotAllowsLargeIntermediateWithRepresentableFinalResult) {
    const float maximum = std::numeric_limits<float>::max();

    Vector a{maximum, maximum};
    Vector b{1.0f, -1.0f};

    EXPECT_FLOAT_EQ(cppml::math::dot(a, b), 0.0f);
}

TEST(NumericalPrecisionTest, SumAllowsLargeIntermediateWithRepresentableFinalResult) {
    const float maximum = std::numeric_limits<float>::max();

    Vector v{maximum, maximum, -maximum, -maximum, 5.0f};

    EXPECT_FLOAT_EQ(cppml::math::sum(v), 5.0f);
}
