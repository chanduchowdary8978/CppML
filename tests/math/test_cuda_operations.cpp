
#include "cppml/math/cuda_operations.hpp"
#include "cppml/math/operations.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <stdexcept>

using cppml::Matrix;
using cppml::Vector;

// ============================================================
// CUDA Matrix-Vector Multiplication Tests
// ============================================================

TEST(CudaMathTest, MatvecMatchesCPUReference) {
    Matrix X(3, 2);

    X(0, 0) = 1.0f;
    X(0, 1) = 2.0f;
    X(1, 0) = 3.0f;
    X(1, 1) = 4.0f;
    X(2, 0) = 5.0f;
    X(2, 1) = 6.0f;

    Vector w{2.0f, 3.0f};

    Vector cpu_result = cppml::math::matvec(X, w);
    Vector gpu_result = cppml::math::cuda::matvec(X, w);

    ASSERT_EQ(cpu_result.size(), gpu_result.size());

    for (std::size_t i = 0; i < cpu_result.size(); ++i) {
        EXPECT_NEAR(gpu_result[i], cpu_result[i], 1e-5f);
    }
}

TEST(CudaMathTest, MatvecProducesExpectedValues) {
    Matrix X(2, 3);

    X(0, 0) = 1.0f;
    X(0, 1) = 2.0f;
    X(0, 2) = 3.0f;

    X(1, 0) = 4.0f;
    X(1, 1) = 5.0f;
    X(1, 2) = 6.0f;

    Vector w{1.0f, 2.0f, 3.0f};

    Vector result = cppml::math::cuda::matvec(X, w);

    ASSERT_EQ(result.size(), 2);
    EXPECT_FLOAT_EQ(result[0], 14.0f);
    EXPECT_FLOAT_EQ(result[1], 32.0f);
}

TEST(CudaMathTest, MatvecRejectsDimensionMismatch) {
    Matrix X(3, 2);
    Vector w(3);

    EXPECT_THROW(
        cppml::math::cuda::matvec(X, w),
        std::invalid_argument
    );
}

TEST(CudaMathTest, MatvecHandlesZeroRows) {
    Matrix X(0, 3);
    Vector w{1.0f, 2.0f, 3.0f};

    Vector result = cppml::math::cuda::matvec(X, w);

    EXPECT_TRUE(result.empty());
}

TEST(CudaMathTest, MatvecHandlesZeroColumns) {
    Matrix X(3, 0);
    Vector w;

    Vector result = cppml::math::cuda::matvec(X, w);

    ASSERT_EQ(result.size(), 3);

    for (std::size_t i = 0; i < result.size(); ++i) {
        EXPECT_FLOAT_EQ(result[i], 0.0f);
    }
}

// ============================================================
// CUDA Transpose Matrix-Vector Multiplication Tests
// ============================================================

TEST(CudaMathTest, TransposeMatvecMatchesCPUReference) {
    Matrix X(3, 2);

    X(0, 0) = 1.0f;
    X(0, 1) = 2.0f;
    X(1, 0) = 3.0f;
    X(1, 1) = 4.0f;
    X(2, 0) = 5.0f;
    X(2, 1) = 6.0f;

    Vector r{2.0f, 3.0f, 4.0f};

    Vector cpu_result = cppml::math::transpose_matvec(X, r);
    Vector gpu_result = cppml::math::cuda::transpose_matvec(X, r);

    ASSERT_EQ(cpu_result.size(), gpu_result.size());

    for (std::size_t i = 0; i < cpu_result.size(); ++i) {
        EXPECT_NEAR(gpu_result[i], cpu_result[i], 1e-5f);
    }
}

TEST(CudaMathTest, TransposeMatvecRejectsDimensionMismatch) {
    Matrix X(3, 2);
    Vector r(2);

    EXPECT_THROW(
        cppml::math::cuda::transpose_matvec(X, r),
        std::invalid_argument
    );
}

TEST(CudaMathTest, TransposeMatvecHandlesZeroRows) {
    Matrix X(0, 3);
    Vector r;

    Vector result = cppml::math::cuda::transpose_matvec(X, r);

    ASSERT_EQ(result.size(), 3);

    for (std::size_t i = 0; i < result.size(); ++i) {
        EXPECT_FLOAT_EQ(result[i], 0.0f);
    }
}

// ============================================================
// CUDA Vector Subtraction Tests
// ============================================================

TEST(CudaMathTest, SubtractMatchesCPUReference) {
    Vector a{10.0f, 20.0f, 30.0f, 40.0f};
    Vector b{1.0f, 2.0f, 3.0f, 4.0f};

    Vector cpu_result = cppml::math::subtract(a, b);
    Vector gpu_result = cppml::math::cuda::subtract(a, b);

    ASSERT_EQ(cpu_result.size(), gpu_result.size());

    for (std::size_t i = 0; i < cpu_result.size(); ++i) {
        EXPECT_NEAR(gpu_result[i], cpu_result[i], 1e-5f);
    }
}

TEST(CudaMathTest, SubtractRejectsDimensionMismatch) {
    Vector a{1.0f, 2.0f, 3.0f};
    Vector b{1.0f, 2.0f};

    EXPECT_THROW(
        cppml::math::cuda::subtract(a, b),
        std::invalid_argument
    );
}

TEST(CudaMathTest, SubtractHandlesEmptyVectors) {
    Vector a;
    Vector b;

    Vector result = cppml::math::cuda::subtract(a, b);

    EXPECT_TRUE(result.empty());
}

TEST(CudaMathTest, SubtractHandlesNegativeResults) {
    Vector a{1.0f, 2.0f, 3.0f};
    Vector b{4.0f, 5.0f, 6.0f};

    Vector result = cppml::math::cuda::subtract(a, b);

    ASSERT_EQ(result.size(), 3);

    EXPECT_FLOAT_EQ(result[0], -3.0f);
    EXPECT_FLOAT_EQ(result[1], -3.0f);
    EXPECT_FLOAT_EQ(result[2], -3.0f);
}

// ============================================================
// CUDA Dot Product Tests
// ============================================================

TEST(CudaMathTest, DotMatchesCPUReference) {
    Vector a{1.0f, 2.0f, 3.0f, 4.0f};
    Vector b{5.0f, 6.0f, 7.0f, 8.0f};

    const float cpu_result = cppml::math::dot(a, b);
    const float gpu_result = cppml::math::cuda::dot(a, b);

    EXPECT_NEAR(gpu_result, cpu_result, 1e-5f);
}

TEST(CudaMathTest, DotProducesExpectedValue) {
    Vector a{1.0f, 2.0f, 3.0f};
    Vector b{4.0f, 5.0f, 6.0f};

    const float result = cppml::math::cuda::dot(a, b);

    EXPECT_FLOAT_EQ(result, 32.0f);
}

TEST(CudaMathTest, DotRejectsDimensionMismatch) {
    Vector a{1.0f, 2.0f, 3.0f};
    Vector b{4.0f, 5.0f};

    EXPECT_THROW(
        cppml::math::cuda::dot(a, b),
        std::invalid_argument
    );
}

TEST(CudaMathTest, DotHandlesEmptyVectors) {
    Vector a;
    Vector b;

    const float result = cppml::math::cuda::dot(a, b);

    EXPECT_FLOAT_EQ(result, 0.0f);
}

// ============================================================
// CUDA Sum Tests
// ============================================================

TEST(CudaMathTest, SumMatchesCPUReference) {
    Vector v{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    const float cpu_result = cppml::math::sum(v);
    const float gpu_result = cppml::math::cuda::sum(v);

    EXPECT_NEAR(gpu_result, cpu_result, 1e-5f);
}

TEST(CudaMathTest, SumProducesExpectedValue) {
    Vector v{10.0f, 20.0f, 30.0f, 40.0f};

    const float result = cppml::math::cuda::sum(v);

    EXPECT_FLOAT_EQ(result, 100.0f);
}

TEST(CudaMathTest, SumHandlesEmptyVector) {
    Vector v;

    const float result = cppml::math::cuda::sum(v);

    EXPECT_FLOAT_EQ(result, 0.0f);
}

TEST(CudaMathTest, SumHandlesMultipleBlocks) {
    Vector v(1000, 1.0f);

    const float result = cppml::math::cuda::sum(v);

    EXPECT_NEAR(result, 1000.0f, 1e-4f);
}

// ============================================================
// CUDA Mean Tests
// ============================================================

TEST(CudaMathTest, MeanMatchesCPUReference) {
    Vector v{2.0f, 4.0f, 6.0f, 8.0f, 10.0f};

    const float cpu_result = cppml::math::mean(v);
    const float gpu_result = cppml::math::cuda::mean(v);

    EXPECT_NEAR(gpu_result, cpu_result, 1e-5f);
}

TEST(CudaMathTest, MeanProducesExpectedValue) {
    Vector v{10.0f, 20.0f, 30.0f, 40.0f};

    const float result = cppml::math::cuda::mean(v);

    EXPECT_FLOAT_EQ(result, 25.0f);
}

TEST(CudaMathTest, MeanRejectsEmptyVector) {
    Vector v;

    EXPECT_THROW(
        cppml::math::cuda::mean(v),
        std::invalid_argument
    );
}

// ============================================================
// CUDA Add Scalar Tests
// ============================================================

TEST(CudaMathTest, AddScalarMatchesCPUReference) {
    Vector v{1.0f, 2.0f, 3.0f, 4.0f};
    const float value = 5.0f;

    Vector cpu_result = cppml::math::add_scalar(v, value);
    Vector gpu_result = cppml::math::cuda::add_scalar(v, value);

    ASSERT_EQ(cpu_result.size(), gpu_result.size());

    for (std::size_t i = 0; i < cpu_result.size(); ++i) {
        EXPECT_NEAR(gpu_result[i], cpu_result[i], 1e-5f);
    }
}

TEST(CudaMathTest, AddScalarProducesExpectedValues) {
    Vector v{10.0f, 20.0f, 30.0f};

    Vector result = cppml::math::cuda::add_scalar(v, 5.0f);

    ASSERT_EQ(result.size(), 3);

    EXPECT_FLOAT_EQ(result[0], 15.0f);
    EXPECT_FLOAT_EQ(result[1], 25.0f);
    EXPECT_FLOAT_EQ(result[2], 35.0f);
}

TEST(CudaMathTest, AddScalarHandlesNegativeScalar) {
    Vector v{10.0f, 20.0f, 30.0f};

    Vector result = cppml::math::cuda::add_scalar(v, -5.0f);

    ASSERT_EQ(result.size(), 3);

    EXPECT_FLOAT_EQ(result[0], 5.0f);
    EXPECT_FLOAT_EQ(result[1], 15.0f);
    EXPECT_FLOAT_EQ(result[2], 25.0f);
}

TEST(CudaMathTest, AddScalarHandlesEmptyVector) {
    Vector v;

    Vector result = cppml::math::cuda::add_scalar(v, 10.0f);

    EXPECT_TRUE(result.empty());
}
