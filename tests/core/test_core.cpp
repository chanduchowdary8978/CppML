
#include "cppml/core/matrix.hpp"
#include "cppml/core/vector.hpp"

#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>

using cppml::Matrix;
using cppml::Vector;

// ---------------- VECTOR TESTS ----------------

TEST(VectorTest, DefaultConstruction) {
    Vector v;

    EXPECT_EQ(v.size(), 0);
    EXPECT_TRUE(v.empty());
}

TEST(VectorTest, SizeConstruction) {
    Vector v(5);

    EXPECT_EQ(v.size(), 5);
    EXPECT_FALSE(v.empty());

    for (std::size_t i = 0; i < v.size(); ++i) {
        EXPECT_FLOAT_EQ(v[i], 0.0f);
    }
}

TEST(VectorTest, ValueConstruction) {
    Vector v(4, 3.5f);

    EXPECT_EQ(v.size(), 4);

    for (std::size_t i = 0; i < v.size(); ++i) {
        EXPECT_FLOAT_EQ(v[i], 3.5f);
    }
}

TEST(VectorTest, InitializerListConstruction) {
    Vector v{1.0f, 2.0f, 3.0f};

    ASSERT_EQ(v.size(), 3);
    EXPECT_FLOAT_EQ(v[0], 1.0f);
    EXPECT_FLOAT_EQ(v[1], 2.0f);
    EXPECT_FLOAT_EQ(v[2], 3.0f);
}

TEST(VectorTest, ElementModification) {
    Vector v{1.0f, 2.0f, 3.0f};

    v[1] = 10.0f;

    EXPECT_FLOAT_EQ(v[1], 10.0f);
}

TEST(VectorTest, Fill) {
    Vector v(5);
    v.fill(7.0f);

    for (std::size_t i = 0; i < v.size(); ++i) {
        EXPECT_FLOAT_EQ(v[i], 7.0f);
    }
}

TEST(VectorTest, CheckedAccess) {
    Vector v{1.0f, 2.0f};

    EXPECT_FLOAT_EQ(v.at(0), 1.0f);
    EXPECT_THROW(v.at(2), std::out_of_range);
}

// ---------------- MATRIX TESTS ----------------

TEST(MatrixTest, DefaultConstruction) {
    Matrix m;

    EXPECT_EQ(m.rows(), 0);
    EXPECT_EQ(m.cols(), 0);
    EXPECT_EQ(m.size(), 0);
    EXPECT_TRUE(m.empty());
}

TEST(MatrixTest, DimensionConstruction) {
    Matrix m(2, 3);

    EXPECT_EQ(m.rows(), 2);
    EXPECT_EQ(m.cols(), 3);
    EXPECT_EQ(m.size(), 6);
}

TEST(MatrixTest, ValueConstruction) {
    Matrix m(2, 3, 4.5f);

    for (std::size_t i = 0; i < m.rows(); ++i) {
        for (std::size_t j = 0; j < m.cols(); ++j) {
            EXPECT_FLOAT_EQ(m(i, j), 4.5f);
        }
    }
}

TEST(MatrixTest, RowMajorLayout) {
    Matrix m(2, 3);

    m(0, 0) = 1.0f;
    m(0, 1) = 2.0f;
    m(0, 2) = 3.0f;
    m(1, 0) = 4.0f;
    m(1, 1) = 5.0f;
    m(1, 2) = 6.0f;

    const float expected[] = {1, 2, 3, 4, 5, 6};

    for (std::size_t i = 0; i < m.size(); ++i) {
        EXPECT_FLOAT_EQ(m.data()[i], expected[i]);
    }
}

TEST(MatrixTest, ElementModification) {
    Matrix m(2, 2);

    m(1, 0) = 9.0f;

    EXPECT_FLOAT_EQ(m(1, 0), 9.0f);
}

TEST(MatrixTest, Fill) {
    Matrix m(2, 3);
    m.fill(8.0f);

    for (std::size_t i = 0; i < m.size(); ++i) {
        EXPECT_FLOAT_EQ(m.data()[i], 8.0f);
    }
}

TEST(MatrixTest, CheckedAccess) {
    Matrix m(2, 3);

    EXPECT_NO_THROW(m.at(1, 2) = 5.0f);
    EXPECT_THROW(m.at(2, 0), std::out_of_range);
    EXPECT_THROW(m.at(0, 3), std::out_of_range);
}

TEST(MatrixTest, ZeroDimension) {
    Matrix m(0, 5);

    EXPECT_EQ(m.rows(), 0);
    EXPECT_EQ(m.cols(), 5);
    EXPECT_EQ(m.size(), 0);
    EXPECT_TRUE(m.empty());
}

TEST(MatrixTest, DimensionOverflow) {
    const auto max = std::numeric_limits<std::size_t>::max();

    EXPECT_THROW(Matrix(max, 2), std::length_error);
}
