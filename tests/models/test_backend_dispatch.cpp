
#include "cppml/models/linear_regression.hpp"
#include "cppml/models/execution_backend.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

#ifdef CPPML_HAS_CUDA
#include <cuda_runtime_api.h>
#endif

using cppml::Matrix;
using cppml::Vector;
using cppml::models::ExecutionBackend;
using cppml::models::LinearRegression;

namespace {

void create_simple_dataset(Matrix& X, Vector& y) {
    for (std::size_t i = 0; i < X.rows(); ++i) {
        const float x = static_cast<float>(i + 1);
        X(i, 0) = x;
        y[i] = 3.0f * x + 2.0f;
    }
}

} // namespace

TEST(BackendDispatchTest, DefaultsToCPU) {
    LinearRegression model;
    EXPECT_EQ(model.backend(), ExecutionBackend::CPU);
}

TEST(BackendDispatchTest, ExplicitCPUSelection) {
    LinearRegression model(ExecutionBackend::CPU);
    EXPECT_EQ(model.backend(), ExecutionBackend::CPU);
}

TEST(BackendDispatchTest, ExplicitCUDASelection) {
#ifdef CPPML_HAS_CUDA
    LinearRegression model{ExecutionBackend::CUDA};
    EXPECT_EQ(model.backend(), ExecutionBackend::CUDA);
#else
    EXPECT_THROW(
        (void)LinearRegression{ExecutionBackend::CUDA},
        std::runtime_error
    );
#endif
}

TEST(BackendDispatchTest, RejectsInvalidBackend) {
    EXPECT_THROW(
        LinearRegression(static_cast<ExecutionBackend>(99)),
        std::invalid_argument
    );
}

TEST(BackendDispatchTest, CPUBackendTrainsCorrectly) {
    Matrix X(5, 1);
    Vector y(5);
    create_simple_dataset(X, y);

    LinearRegression model(ExecutionBackend::CPU);
    model.fit(X, y, 0.01f, 5000);

    EXPECT_EQ(model.active_backend(), ExecutionBackend::CPU);
    EXPECT_NEAR(model.coefficients()[0], 3.0f, 0.02f);
    EXPECT_NEAR(model.intercept(), 2.0f, 0.05f);
}

TEST(BackendDispatchTest, CPUBackendPredictsCorrectly) {
    Matrix X(5, 1);
    Vector y(5);
    create_simple_dataset(X, y);

    LinearRegression model(ExecutionBackend::CPU);
    model.fit(X, y, 0.01f, 5000);

    Vector predictions = model.predict(X);

    ASSERT_EQ(predictions.size(), y.size());

    for (std::size_t i = 0; i < y.size(); ++i) {
        EXPECT_NEAR(predictions[i], y[i], 0.05f);
    }
}

TEST(BackendDispatchTest, AutoUsesCPUForSmallWorkload) {
    Matrix X(5, 1);
    Vector y(5);
    create_simple_dataset(X, y);

    LinearRegression model(ExecutionBackend::AUTO);
    model.fit(X, y, 0.01f, 5000);

    EXPECT_EQ(model.active_backend(), ExecutionBackend::CPU);
}

TEST(BackendDispatchTest, AutoTrainsCorrectly) {
    Matrix X(5, 1);
    Vector y(5);
    create_simple_dataset(X, y);

    LinearRegression model(ExecutionBackend::AUTO);
    model.fit(X, y, 0.01f, 5000);

    EXPECT_NEAR(model.coefficients()[0], 3.0f, 0.02f);
    EXPECT_NEAR(model.intercept(), 2.0f, 0.05f);
}

#ifdef CPPML_HAS_CUDA
TEST(BackendDispatchTest, CUDABackendTrainsCorrectly) {
    Matrix X(5, 1);
    Vector y(5);
    create_simple_dataset(X, y);

    LinearRegression model(ExecutionBackend::CUDA);
    model.fit(X, y, 0.01f, 5000);

    EXPECT_EQ(model.active_backend(), ExecutionBackend::CUDA);
    EXPECT_NEAR(model.coefficients()[0], 3.0f, 0.02f);
    EXPECT_NEAR(model.intercept(), 2.0f, 0.05f);
}

TEST(BackendDispatchTest, CUDABackendPredictsCorrectly) {
    Matrix X(5, 1);
    Vector y(5);
    create_simple_dataset(X, y);

    LinearRegression model(ExecutionBackend::CUDA);
    model.fit(X, y, 0.01f, 5000);

    Vector predictions = model.predict(X);

    ASSERT_EQ(predictions.size(), y.size());

    for (std::size_t i = 0; i < y.size(); ++i) {
        EXPECT_NEAR(predictions[i], y[i], 0.05f);
    }
}
#endif

TEST(BackendDispatchTest, RejectsInvalidLearningRate) {
    Matrix X(5, 1);
    Vector y(5);
    create_simple_dataset(X, y);

    LinearRegression model(ExecutionBackend::CPU);

    EXPECT_THROW(
        model.fit(X, y, 0.0f, 100),
        std::invalid_argument
    );
}

TEST(BackendDispatchTest, RejectsZeroEpochs) {
    Matrix X(5, 1);
    Vector y(5);
    create_simple_dataset(X, y);

    LinearRegression model(ExecutionBackend::CPU);

    EXPECT_THROW(
        model.fit(X, y, 0.01f, 0),
        std::invalid_argument
    );
}

TEST(BackendDispatchTest, PredictBeforeFitThrows) {
    LinearRegression model(ExecutionBackend::CPU);
    Matrix X(2, 1);

    EXPECT_THROW(
        model.predict(X),
        std::logic_error
    );
}
