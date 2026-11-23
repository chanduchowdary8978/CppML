
#include "cppml/models/cuda_linear_regression.hpp"
#include "cppml/models/linear_regression.hpp"
#include "cppml/math/metrics.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

using cppml::Matrix;
using cppml::Vector;
using cppml::models::CudaLinearRegression;
using cppml::models::LinearRegression;

TEST(CudaLinearRegressionTest, LearnsSimpleLinearRelationship) {
    Matrix X(5, 1);

    for (std::size_t i = 0; i < 5; ++i) {
        X(i, 0) = static_cast<float>(i + 1);
    }

    Vector y{5.0f, 8.0f, 11.0f, 14.0f, 17.0f};

    CudaLinearRegression model;
    model.fit(X, y, 0.01f, 5000);

    EXPECT_NEAR(model.coefficients()[0], 3.0f, 0.02f);
    EXPECT_NEAR(model.intercept(), 2.0f, 0.05f);

    const Vector predictions = model.predict(X);

    for (std::size_t i = 0; i < y.size(); ++i) {
        EXPECT_NEAR(predictions[i], y[i], 0.05f);
    }
}

TEST(CudaLinearRegressionTest, LearnsMultipleLinearRelationship) {
    Matrix X(6, 2);

    X(0, 0) = 1.0f; X(0, 1) = 2.0f;
    X(1, 0) = 2.0f; X(1, 1) = 1.0f;
    X(2, 0) = 3.0f; X(2, 1) = 2.0f;
    X(3, 0) = 4.0f; X(3, 1) = 1.0f;
    X(4, 0) = 5.0f; X(4, 1) = 3.0f;
    X(5, 0) = 6.0f; X(5, 1) = 2.0f;

    Vector y{13.0f, 12.0f, 17.0f, 16.0f, 24.0f, 23.0f};

    CudaLinearRegression model;
    model.fit(X, y, 0.01f, 10000);

    EXPECT_NEAR(model.coefficients()[0], 2.0f, 0.05f);
    EXPECT_NEAR(model.coefficients()[1], 3.0f, 0.05f);
    EXPECT_NEAR(model.intercept(), 5.0f, 0.1f);

    const Vector predictions = model.predict(X);

    const float mse =
        cppml::math::Metrics::mean_squared_error(y, predictions);

    EXPECT_LT(mse, 0.01f);
}

TEST(CudaLinearRegressionTest, MatchesCPUReference) {
    Matrix X(5, 2);

    X(0, 0) = 1.0f; X(0, 1) = 2.0f;
    X(1, 0) = 2.0f; X(1, 1) = 1.0f;
    X(2, 0) = 3.0f; X(2, 1) = 2.0f;
    X(3, 0) = 4.0f; X(3, 1) = 1.0f;
    X(4, 0) = 5.0f; X(4, 1) = 3.0f;

    Vector y{13.0f, 12.0f, 17.0f, 16.0f, 24.0f};

    LinearRegression cpu_model;
    CudaLinearRegression cuda_model;

    cpu_model.fit(X, y, 0.01f, 3000);
    cuda_model.fit(X, y, 0.01f, 3000);

    ASSERT_EQ(
        cpu_model.coefficients().size(),
        cuda_model.coefficients().size()
    );

    for (std::size_t i = 0; i < cpu_model.coefficients().size(); ++i) {
        EXPECT_NEAR(
            cuda_model.coefficients()[i],
            cpu_model.coefficients()[i],
            0.01f
        );
    }

    EXPECT_NEAR(
        cuda_model.intercept(),
        cpu_model.intercept(),
        0.02f
    );
}

TEST(CudaLinearRegressionTest, PredictBeforeFitThrows) {
    CudaLinearRegression model;
    Matrix X(2, 1);

    EXPECT_THROW(model.predict(X), std::logic_error);
}

TEST(CudaLinearRegressionTest, RejectsTargetSizeMismatch) {
    Matrix X(3, 2);
    Vector y(2);

    CudaLinearRegression model;

    EXPECT_THROW(
        model.fit(X, y, 0.01f, 100),
        std::invalid_argument
    );
}

TEST(CudaLinearRegressionTest, RejectsEmptyTrainingData) {
    Matrix X;
    Vector y;

    CudaLinearRegression model;

    EXPECT_THROW(
        model.fit(X, y, 0.01f, 100),
        std::invalid_argument
    );
}

TEST(CudaLinearRegressionTest, RejectsZeroFeatures) {
    Matrix X(3, 0);
    Vector y{1.0f, 2.0f, 3.0f};

    CudaLinearRegression model;

    EXPECT_THROW(
        model.fit(X, y, 0.01f, 100),
        std::invalid_argument
    );
}

TEST(CudaLinearRegressionTest, RejectsInvalidLearningRate) {
    Matrix X(2, 1);
    Vector y{1.0f, 2.0f};

    CudaLinearRegression model;

    EXPECT_THROW(
        model.fit(X, y, 0.0f, 100),
        std::invalid_argument
    );

    EXPECT_THROW(
        model.fit(X, y, -0.01f, 100),
        std::invalid_argument
    );

    EXPECT_THROW(
        model.fit(
            X,
            y,
            std::numeric_limits<float>::infinity(),
            100
        ),
        std::invalid_argument
    );

    EXPECT_THROW(
        model.fit(
            X,
            y,
            std::numeric_limits<float>::quiet_NaN(),
            100
        ),
        std::invalid_argument
    );
}

TEST(CudaLinearRegressionTest, RejectsZeroEpochs) {
    Matrix X(2, 1);
    Vector y{1.0f, 2.0f};

    CudaLinearRegression model;

    EXPECT_THROW(
        model.fit(X, y, 0.01f, 0),
        std::invalid_argument
    );
}

TEST(CudaLinearRegressionTest, RejectsPredictionFeatureMismatch) {
    Matrix training_X(3, 2);

    training_X(0, 0) = 1.0f;
    training_X(0, 1) = 2.0f;
    training_X(1, 0) = 2.0f;
    training_X(1, 1) = 3.0f;
    training_X(2, 0) = 3.0f;
    training_X(2, 1) = 4.0f;

    Vector y{5.0f, 8.0f, 11.0f};

    CudaLinearRegression model;
    model.fit(training_X, y, 0.01f, 1000);

    Matrix invalid_X(2, 3);

    EXPECT_THROW(
        model.predict(invalid_X),
        std::invalid_argument
    );
}

TEST(CudaLinearRegressionTest, RejectsNonFiniteTrainingFeatures) {
    Matrix X(2, 1);

    X(0, 0) = 1.0f;
    X(1, 0) = std::numeric_limits<float>::quiet_NaN();

    Vector y{1.0f, 2.0f};

    CudaLinearRegression model;

    EXPECT_THROW(
        model.fit(X, y, 0.01f, 100),
        std::invalid_argument
    );
}

TEST(CudaLinearRegressionTest, RejectsNonFiniteTargets) {
    Matrix X(2, 1);

    X(0, 0) = 1.0f;
    X(1, 0) = 2.0f;

    Vector y{
        1.0f,
        std::numeric_limits<float>::infinity()
    };

    CudaLinearRegression model;

    EXPECT_THROW(
        model.fit(X, y, 0.01f, 100),
        std::invalid_argument
    );
}

TEST(CudaLinearRegressionTest, HandlesEmptyPredictionMatrix) {
    Matrix X(3, 1);
    Vector y{5.0f, 8.0f, 11.0f};

    CudaLinearRegression model;
    model.fit(X, y, 0.01f, 1000);

    Matrix empty_X(0, 1);

    const Vector predictions = model.predict(empty_X);

    EXPECT_TRUE(predictions.empty());
}

// ============================================================
// Numerical robustness tests
// ============================================================

TEST(CudaLinearRegressionTest, HandlesCancellationInGradientAccumulation) {
    Matrix X(4, 1);

    X(0, 0) = 100000000.0f;
    X(1, 0) = 1.0f;
    X(2, 0) = -100000000.0f;
    X(3, 0) = 1.0f;

    Vector y{1.0f, 1.0f, 1.0f, 1.0f};

    CudaLinearRegression model;
    model.fit(X, y, 0.1f, 1);

    // Exact gradient:
    // (-1e8 - 1 + 1e8 - 1) / 4 = -0.5
    // w_new = 0 - 0.1 * (-0.5) = 0.05

    EXPECT_NEAR(model.coefficients()[0], 0.05f, 1e-5f);
    EXPECT_NEAR(model.intercept(), 0.1f, 1e-6f);
}

TEST(CudaLinearRegressionTest, ThrowsWhenWeightUpdateExceedsFloatRange) {
    Matrix X(1, 1);
    X(0, 0) = 1.0f;

    Vector y{
        std::numeric_limits<float>::max()
    };

    CudaLinearRegression model;

    EXPECT_THROW(
        model.fit(X, y, 2.0f, 1),
        std::overflow_error
    );
}

TEST(CudaLinearRegressionTest, ThrowsWhenPredictionExceedsFloatRange) {
    Matrix training_X(1, 1);
    training_X(0, 0) = 1.0f;

    Vector y{4.0f};

    CudaLinearRegression model;
    model.fit(training_X, y, 0.5f, 1);

    Matrix prediction_X(1, 1);
    prediction_X(0, 0) = std::numeric_limits<float>::max();

    EXPECT_THROW(
        model.predict(prediction_X),
        std::overflow_error
    );
}

TEST(CudaLinearRegressionTest, FailedFitLeavesModelUnfitted) {
    Matrix X(1, 1);
    X(0, 0) = 1.0f;

    Vector y{
        std::numeric_limits<float>::max()
    };

    CudaLinearRegression model;

    EXPECT_THROW(
        model.fit(X, y, 2.0f, 1),
        std::overflow_error
    );

    EXPECT_THROW(
        model.predict(X),
        std::logic_error
    );
}
