
#include "cppml/models/linear_regression.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>

using cppml::Matrix;
using cppml::Vector;
using cppml::models::LinearRegression;

TEST(LinearRegressionTest, PredictBeforeFitThrows) {
    LinearRegression model;
    Matrix X(1, 1, 1.0f);

    EXPECT_THROW(model.predict(X), std::logic_error);
}

TEST(LinearRegressionTest, LearnsSimpleLinearRelationship) {
    Matrix X(5, 1);
    Vector y{-4.0f, -1.0f, 2.0f, 5.0f, 8.0f};

    for (std::size_t i = 0; i < 5; ++i) {
        X(i, 0) = static_cast<float>(i) - 2.0f;
    }

    LinearRegression model;
    model.fit(X, y, 0.1f, 1000);

    EXPECT_NEAR(model.coefficients()[0], 3.0f, 1e-3f);
    EXPECT_NEAR(model.intercept(), 2.0f, 1e-3f);

    Vector predictions = model.predict(X);

    for (std::size_t i = 0; i < y.size(); ++i) {
        EXPECT_NEAR(predictions[i], y[i], 1e-3f);
    }
}

TEST(LinearRegressionTest, LearnsMultipleLinearRelationship) {
    Matrix X(4, 2);
    Vector y(4);

    X(0, 0) = -1.0f; X(0, 1) = -1.0f;
    X(1, 0) = -1.0f; X(1, 1) =  1.0f;
    X(2, 0) =  1.0f; X(2, 1) = -1.0f;
    X(3, 0) =  1.0f; X(3, 1) =  1.0f;

    for (std::size_t i = 0; i < 4; ++i) {
        y[i] = 2.0f * X(i, 0) + 3.0f * X(i, 1) + 5.0f;
    }

    LinearRegression model;
    model.fit(X, y, 0.1f, 200);

    EXPECT_NEAR(model.coefficients()[0], 2.0f, 1e-3f);
    EXPECT_NEAR(model.coefficients()[1], 3.0f, 1e-3f);
    EXPECT_NEAR(model.intercept(), 5.0f, 1e-3f);
}

TEST(LinearRegressionTest, RejectsTargetSizeMismatch) {
    LinearRegression model;
    Matrix X(3, 2);
    Vector y(2);

    EXPECT_THROW(model.fit(X, y, 0.01f, 100), std::invalid_argument);
}

TEST(LinearRegressionTest, RejectsEmptyTrainingData) {
    LinearRegression model;
    Matrix X(0, 2);
    Vector y;

    EXPECT_THROW(model.fit(X, y, 0.01f, 100), std::invalid_argument);
}

TEST(LinearRegressionTest, RejectsZeroFeatures) {
    LinearRegression model;
    Matrix X(3, 0);
    Vector y(3);

    EXPECT_THROW(model.fit(X, y, 0.01f, 100), std::invalid_argument);
}

TEST(LinearRegressionTest, RejectsInvalidLearningRate) {
    LinearRegression model;
    Matrix X(2, 1);
    Vector y{1.0f, 2.0f};

    EXPECT_THROW(model.fit(X, y, 0.0f, 100), std::invalid_argument);
    EXPECT_THROW(model.fit(X, y, -0.1f, 100), std::invalid_argument);
    EXPECT_THROW(
        model.fit(X, y, std::numeric_limits<float>::quiet_NaN(), 100),
        std::invalid_argument
    );
    EXPECT_THROW(
        model.fit(X, y, std::numeric_limits<float>::infinity(), 100),
        std::invalid_argument
    );
}

TEST(LinearRegressionTest, RejectsZeroEpochs) {
    LinearRegression model;
    Matrix X(2, 1);
    Vector y{1.0f, 2.0f};

    EXPECT_THROW(model.fit(X, y, 0.01f, 0), std::invalid_argument);
}

TEST(LinearRegressionTest, RejectsPredictionFeatureMismatch) {
    LinearRegression model;
    Matrix training_X(3, 2);
    Vector y{1.0f, 2.0f, 3.0f};

    model.fit(training_X, y, 0.01f, 10);

    Matrix prediction_X(2, 3);

    EXPECT_THROW(model.predict(prediction_X), std::invalid_argument);
}

// ---------- INPUT VALIDATION ----------

TEST(LinearRegressionTest, RejectsNonFiniteTrainingFeatures) {
    Matrix X(2, 1);
    X(0, 0) = 1.0f;
    X(1, 0) = std::numeric_limits<float>::quiet_NaN();

    Vector y{1.0f, 2.0f};

    LinearRegression model;
    EXPECT_THROW(model.fit(X, y, 0.01f, 10), std::invalid_argument);
}

TEST(LinearRegressionTest, RejectsInfiniteTrainingTargets) {
    Matrix X(2, 1, 1.0f);
    Vector y{1.0f, std::numeric_limits<float>::infinity()};

    LinearRegression model;
    EXPECT_THROW(model.fit(X, y, 0.01f, 10), std::invalid_argument);
}

TEST(LinearRegressionTest, RejectsNonFinitePredictionFeatures) {
    Matrix training_X(2, 1, 1.0f);
    Vector y{2.0f, 2.0f};

    LinearRegression model;
    model.fit(training_X, y, 0.1f, 10);

    Matrix prediction_X(1, 1);
    prediction_X(0, 0) = std::numeric_limits<float>::quiet_NaN();

    EXPECT_THROW(model.predict(prediction_X), std::invalid_argument);
}

// ---------- NUMERICAL ROBUSTNESS ----------

TEST(LinearRegressionTest, HandlesCancellationInGradientAccumulation) {
    const float maximum = std::numeric_limits<float>::max();

    Matrix X(4, 1);
    X(0, 0) = maximum;
    X(1, 0) = maximum;
    X(2, 0) = -maximum;
    X(3, 0) = -maximum;

    Vector y{1.0f, 1.0f, 1.0f, 1.0f};

    LinearRegression model;
    EXPECT_NO_THROW(model.fit(X, y, 0.1f, 1));

    EXPECT_FLOAT_EQ(model.coefficients()[0], 0.0f);
    EXPECT_NEAR(model.intercept(), 0.1f, 1e-6f);
}

TEST(LinearRegressionTest, ThrowsWhenWeightUpdateExceedsFloatRange) {
    Matrix X(1, 1, 1.0f);
    Vector y{std::numeric_limits<float>::max()};

    LinearRegression model;

    EXPECT_THROW(
        model.fit(X, y, 2.0f, 1),
        std::overflow_error
    );
}

TEST(LinearRegressionTest, ThrowsWhenPredictionExceedsFloatRange) {
    Matrix training_X(1, 1, 1.0f);
    Vector y{4.0f};

    LinearRegression model;
    model.fit(training_X, y, 0.5f, 1);

    Matrix prediction_X(1, 1);
    prediction_X(0, 0) = std::numeric_limits<float>::max();

    EXPECT_THROW(model.predict(prediction_X), std::overflow_error);
}

TEST(LinearRegressionTest, FailedFitLeavesModelUnfitted) {
    Matrix X(1, 1, 1.0f);
    Vector y{1.0f};

    LinearRegression model;
    model.fit(X, y, 0.1f, 10);

    Matrix invalid_X(1, 1);
    invalid_X(0, 0) = std::numeric_limits<float>::infinity();

    Vector invalid_y{1.0f};

    EXPECT_THROW(
        model.fit(invalid_X, invalid_y, 0.1f, 10),
        std::invalid_argument
    );

    EXPECT_THROW(model.predict(X), std::logic_error);
}

TEST(LinearRegressionTest, PredictionSupportsEmptyInputRows) {
    Matrix training_X(2, 1, 1.0f);
    Vector y{2.0f, 2.0f};

    LinearRegression model;
    model.fit(training_X, y, 0.1f, 10);

    Matrix empty_X(0, 1);
    const Vector predictions = model.predict(empty_X);

    EXPECT_TRUE(predictions.empty());
}
