
#include "benchmark_data.hpp"

#include "cppml/models/linear_regression.hpp"

#include <benchmark/benchmark.h>

#include <cstddef>

using cppml::benchmarks::make_dataset;
using cppml::models::ExecutionBackend;
using cppml::models::LinearRegression;

static void BM_LinearRegression_CPU(benchmark::State& state) {
    const auto rows = static_cast<std::size_t>(state.range(0));
    const auto cols = static_cast<std::size_t>(state.range(1));
    const auto epochs = static_cast<std::size_t>(state.range(2));

    auto data = make_dataset(rows, cols);

    for (auto _ : state) {
        LinearRegression model(ExecutionBackend::CPU);
        model.fit(data.X, data.y, 0.01f, epochs);

        benchmark::DoNotOptimize(model.coefficients().data());
        benchmark::DoNotOptimize(model.intercept());
    }

    state.counters["rows"] = static_cast<double>(rows);
    state.counters["cols"] = static_cast<double>(cols);
    state.counters["epochs"] = static_cast<double>(epochs);
    state.counters["work"] =
        static_cast<double>(rows) *
        static_cast<double>(cols) *
        static_cast<double>(epochs);

    state.SetItemsProcessed(
        state.iterations() * rows * cols * epochs
    );
}

#ifdef CPPML_HAS_CUDA
static void BM_LinearRegression_CUDA(benchmark::State& state) {
    const auto rows = static_cast<std::size_t>(state.range(0));
    const auto cols = static_cast<std::size_t>(state.range(1));
    const auto epochs = static_cast<std::size_t>(state.range(2));

    auto data = make_dataset(rows, cols);

    for (auto _ : state) {
        LinearRegression model(ExecutionBackend::CUDA);
        model.fit(data.X, data.y, 0.01f, epochs);

        benchmark::DoNotOptimize(model.coefficients().data());
        benchmark::DoNotOptimize(model.intercept());
    }

    state.counters["rows"] = static_cast<double>(rows);
    state.counters["cols"] = static_cast<double>(cols);
    state.counters["epochs"] = static_cast<double>(epochs);
    state.counters["work"] =
        static_cast<double>(rows) *
        static_cast<double>(cols) *
        static_cast<double>(epochs);

    state.SetItemsProcessed(
        state.iterations() * rows * cols * epochs
    );
}

static void BM_LinearRegression_AUTO(benchmark::State& state) {
    const auto rows = static_cast<std::size_t>(state.range(0));
    const auto cols = static_cast<std::size_t>(state.range(1));
    const auto epochs = static_cast<std::size_t>(state.range(2));

    auto data = make_dataset(rows, cols);

    for (auto _ : state) {
        LinearRegression model(ExecutionBackend::AUTO);
        model.fit(data.X, data.y, 0.01f, epochs);

        benchmark::DoNotOptimize(model.coefficients().data());
        benchmark::DoNotOptimize(model.intercept());

        state.counters["cuda_selected"] =
            model.active_backend() == ExecutionBackend::CUDA ? 1.0 : 0.0;
    }

    state.counters["rows"] = static_cast<double>(rows);
    state.counters["cols"] = static_cast<double>(cols);
    state.counters["epochs"] = static_cast<double>(epochs);
    state.counters["work"] =
        static_cast<double>(rows) *
        static_cast<double>(cols) *
        static_cast<double>(epochs);

    state.SetItemsProcessed(
        state.iterations() * rows * cols * epochs
    );
}
#endif

static void BM_LinearRegression_Predict_CPU(benchmark::State& state) {
    const auto rows = static_cast<std::size_t>(state.range(0));
    const auto cols = static_cast<std::size_t>(state.range(1));

    auto train = make_dataset(1024, cols);
    auto test = make_dataset(rows, cols);

    LinearRegression model(ExecutionBackend::CPU);
    model.fit(train.X, train.y, 0.01f, 100);

    for (auto _ : state) {
        auto predictions = model.predict(test.X);
        benchmark::DoNotOptimize(predictions.data());
    }

    state.counters["rows"] = static_cast<double>(rows);
    state.counters["cols"] = static_cast<double>(cols);
    state.SetItemsProcessed(state.iterations() * rows * cols);
}

#ifdef CPPML_HAS_CUDA
static void BM_LinearRegression_Predict_CUDA(benchmark::State& state) {
    const auto rows = static_cast<std::size_t>(state.range(0));
    const auto cols = static_cast<std::size_t>(state.range(1));

    auto train = make_dataset(1024, cols);
    auto test = make_dataset(rows, cols);

    LinearRegression model(ExecutionBackend::CUDA);
    model.fit(train.X, train.y, 0.01f, 100);

    for (auto _ : state) {
        auto predictions = model.predict(test.X);
        benchmark::DoNotOptimize(predictions.data());
    }

    state.counters["rows"] = static_cast<double>(rows);
    state.counters["cols"] = static_cast<double>(cols);
    state.SetItemsProcessed(state.iterations() * rows * cols);
}
#endif

#define REGISTER_TRAINING_BENCHMARK(fn) \
    BENCHMARK(fn) \
        ->Args({256, 8, 20}) \
        ->Args({1024, 16, 20}) \
        ->Args({2048, 16, 50}) \
        ->Args({4096, 32, 20}) \
        ->Args({8192, 32, 20})

REGISTER_TRAINING_BENCHMARK(BM_LinearRegression_CPU);

#ifdef CPPML_HAS_CUDA
REGISTER_TRAINING_BENCHMARK(BM_LinearRegression_CUDA);
REGISTER_TRAINING_BENCHMARK(BM_LinearRegression_AUTO);
#endif

BENCHMARK(BM_LinearRegression_Predict_CPU)
    ->Args({256, 8})
    ->Args({4096, 32})
    ->Args({16384, 32});

#ifdef CPPML_HAS_CUDA
BENCHMARK(BM_LinearRegression_Predict_CUDA)
    ->Args({256, 8})
    ->Args({4096, 32})
    ->Args({16384, 32});
#endif
