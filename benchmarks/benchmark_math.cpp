
#include "benchmark_data.hpp"

#include "cppml/math/operations.hpp"

#ifdef CPPML_HAS_CUDA
#include "cppml/math/cuda_operations.hpp"
#endif

#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>

using cppml::benchmarks::make_vector;
using cppml::benchmarks::make_dataset;

static void BM_Matvec_CPU(benchmark::State& state) {
    const auto rows = static_cast<std::size_t>(state.range(0));
    const auto cols = static_cast<std::size_t>(state.range(1));

    auto data = make_dataset(rows, cols);
    auto w = make_vector(cols);

    for (auto _ : state) {
        auto result = cppml::math::matvec(data.X, w);
        benchmark::DoNotOptimize(result.data());
        benchmark::ClobberMemory();
    }

    state.counters["rows"] = static_cast<double>(rows);
    state.counters["cols"] = static_cast<double>(cols);
    state.SetItemsProcessed(state.iterations() * rows * cols);
}

#ifdef CPPML_HAS_CUDA
static void BM_Matvec_CUDA(benchmark::State& state) {
    const auto rows = static_cast<std::size_t>(state.range(0));
    const auto cols = static_cast<std::size_t>(state.range(1));

    auto data = make_dataset(rows, cols);
    auto w = make_vector(cols);

    for (auto _ : state) {
        auto result = cppml::math::cuda::matvec(data.X, w);
        benchmark::DoNotOptimize(result.data());
        benchmark::ClobberMemory();
    }

    state.counters["rows"] = static_cast<double>(rows);
    state.counters["cols"] = static_cast<double>(cols);
    state.SetItemsProcessed(state.iterations() * rows * cols);
}
#endif

static void BM_TransposeMatvec_CPU(benchmark::State& state) {
    const auto rows = static_cast<std::size_t>(state.range(0));
    const auto cols = static_cast<std::size_t>(state.range(1));

    auto data = make_dataset(rows, cols);
    auto r = make_vector(rows);

    for (auto _ : state) {
        auto result = cppml::math::transpose_matvec(data.X, r);
        benchmark::DoNotOptimize(result.data());
        benchmark::ClobberMemory();
    }

    state.counters["rows"] = static_cast<double>(rows);
    state.counters["cols"] = static_cast<double>(cols);
    state.SetItemsProcessed(state.iterations() * rows * cols);
}

#ifdef CPPML_HAS_CUDA
static void BM_TransposeMatvec_CUDA(benchmark::State& state) {
    const auto rows = static_cast<std::size_t>(state.range(0));
    const auto cols = static_cast<std::size_t>(state.range(1));

    auto data = make_dataset(rows, cols);
    auto r = make_vector(rows);

    for (auto _ : state) {
        auto result = cppml::math::cuda::transpose_matvec(data.X, r);
        benchmark::DoNotOptimize(result.data());
        benchmark::ClobberMemory();
    }

    state.counters["rows"] = static_cast<double>(rows);
    state.counters["cols"] = static_cast<double>(cols);
    state.SetItemsProcessed(state.iterations() * rows * cols);
}
#endif

static void BM_Subtract_CPU(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    auto a = make_vector(n);
    auto b = make_vector(n, 1.0f);

    for (auto _ : state) {
        auto result = cppml::math::subtract(a, b);
        benchmark::DoNotOptimize(result.data());
        benchmark::ClobberMemory();
    }

    state.counters["elements"] = static_cast<double>(n);
    state.SetItemsProcessed(state.iterations() * n);
}

#ifdef CPPML_HAS_CUDA
static void BM_Subtract_CUDA(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    auto a = make_vector(n);
    auto b = make_vector(n, 1.0f);

    for (auto _ : state) {
        auto result = cppml::math::cuda::subtract(a, b);
        benchmark::DoNotOptimize(result.data());
        benchmark::ClobberMemory();
    }

    state.counters["elements"] = static_cast<double>(n);
    state.SetItemsProcessed(state.iterations() * n);
}
#endif

static void BM_Dot_CPU(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    auto a = make_vector(n);
    auto b = make_vector(n, 1.0f);

    for (auto _ : state) {
        const float result = cppml::math::dot(a, b);
        benchmark::DoNotOptimize(result);
    }

    state.counters["elements"] = static_cast<double>(n);
    state.SetItemsProcessed(state.iterations() * n);
}

#ifdef CPPML_HAS_CUDA
static void BM_Dot_CUDA(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    auto a = make_vector(n);
    auto b = make_vector(n, 1.0f);

    for (auto _ : state) {
        const float result = cppml::math::cuda::dot(a, b);
        benchmark::DoNotOptimize(result);
    }

    state.counters["elements"] = static_cast<double>(n);
    state.SetItemsProcessed(state.iterations() * n);
}
#endif

static void BM_Sum_CPU(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    auto a = make_vector(n);

    for (auto _ : state) {
        const float result = cppml::math::sum(a);
        benchmark::DoNotOptimize(result);
    }

    state.counters["elements"] = static_cast<double>(n);
    state.SetItemsProcessed(state.iterations() * n);
}

#ifdef CPPML_HAS_CUDA
static void BM_Sum_CUDA(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    auto a = make_vector(n);

    for (auto _ : state) {
        const float result = cppml::math::cuda::sum(a);
        benchmark::DoNotOptimize(result);
    }

    state.counters["elements"] = static_cast<double>(n);
    state.SetItemsProcessed(state.iterations() * n);
}
#endif

static void BM_Mean_CPU(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    auto a = make_vector(n);

    for (auto _ : state) {
        const float result = cppml::math::mean(a);
        benchmark::DoNotOptimize(result);
    }

    state.counters["elements"] = static_cast<double>(n);
    state.SetItemsProcessed(state.iterations() * n);
}

#ifdef CPPML_HAS_CUDA
static void BM_Mean_CUDA(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    auto a = make_vector(n);

    for (auto _ : state) {
        const float result = cppml::math::cuda::mean(a);
        benchmark::DoNotOptimize(result);
    }

    state.counters["elements"] = static_cast<double>(n);
    state.SetItemsProcessed(state.iterations() * n);
}
#endif

static void BM_AddScalar_CPU(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    auto a = make_vector(n);

    for (auto _ : state) {
        auto result = cppml::math::add_scalar(a, 2.0f);
        benchmark::DoNotOptimize(result.data());
        benchmark::ClobberMemory();
    }

    state.counters["elements"] = static_cast<double>(n);
    state.SetItemsProcessed(state.iterations() * n);
}

#ifdef CPPML_HAS_CUDA
static void BM_AddScalar_CUDA(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    auto a = make_vector(n);

    for (auto _ : state) {
        auto result = cppml::math::cuda::add_scalar(a, 2.0f);
        benchmark::DoNotOptimize(result.data());
        benchmark::ClobberMemory();
    }

    state.counters["elements"] = static_cast<double>(n);
    state.SetItemsProcessed(state.iterations() * n);
}
#endif

#define REGISTER_MATRIX_BENCHMARK(fn) \
    BENCHMARK(fn)->Args({256, 8})->Args({1024, 16})->Args({4096, 32})->Args({16384, 64})

#define REGISTER_VECTOR_BENCHMARK(fn) \
    BENCHMARK(fn)->Arg(1024)->Arg(16384)->Arg(65536)->Arg(262144)

REGISTER_MATRIX_BENCHMARK(BM_Matvec_CPU);
REGISTER_MATRIX_BENCHMARK(BM_TransposeMatvec_CPU);

REGISTER_VECTOR_BENCHMARK(BM_Subtract_CPU);
REGISTER_VECTOR_BENCHMARK(BM_Dot_CPU);
REGISTER_VECTOR_BENCHMARK(BM_Sum_CPU);
REGISTER_VECTOR_BENCHMARK(BM_Mean_CPU);
REGISTER_VECTOR_BENCHMARK(BM_AddScalar_CPU);

#ifdef CPPML_HAS_CUDA
REGISTER_MATRIX_BENCHMARK(BM_Matvec_CUDA);
REGISTER_MATRIX_BENCHMARK(BM_TransposeMatvec_CUDA);

REGISTER_VECTOR_BENCHMARK(BM_Subtract_CUDA);
REGISTER_VECTOR_BENCHMARK(BM_Dot_CUDA);
REGISTER_VECTOR_BENCHMARK(BM_Sum_CUDA);
REGISTER_VECTOR_BENCHMARK(BM_Mean_CUDA);
REGISTER_VECTOR_BENCHMARK(BM_AddScalar_CUDA);
#endif
