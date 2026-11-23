
cmake_minimum_required(VERSION 3.25)

project(CppML VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(CPPML_ENABLE_CUDA "Enable CUDA backend" OFF)
option(CPPML_BUILD_BENCHMARKS "Build benchmarks" ON)

if(CPPML_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)

    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR
            "CUDA was requested but no CUDA compiler was found."
        )
    endif()

    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 86)
    endif()

    message(STATUS "CppML CUDA backend enabled")
    message(STATUS "CUDA architecture: ${CMAKE_CUDA_ARCHITECTURES}")
    message(STATUS "CUDA Toolkit: ${CUDAToolkit_VERSION}")
else()
    message(STATUS "CppML CUDA backend disabled")
endif()

add_library(cppml
    src/core/vector.cpp
    src/core/matrix.cpp
    src/math/cpu_operations.cpp
    src/math/metrics.cpp
    src/models/linear_regression.cpp
    src/models/least_squares.cpp
    src/models/svd.cpp
)

target_include_directories(cppml
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_compile_features(cppml PUBLIC cxx_std_20)

if(CPPML_ENABLE_CUDA)
    target_sources(cppml PRIVATE
        src/math/cuda_operations.cu
        src/models/cuda_linear_regression.cu
    )

    target_compile_definitions(cppml PUBLIC CPPML_HAS_CUDA=1)
    target_link_libraries(cppml PUBLIC CUDA::cudart)

    set_target_properties(cppml PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

include(CTest)

if(BUILD_TESTING)
    find_package(GTest CONFIG REQUIRED)

    add_executable(cppml_core_tests
        tests/core/test_core.cpp
        tests/math/test_operations.cpp
        tests/math/test_metrics.cpp
        tests/math/test_numerical_precision.cpp
        tests/models/test_linear_regression.cpp
        tests/models/test_linear_regression_conditioning.cpp
        tests/models/test_lr09_final_validation.cpp
        tests/models/test_backend_dispatch.cpp
        tests/models/test_least_squares.cpp
        tests/models/test_svd.cpp
        tests/models/test_solver_comparison.cpp
    )

    if(CPPML_ENABLE_CUDA)
        target_sources(cppml_core_tests PRIVATE
            tests/math/test_cuda_operations.cpp
            tests/models/test_cuda_linear_regression.cpp
        )
    endif()

    target_link_libraries(cppml_core_tests PRIVATE
        cppml
        GTest::gtest_main
    )

    include(GoogleTest)
    gtest_discover_tests(cppml_core_tests)
endif()

if(CPPML_BUILD_BENCHMARKS)
    find_package(benchmark CONFIG REQUIRED)

    add_executable(cppml_benchmarks
        benchmarks/benchmark_math.cpp
        benchmarks/benchmark_models.cpp
        benchmarks/benchmark_solvers.cpp
    )

    target_include_directories(cppml_benchmarks PRIVATE
        benchmarks
    )

    target_link_libraries(cppml_benchmarks PRIVATE
        cppml
        benchmark::benchmark_main
    )

    if(MSVC)
        target_compile_options(cppml_benchmarks PRIVATE /O2)
    endif()

    message(STATUS "CppML benchmarks enabled")
endif()
