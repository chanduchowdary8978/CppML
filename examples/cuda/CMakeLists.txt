
cmake_minimum_required(VERSION 3.25)

project(CppMLCUDAExample LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(cuda_vector_add
    vector_add.cu
)

set_target_properties(cuda_vector_add PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
