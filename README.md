

\# CppML



A C++20 machine learning library focused on understanding and implementing

core ML algorithms, numerical methods, CPU execution, and CUDA acceleration.



CppML currently focuses on Linear Regression and Least Squares, with CPU

and CUDA implementations for supported operations.



\## Features



\### Core data structures

\- Row-major `Matrix` and `Vector` containers.

\- Checked element access and dimension validation.

\- Contiguous `float32` storage.



\### Mathematical operations

\- Matrix-vector multiplication.

\- Transpose matrix-vector multiplication.

\- Vector subtraction.

\- Dot product.

\- Sum and mean.

\- Scalar addition.

\- Double-precision intermediate accumulation and numerical range checks.



\### Linear Regression

\- Batch Gradient Descent.

\- CPU and CUDA implementations.

\- CPU, CUDA, and AUTO execution backend selection.

\- Prediction and model parameter access.

\- Input validation and numerical overflow checks.



\### Least Squares

\- Householder QR decomposition with column equilibration.

\- One-sided Jacobi SVD.

\- Pseudoinverse solution for rank-deficient and underdetermined systems.

\- Numerical validation against known synthetic datasets.



\### CUDA

\- CUDA implementations of the seven mathematical operations.

\- CUDA Gradient Descent training.

\- GPU-resident training data and intermediate values during training.

\- Tiled partial-gradient computation, shared-memory reduction, and fused

&#x20; parameter updates.

\- Automatic CPU/CUDA dispatch for Linear Regression.



\### Evaluation and benchmarks

\- MSE, MAE, RMSE, and R² metrics.

\- Google Benchmark-based C++ benchmarks.

\- scikit-learn baseline scripts.

\- CPU/CUDA/AUTO dispatch analysis.

\- Numerical accuracy measurements against ground truth.



\## Project structure



```text

CppML/

├── benchmarks/

│   ├── benchmark\_data.hpp

│   ├── benchmark\_math.cpp

│   ├── benchmark\_models.cpp

│   └── benchmark\_solvers.cpp

├── docs/

│   └── LR10\_BENCHMARKING.md

├── examples/

│   └── cuda/

│       ├── CMakeLists.txt

│       └── vector\_add.cu

├── include/

│   └── cppml/

│       ├── core/

│       ├── math/

│       └── models/

├── python/

│   ├── analyze\_benchmarks.py

│   └── benchmark\_sklearn.py

├── scripts/

│   ├── build-cuda.ps1

│   └── enable-cuda.ps1

├── src/

│   ├── core/

│   ├── math/

│   └── models/

├── tests/

│   ├── core/

│   ├── math/

│   └── models/

├── CMakeLists.txt

├── CMakePresets.json

├── vcpkg.json

├── .gitignore

└── LICENSE

```



\## Requirements



The project has been validated on Windows with:



\- C++20 compiler.

\- CMake 3.25 or newer.

\- Ninja.

\- vcpkg.

\- GoogleTest.

\- Google Benchmark.



For CUDA builds, additionally install:



\- NVIDIA GPU and a compatible NVIDIA driver.

\- CUDA Toolkit with `nvcc`.

\- Visual Studio C++ build tools compatible with the installed CUDA Toolkit.



The tested configuration is documented in

\[`docs/LR10\_BENCHMARKING.md`](docs/LR10\_BENCHMARKING.md).



\## Dependencies



Dependencies are managed through the vcpkg manifest:



\- GoogleTest — unit testing.

\- Google Benchmark — performance benchmarking.



The core implementation does not depend on Eigen, fmt, PyTorch, or

scikit-learn.



Python benchmark scripts use NumPy and scikit-learn as external baselines.

They are not dependencies of the C++ library.



\## Build on Windows



Set `VCPKG\_ROOT` to your vcpkg installation directory before using the

CMake presets.



For example, in PowerShell:



```powershell

$env:VCPKG\_ROOT = "D:\\DevTools\\vcpkg"

```



\### CPU Release



Configure:



```powershell

cmake --preset windows-cpu-release

```



Build:



```powershell

cmake --build --preset cpu-release

```



Run tests:



```powershell

ctest --preset cpu-release

```



\### CPU Debug



Configure:



```powershell

cmake --preset windows-cpu-debug

```



Build:



```powershell

cmake --build --preset cpu-debug

```



Run tests:



```powershell

ctest --preset cpu-debug

```



\### CUDA Release



The CUDA preset uses architecture `86`, suitable for the NVIDIA RTX 3050

used during project validation.



Configure:



```powershell

cmake --preset windows-cuda-release

```



Build:



```powershell

cmake --build --preset cuda-release

```



Run tests:



```powershell

ctest --preset cuda-release

```



CUDA builds require the CUDA Toolkit and a compatible host compiler.

On Windows, initialize the Visual Studio developer environment before

running CMake if the compiler or Windows SDK is not already available

in the shell environment.



\## Execution backends



Linear Regression supports three backend selections:



| Backend | Behavior |

|---|---|

| `CPU` | Executes training and prediction on the CPU. |

| `CUDA` | Executes training and prediction on the CUDA backend. |

| `AUTO` | Selects CPU or CUDA using the current workload threshold. |



The current AUTO dispatch threshold is provisional and set to

1,000,000 estimated operations.



This threshold is based on measurements from one CPU/GPU configuration.

It is not a universal CPU/GPU crossover point.



\## Basic API example



```cpp

\#include <cppml/core/matrix.hpp>

\#include <cppml/core/vector.hpp>

\#include <cppml/models/execution\_backend.hpp>

\#include <cppml/models/linear\_regression.hpp>



int main() {

&#x20;   cppml::Matrix X(4, 1);

&#x20;   X(0, 0) = 1.0f;

&#x20;   X(1, 0) = 2.0f;

&#x20;   X(2, 0) = 3.0f;

&#x20;   X(3, 0) = 4.0f;



&#x20;   cppml::Vector y{3.0f, 5.0f, 7.0f, 9.0f};



&#x20;   cppml::models::LinearRegression model(

&#x20;       cppml::models::ExecutionBackend::CPU

&#x20;   );



&#x20;   model.fit(X, y, 0.01f, 1000);



&#x20;   const cppml::Vector predictions = model.predict(X);



&#x20;   return predictions.size() == y.size() ? 0 : 1;

}

```



\## Validation status



Latest clean Release validation:



| Configuration | Tests passed | Tests failed |

|---|---:|---:|

| CPU-only | 141 | 0 |

| CUDA-enabled | 191 | 0 |



A standalone CUDA vector-add example was also built and executed

successfully on an NVIDIA GeForce RTX 3050 Laptop GPU.



These are functional test results. Compute Sanitizer validation is

still incomplete because the instrumented application terminated

before its first instrumented API call.



\## Benchmark results



Benchmark methodology, hardware details, numerical accuracy results,

CUDA optimization measurements, and limitations are documented in:



\[`docs/LR10\_BENCHMARKING.md`](docs/LR10\_BENCHMARKING.md)



Benchmark outputs are generated under `results/` and are excluded from

Git by default.



\## Current scope and limitations



\- The initial focus is Linear Regression, not a general-purpose tensor

&#x20; framework.

\- Matrix and Vector storage currently use `float32`.

\- QR and SVD solvers execute on the CPU, including in CUDA-enabled builds.

\- AUTO dispatch is a provisional heuristic, not a runtime autotuner.

\- Benchmark results are specific to the tested hardware and workloads.

\- Numerical accuracy on synthetic well-conditioned datasets does not

&#x20; guarantee stability on arbitrary real-world data.



\## License



CppML is distributed under the MIT License. See \[LICENSE](LICENSE).
