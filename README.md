# CppML

**A C++20 Machine Learning and Numerical Computing Library with CPU and CUDA backends.**

CppML is a lightweight machine learning library focused on implementing numerical algorithms, optimization, and high-performance computing using modern C++.

## Features

| Component | Implementation |
|---|---|
| Matrix & Vector | Contiguous float32 data structures |
| Mathematical Operations | MatVec, transpose MatVec, dot product, sum, mean |
| Linear Regression | Batch Gradient Descent |
| Least Squares | Householder QR decomposition |
| SVD | One-sided Jacobi pseudoinverse |
| Evaluation Metrics | MSE, RMSE, MAE, R² |
| Execution Backends | CPU, CUDA, AUTO |
| Python Bindings | pybind11 |

## Architecture

```text
                 CppML
                   |
          Public C++20 API
                   |
        Numerical Operations
                   |
        +----------+----------+
        |          |          |
       CPU        CUDA       Models
        |          |          |
        +----------+----------+
                   |
             Python API
                pybind11
```

## Performance Benchmarks

### 1. CPU vs CUDA vs AUTO

Linear Regression training benchmark using Batch Gradient Descent.

| Samples | Features | Epochs | CPU (ms) | CUDA (ms) | AUTO (ms) | CUDA Speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 8 | 20 | 0.120 | 1.034 | 0.135 | 0.12× |
| 1,024 | 16 | 20 | 1.056 | 1.193 | 1.051 | 0.89× |
| 2,048 | 16 | 50 | 5.188 | 2.831 | 2.889 | 1.83× |
| 4,096 | 32 | 20 | 7.054 | 2.328 | 2.394 | 3.03× |
| 8,192 | 32 | 20 | 14.284 | 4.394 | 3.834 | 3.25× |

**Speedup = CPU execution time / CUDA execution time**

- CUDA is slower for small workloads because GPU launch and transfer overhead can dominate.
- CUDA becomes beneficial as workload size increases.
- AUTO selects CPU for smaller workloads and CUDA for larger workloads.
- The backend selection threshold is workload-dependent.

### 2. Matrix-Vector Multiplication

| Matrix Dimensions | CPU (ms) | CUDA (ms) | CUDA Speedup |
|---|---:|---:|---:|
| 16,384 × 64 | 2.095 | 0.919 | 2.28× |

### 3. CppML vs Scikit-learn

Compared CppML's Householder QR least-squares solver against `sklearn.linear_model.LinearRegression`.

| Samples | Features | CppML (ms) | Scikit-learn (ms) | Speedup |
|---:|---:|---:|---:|---:|
| 256 | 8 | 0.0204 | 0.2787 | **13.63×** |
| 1,024 | 16 | 0.3040 | 0.4256 | **1.40×** |
| 2,048 | 16 | 0.5897 | 0.5702 | 0.97× |
| 4,096 | 32 | 3.9377 | 1.2787 | 0.32× |
| 8,192 | 32 | 10.0497 | 3.4047 | 0.34× |

**Accuracy comparison**

| Samples | CppML R² | Scikit-learn R² |
|---:|---:|---:|
| 256 | 0.99999346 | 0.99999344 |
| 1,024 | 0.99999408 | 0.99999410 |
| 2,048 | 0.99999117 | 0.99999118 |
| 4,096 | 0.99999640 | 0.99999642 |
| 8,192 | 0.99999519 | 0.99999517 |

Maximum coefficient difference: approximately `0.000002`.

**Interpretation:** CppML performs well on the smaller workloads tested, while Scikit-learn performs better on larger workloads. Both achieve nearly identical numerical accuracy.

### Benchmark Methodology

**CPU vs CUDA**
- Google Benchmark executables.
- CPU, CUDA and AUTO benchmark cases measured using the same CUDA-enabled executable.
- Speedup = CPU time / CUDA time.
- Results are execution times in milliseconds.

**CppML vs Scikit-learn**
- 2 warm-up runs and 10 measured runs.
- Median execution time reported.
- Single-threaded numerical libraries using `threadpoolctl`.
- Identical generated datasets and regression targets.
- Dataset generation and CppML Matrix/Vector conversion excluded from training time.
- CppML timing measures `solve_qr()` execution.
- Scikit-learn timing measures estimator creation and `fit()` execution.

These are initial benchmark results from one machine. They are not universal performance guarantees.

## Benchmark Hardware

| Component | Specification |
|---|---|
| CPU | Intel Core i5-13450HX |
| CPU Threads | 16 logical processors |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| GPU Memory | 6 GB |
| CUDA | 13.3 |
| Compiler | MSVC |
| Language | C++20 |

## Python Quick Start

```python
import cppml

X = cppml.Matrix([
    [1.0],
    [2.0],
    [3.0],
    [4.0],
    [5.0]
])

y = cppml.Vector([3.0, 5.0, 7.0, 9.0, 11.0])

model = cppml.LinearRegression(
    cppml.ExecutionBackend.CPU
)

model.fit(X, y, 0.01, 2000)

predictions = model.predict(X)

print(predictions)
```

### Least Squares

```python
result = cppml.LeastSquares.solve_qr(X, y)

print(result.coefficients)
print(result.intercept)
```

### Evaluation Metrics

```python
print(cppml.Metrics.mse(y, predictions))
print(cppml.Metrics.rmse(y, predictions))
print(cppml.Metrics.mae(y, predictions))
print(cppml.Metrics.r2(y, predictions))
```

## Project Structure

```text
CppML/
├── include/
│   └── cppml/
├── src/
│   ├── math/
│   ├── models/
│   ├── solvers/
│   └── cuda/
├── python/
│   ├── cppml/
│   └── bindings.cpp
├── benchmarks/
├── tests/
├── CMakeLists.txt
└── pyproject.toml
```

## Current Limitations

- QR performance on larger matrices needs optimization.
- CUDA acceleration is workload-dependent.
- Python NumPy/Pandas interoperability is not yet implemented.
- Numerical algorithms and solver coverage are still expanding.

## Roadmap

- Optimize QR and matrix operations.
- Improve CPU cache utilization and memory access.
- Expand CUDA-accelerated algorithms.
- Add NumPy interoperability.
- Expand machine learning algorithms.
- Improve benchmark coverage and numerical validation.

## License

MIT License