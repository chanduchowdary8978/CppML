
# LR-10: Benchmarking and Performance Evaluation

## Objective

Measure the performance of CppML's CPU and CUDA implementations and
compare the CPU Linear Regression and Least Squares implementations
against relevant scikit-learn baselines.

The benchmark results evaluate:

- CPU versus CUDA performance.
- AUTO backend dispatch behavior.
- CUDA gradient descent optimization.
- QR and SVD solver performance.
- Numerical accuracy against known ground truth.
- Numerical accuracy relative to scikit-learn.

The results are specific to the tested hardware and software
environment. They should not be interpreted as universal performance
guarantees.

## Hardware and software

| Component | Configuration |
|---|---|
| CPU | Intel Core i5-13450HX |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU, 6 GB |
| CUDA Toolkit | 13.3.33 |
| NVIDIA Driver | 610.60 |
| Compiler | MSVC 19.51.36260.0 |
| MSVC Toolset | 14.51.36231 |
| Build system | CMake 4.1.0 + Ninja 1.13.1 |
| C++ standard | C++20 |
| Python | 3.12.10 |
| NumPy | 2.5.4 |
| scikit-learn | 1.6.1 |
| Google Benchmark | 1.10.0 |
| Nsight Compute | 2026.2.0.0 |
| Nsight Systems | 2026.1.3.243 |

## Benchmark categories

### 1. Mathematical operations

Measure the public CPU and CUDA APIs for:

- Matrix-vector multiplication
- Transpose matrix-vector multiplication
- Vector subtraction
- Dot product
- Sum
- Mean
- Scalar addition

The CUDA public APIs include:

- Host-to-device transfers
- Device allocations
- Kernel execution
- Result transfer back to the host

Therefore, these timings represent end-to-end API cost, not
isolated CUDA kernel execution time.

The public API measurements are useful for evaluating actual
application-level overhead, especially for small workloads.

### 2. Linear Regression

Measure:

- CPU training
- CUDA training
- AUTO training
- CPU prediction
- CUDA prediction

Training measurements include model construction and fitting.

Prediction measurements exclude model fitting.

The CUDA implementation keeps training data, weights, predictions,
residuals, and gradients on the GPU during the training loop.
Final model parameters are transferred back to the host.

Dataset dimensions and epoch counts are recorded for each benchmark.

### 3. Least Squares solvers

Measure:

- Householder QR
- One-sided Jacobi SVD

Both implementations execute on the CPU.

QR uses Householder transformations and column equilibration.

SVD uses one-sided Jacobi rotations and computes a pseudoinverse
solution, including minimum-norm solutions for rank-deficient
or underdetermined systems.

The benchmark does not imply GPU execution for these solvers.

### 4. scikit-learn baseline

Measure:

- scikit-learn LinearRegression fitting
- Prediction
- Mean squared error
- Intercept
- Coefficients

The Python dataset generator follows the same mathematical
construction as the C++ benchmark dataset generator.

The target values are calculated using double-precision feature
values before the feature matrix is converted to float32.

scikit-learn LinearRegression is a least-squares estimator.
It is not equivalent to CppML's gradient descent training algorithm.

Therefore, comparisons must distinguish between:

- Gradient descent performance.
- Least Squares solver performance.
- Prediction accuracy.
- End-to-end training and prediction time.

## Measurement methodology

1. Build in Release mode.
2. Run correctness tests before benchmarking.
3. Use warm-up iterations.
4. Use multiple repetitions.
5. Report median and mean timings.
6. Keep CPU and GPU workloads identical where applicable.
7. Avoid running unrelated GPU workloads simultaneously.
8. Record hardware and software configuration.
9. Do not compare Debug timings with Release timings.
10. Separate setup, allocation, transfer, and computation costs
    whenever the benchmark is intended to measure a specific layer.

## CUDA timing considerations

CUDA execution is asynchronous in many contexts.

Use wall-clock timing for public API measurements.

The public CUDA API measurements include transfer and allocation
overheads. These costs are relevant to real application performance.

Small workloads may run faster on CPU because CUDA launch,
allocation, and transfer overhead can dominate computation.

Kernel-only timing and end-to-end API timing answer different
performance questions and should not be mixed.

## CUDA gradient descent optimization

### Initial implementation

The initial CUDA gradient implementation used separate weight
gradient and bias gradient kernels.

Profiling showed that these kernels dominated the measured GPU
execution time.

The Nsight Systems profile showed:

- Weight gradient kernel: approximately 55.7% of kernel time.
- Bias gradient kernel: approximately 42.2% of kernel time.
- 1,540 instances of each gradient kernel in the profiled run.
- Six kernel launches per training epoch.
- Nine allocations and nine frees per fit.
- Five memory-copy operations per fit.

The profiled workload contained 77 fits with 20 epochs each.

### Optimized implementation

The CUDA implementation was optimized using:

- A tiled partial-gradient kernel.
- Shared-memory reduction across rows.
- A final gradient reduction kernel.
- A fused parameter update kernel.

The partial-gradient kernel uses:

- FEATURE_TILE = 32
- ROW_TILE = 8

The optimization reduced the number of kernel launches from
six to five per epoch.

Double-precision accumulation and numerical overflow checks
were preserved.

### Measured result

For the 8192 x 32 dataset with 20 epochs:

| Implementation | Training time |
|---|---:|
| Before optimization | Approximately 19.7 ms |
| After optimization | Approximately 3.8 ms |

Measured speedup: approximately 5x.

The optimized implementation passed all 191 CUDA-enabled tests.

This result applies to the tested workload and hardware.

## CPU versus CUDA versus AUTO

The following results were obtained using five benchmark repetitions.

| Rows | Features | Epochs | Estimated operations | CPU (ms) | CUDA (ms) | AUTO (ms) | CUDA speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 8 | 20 | 40,960 | 0.118360 | 0.962871 | 0.120612 | 0.123x |
| 1,024 | 16 | 20 | 327,680 | 1.059425 | 1.150548 | 1.065593 | 0.921x |
| 2,048 | 16 | 50 | 1,638,400 | 5.148111 | 2.845660 | 2.886588 | 1.809x |
| 4,096 | 32 | 20 | 2,621,440 | 6.845226 | 2.214224 | 2.195842 | 3.091x |
| 8,192 | 32 | 20 | 5,242,880 | 13.795080 | 3.806125 | 3.790159 | 3.624x |

CUDA speedup is calculated as:

    CPU training time / CUDA training time

Interpretation:

- CPU is faster for the two smaller workloads.
- CUDA is faster for the three larger workloads.
- CUDA speedup increases with workload size in these measurements.
- AUTO selects CPU for the first two workloads and CUDA for
  the remaining three.
- AUTO remains close to the faster measured backend for all
  five tested workloads.

### Current AUTO dispatch threshold

The current provisional AUTO threshold is:

    1,000,000 estimated operations

The threshold selects the backend using an estimated workload size.

It is not a universal CPU/GPU crossover point.

Actual crossover behavior depends on:

- CPU architecture and memory performance.
- GPU architecture.
- Dataset dimensions.
- Epoch count.
- Kernel launch overhead.
- Allocation and transfer costs.
- Data residency and future batching behavior.

The threshold should be reevaluated when the implementation,
hardware, or workload characteristics change.

The benchmark analyzer reports a candidate threshold.
It does not modify the CppML implementation.

## Least Squares solver performance

The following are median timings from the latest solver accuracy
benchmark run.

| Dataset | QR median | SVD median |
|---|---:|---:|
| 256 x 4 | 9.188 us | 18.451 us |
| 512 x 8 | 42.103 us | 140.455 us |
| 1024 x 16 | 307.939 us | 1.734 ms |
| 2048 x 16 | 625.632 us | 3.814 ms |

Observations:

- QR is faster than SVD for all four tested datasets.
- The performance difference increases with dataset size.
- SVD provides pseudoinverse behavior useful for rank-deficient
  and underdetermined systems.
- QR is a strong choice for the tested full-rank least-squares
  workloads.

These timings are specific to the current implementations and
benchmark environment.

## Numerical accuracy validation

Both CppML solvers were evaluated against known ground-truth
coefficients and intercept values.

The benchmark used four synthetic datasets:

- 256 x 4
- 512 x 8
- 1024 x 16
- 2048 x 16

Ground-truth parameters:

    intercept = 0.75

    coefficient[j] = 0.05 * (j + 1)

The benchmark recorded:

- Prediction MSE.
- Coefficient RMSE.
- Maximum prediction absolute error.
- Maximum coefficient absolute error.
- Intercept absolute error.

### CppML QR and SVD results

Both solvers produced the following measured accuracy values
to the displayed precision.

| Dataset | Prediction MSE | Coefficient RMSE | Max prediction error | Max coefficient error | Intercept error |
|---|---:|---:|---:|---:|---:|
| 256 x 4 | 5.88e-16 | 4.08e-9 | 9.13e-8 | 5.96e-9 | 0 |
| 512 x 8 | 1.47e-15 | 5.70e-9 | 1.60e-7 | 1.19e-8 | 0 |
| 1024 x 16 | 7.72e-15 | 1.11e-8 | 4.28e-7 | 2.38e-8 | 0 |
| 2048 x 16 | 7.06e-15 | 1.11e-8 | 4.28e-7 | 2.38e-8 | 0 |

The measured errors are very small for these synthetic,
well-behaved datasets.

### scikit-learn accuracy results

| Dataset | Prediction MSE | Coefficient RMSE | Max prediction error | Max coefficient error | Intercept error |
|---|---:|---:|---:|---:|---:|
| 256 x 4 | 2.50e-14 | 7.07e-8 | 3.58e-7 | 8.79e-8 | 0 |
| 512 x 8 | 1.11e-13 | 1.11e-7 | 7.15e-7 | 2.26e-7 | 5.96e-8 |
| 1024 x 16 | 1.85e-12 | 3.33e-7 | 3.81e-6 | 7.51e-7 | 0 |
| 2048 x 16 | 2.15e-12 | 3.74e-7 | 4.26e-6 | 7.75e-7 | 0 |

Interpretation:

- Both implementations recover the ground-truth parameters
  accurately on these datasets.
- CppML produced lower measured residual MSE in this run.
- CppML uses double-precision internal calculations in its
  solvers and converts final parameters to float.
- scikit-learn follows its own numerical implementation and
  library execution path.
- Small differences in floating-point arithmetic and summation
  order can affect measured errors.

These results do not establish that CppML is universally
more accurate than scikit-learn.

A broader accuracy evaluation must include ill-conditioned,
rank-deficient, noisy, and differently scaled datasets.

## scikit-learn timing baseline

The latest scikit-learn benchmark used five warm-up iterations
and ten repetitions.

| Dataset | Fit median | Predict median |
|---|---:|---:|
| 256 x 4 | 0.24535 ms | 0.03610 ms |
| 512 x 8 | 0.29260 ms | 0.03905 ms |
| 1024 x 16 | 0.58285 ms | 0.04260 ms |
| 2048 x 16 | 0.76445 ms | 0.04705 ms |

scikit-learn LinearRegression uses a least-squares approach.
CppML's gradient descent model is a different algorithm.

Therefore, scikit-learn fit timings should not be presented
as a direct algorithmically equivalent comparison with
CppML gradient descent.

QR and SVD comparisons are more relevant because they are
also least-squares approaches.

Even then, implementation details, numerical libraries,
preprocessing, and timing methodology differ.

## Interpretation limitations

- A faster training time does not automatically mean better
  numerical accuracy.
- Compare prediction error and parameter recovery before
  drawing conclusions about equivalent model quality.
- CUDA performance depends on workload size and transfer overhead.
- The AUTO threshold is hardware- and workload-dependent.
- The measured CUDA speedup is specific to the tested GPU,
  CPU, compiler, and workload.
- QR and SVD timings are CPU-only.
- scikit-learn uses its own numerical libraries and optimized
  implementations.
- scikit-learn LinearRegression is not the same algorithm
  as CppML gradient descent.
- The synthetic benchmark datasets are not representative
  of every real-world dataset.
- Good results on well-conditioned synthetic data do not
  guarantee stability on ill-conditioned data.
- Benchmark results are specific to the tested hardware,
  compiler, drivers, and software versions.
- Sanitizer validation remains incomplete because the
  instrumented application terminated before the first
  instrumented API call. Ordinary functional test success
  must not be interpreted as a successful sanitizer run.

## Functional test validation

The latest recorded test results:

| Build configuration | Tests passed | Tests failed |
|---|---:|---:|
| CPU-only | 141 | 0 |
| CUDA-enabled | 191 | 0 |

These results include the numerical validation and solver
comparison tests.

## Result files

Store raw benchmark outputs under `results/`.

Current benchmark artifacts include:

- `cppml_cuda_benchmarks.json`
- `sklearn_baseline.json`
- `dispatch_analysis.csv`
- `cppml_optimized_model_benchmarks.json`
- `cppml_final_dispatch_benchmarks.json`
- `final_dispatch_analysis.csv`
- `sklearn_final_benchmarks.json`
- `cppml_solver_benchmarks.json`
- `cppml_solver_accuracy_benchmarks.json`
- `sklearn_accuracy.json`

Do not commit generated benchmark outputs unless they are
intentionally included as documented experimental results.

## Conclusion

The benchmark results demonstrate that:

1. CPU execution is preferable for the tested small workloads.
2. CUDA becomes advantageous for the tested larger workloads.
3. The optimized CUDA gradient implementation achieves
   approximately 5x speedup on the profiled 8192 x 32 workload.
4. The provisional 1,000,000-operation AUTO threshold performs
   well across the five tested workloads.
5. QR is faster than Jacobi SVD for the four tested datasets.
6. Both CppML solvers recover the known synthetic parameters
   with very small numerical errors.

These results provide an initial performance baseline for CppML.
Further work should focus on broader numerical stress testing,
additional hardware, and more representative real-world workloads.
