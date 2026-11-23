
import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def make_dataset(rows: int, cols: int):
    row = np.arange(1, rows + 1, dtype=np.float64)[:, None]
    col = np.arange(1, cols + 1, dtype=np.float64)[None, :]

    X64 = (
        np.sin(row * col * 0.013)
        + np.cos(row * (col + 1.0) * 0.007)
    )

    diagonal = min(rows, cols)
    indices = np.arange(diagonal)
    X64[indices, indices] += 0.1

    true_coefficients = 0.05 * np.arange(
        1, cols + 1, dtype=np.float64
    )
    true_intercept = 0.75

    y64 = true_intercept + X64 @ true_coefficients

    X = X64.astype(np.float32)
    y = y64.astype(np.float32)

    return X, y, true_coefficients, true_intercept


def measure_fit(X, y, warmups, repetitions):
    for _ in range(warmups):
        LinearRegression().fit(X, y)

    timings = []
    model = None

    for _ in range(repetitions):
        model = LinearRegression()
        start = time.perf_counter()
        model.fit(X, y)
        timings.append(time.perf_counter() - start)

    return model, timings


def measure_predict(model, X, warmups, repetitions):
    for _ in range(warmups):
        model.predict(X)

    timings = []

    for _ in range(repetitions):
        start = time.perf_counter()
        predictions = model.predict(X)
        timings.append(time.perf_counter() - start)

    return predictions, timings


def summarize(values):
    return {
        "median_seconds": statistics.median(values),
        "mean_seconds": statistics.mean(values),
        "stdev_seconds": (
            statistics.stdev(values) if len(values) > 1 else 0.0
        ),
        "samples_seconds": values,
    }


def run_case(rows, cols, warmups, repetitions):
    X, y, true_coefficients, true_intercept = make_dataset(rows, cols)

    model, fit_times = measure_fit(X, y, warmups, repetitions)
    predictions, predict_times = measure_predict(
        model, X, warmups, repetitions
    )

    coefficient_errors = model.coef_.astype(np.float64) - true_coefficients
    intercept_error = float(model.intercept_) - true_intercept
    prediction_errors = predictions.astype(np.float64) - y.astype(np.float64)

    return {
        "rows": rows,
        "cols": cols,
        "dtype": "float32",
        "solver": "sklearn LinearRegression",
        "fit": summarize(fit_times),
        "predict": summarize(predict_times),
        "accuracy": {
            "prediction_mse": float(mean_squared_error(y, predictions)),
            "prediction_r2": float(r2_score(y, predictions)),
            "prediction_max_abs_error": float(
                np.max(np.abs(prediction_errors))
            ),
            "coefficient_rmse": float(
                np.sqrt(np.mean(coefficient_errors ** 2))
            ),
            "coefficient_max_abs_error": float(
                np.max(np.abs(coefficient_errors))
            ),
            "intercept_abs_error": float(abs(intercept_error)),
        },
        "ground_truth": {
            "intercept": true_intercept,
            "coefficients": true_coefficients.tolist(),
        },
        "fitted": {
            "intercept": float(model.intercept_),
            "coefficients": model.coef_.astype(float).tolist(),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument(
        "--output",
        type=str,
        default="results/sklearn_accuracy.json",
    )
    args = parser.parse_args()

    if args.warmups < 0:
        raise ValueError("--warmups must be non-negative")

    if args.repetitions < 2:
        raise ValueError("--repetitions must be at least 2")

    configurations = [
        (256, 4),
        (512, 8),
        (1024, 16),
        (2048, 16),
    ]

    results = [
        run_case(rows, cols, args.warmups, args.repetitions)
        for rows, cols in configurations
    ]

    output = {
        "library": "scikit-learn",
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "results": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print(f"Saved: {output_path.resolve()}")

    for result in results:
        accuracy = result["accuracy"]
        print(
            f"{result['rows']}x{result['cols']} | "
            f"fit: {result['fit']['median_seconds']:.6f}s | "
            f"predict: {result['predict']['median_seconds']:.6f}s | "
            f"MSE: {accuracy['prediction_mse']:.8g} | "
            f"coefficient RMSE: {accuracy['coefficient_rmse']:.8g} | "
            f"intercept error: {accuracy['intercept_abs_error']:.8g}"
        )


if __name__ == "__main__":
    main()
