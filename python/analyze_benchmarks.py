import argparse
import csv
import json
import re
from pathlib import Path

PATTERN = re.compile(
    r"BM_LinearRegression_(CPU|CUDA|AUTO)/(\d+)/(\d+)/(\d+)_median"
)


def load_benchmarks(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    records = {}

    for entry in data.get("benchmarks", []):
        match = PATTERN.search(entry.get("name", ""))
        if not match:
            continue

        backend, rows, cols, epochs = match.groups()
        timing = entry.get("real_time")

        if timing is None:
            continue

        key = (int(rows), int(cols), int(epochs))
        records.setdefault(key, {})[backend] = float(timing)

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="results/dispatch_analysis.csv")
    parser.add_argument("--current-threshold", type=int, default=5_000_000)
    args = parser.parse_args()

    records = load_benchmarks(args.input)

    if not records:
        raise SystemExit("No valid median benchmark entries found.")

    output_rows = []

    for (rows, cols, epochs), timings in sorted(records.items()):
        cpu = timings.get("CPU")
        cuda = timings.get("CUDA")
        auto = timings.get("AUTO")

        operations = rows * cols * epochs

        speedup = cpu / cuda if cpu is not None and cuda else None
        cuda_faster = cuda < cpu if cuda is not None and cpu is not None else None

        output_rows.append({
            "rows": rows,
            "cols": cols,
            "epochs": epochs,
            "estimated_operations": operations,
            "cpu_time_ns": cpu,
            "cuda_time_ns": cuda,
            "auto_time_ns": auto,
            "cuda_speedup_over_cpu": speedup,
            "cuda_faster": cuda_faster,
        })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=output_rows[0].keys())
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Saved: {output.resolve()}")

    comparable = [
        row for row in output_rows
        if row["cpu_time_ns"] is not None
        and row["cuda_time_ns"] is not None
    ]

    wins = sum(row["cuda_faster"] for row in comparable)
    print(f"Configurations compared: {len(comparable)}")
    print(f"CUDA wins: {wins}/{len(comparable)}")

    for row in output_rows:
        print(
            f"{row['rows']}x{row['cols']} epochs={row['epochs']} | "
            f"CPU={row['cpu_time_ns'] / 1e6:.3f} ms | "
            f"CUDA={row['cuda_time_ns'] / 1e6:.3f} ms | "
            f"AUTO={row['auto_time_ns'] / 1e6:.3f} ms"
        )

    if comparable and wins == 0:
        print("CUDA did not win any tested configuration.")
        print("Do not lower the AUTO threshold.")
    elif comparable:
        print("More workload sizes and repetitions are needed before changing AUTO.")


if __name__ == "__main__":
    main()