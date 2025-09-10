from datetime import datetime
import multiprocessing as mp
import os
from pathlib import Path
import sys
from typing import Callable, NamedTuple
import math

from benchmark import (
    Result,
    display_results,
    plot_results,
    save_results_to_csv,
    setup_logging,
)
from benchmark.args import CommonArgs, create_parser
from benchmark.cases import array_sum, prime_count


class BenchmarkCase(NamedTuple):
    name: str
    output_filename: str
    problem_size: int
    max_workers_list: list[int]
    run_function: Callable[[int, int, int, int], list[Result]]


MAX_WORKERS_LIST = [2**i for i in range(int(math.log2(os.cpu_count() or 1) + 1))]

BENCHMARK_CASES: list[BenchmarkCase] = [
    BenchmarkCase(
        "Array Summation Benchmark",
        "array_sum",
        100_000_000,
        MAX_WORKERS_LIST,
        array_sum.main,
    ),
    BenchmarkCase(
        "Prime Counting Benchmark",
        "prime_count",
        10_000_000,
        MAX_WORKERS_LIST,
        prime_count.main,
    ),
]


def main(runs: int) -> None:
    all_plot_data: list[tuple[str, Path, list[Result]]] = []
    python_version = sys.version.partition("(")[0].strip()

    base_results_dir = Path("results")
    base_results_dir.mkdir(exist_ok=True)

    results_dir = base_results_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir.mkdir(exist_ok=True)

    for case in BENCHMARK_CASES:
        all_results_for_case: list[Result] = []
        title = f"{case.name}\n({python_version})"
        for max_workers in case.max_workers_list:
            results = case.run_function(
                case.problem_size, max_workers, max_workers, runs
            )
            display_results(title, results)
            for r in results:
                all_results_for_case.append(
                    Result(r.method, r.avg_time, r.std_time, max_workers)
                )

        base_filename = case.output_filename
        if not sys._is_gil_enabled():  # pyright: ignore[reportPrivateUsage]
            base_filename += "_nogil"
        png_path = results_dir / f"{base_filename}.png"
        csv_path = results_dir / f"{base_filename}.csv"

        save_results_to_csv(all_results_for_case, csv_path)
        all_plot_data.append((title.replace("\n", " "), png_path, all_results_for_case))

    for title, output_filename, results in all_plot_data:
        plot_results(title, results, output_filename)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args(namespace=CommonArgs)

    mp.set_start_method(args.start_method)
    setup_logging(args.log_level)
    main(args.runs)
