# pyright: reportUnusedCallResult=false
import argparse
import os
import multiprocessing as mp
from dataclasses import dataclass
import sys
from typing import Callable, NamedTuple

import array_sum
import prime_count
from utils import Result, display_results, plot_results, setup_logging


class BenchmarkCase(NamedTuple):
    name: str
    output_filename: str
    problem_sizes: list[int]
    run_function: Callable[[int, int, int, int], list[Result]]


BENCHMARK_CASES: list[BenchmarkCase] = [
    BenchmarkCase(
        "Array Summation Benchmark",
        "array_sum.png",
        [
            10_000,
            50_000,
            100_000,
            500_000,
            1_000_000,
            5_000_000,
            10_000_000,
            50_000_000,
            100_000_000,
        ],
        array_sum.main,
    ),
    BenchmarkCase(
        "Prime Counting Benchmark",
        "prime_count.png",
        [1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000],
        prime_count.main,
    ),
]


@dataclass
class Args:
    log_level: str
    max_workers: int
    num_tasks: int
    runs: int
    start_method: str


def main(max_workers: int, num_tasks: int, runs: int) -> None:
    all_plot_data: list[tuple[str, str, list[Result]]] = []
    python_version = sys.version.partition("(")[0].strip()

    for case in BENCHMARK_CASES:
        all_results_for_case: list[Result] = []
        title = f"{case.name}\n({python_version})"
        for size in case.problem_sizes:
            results = case.run_function(size, max_workers, num_tasks, runs)
            display_results(title, results)
            for r in results:
                all_results_for_case.append(
                    Result(r.method, r.avg_time, r.std_time, size)
                )

        output_filename = case.output_filename
        if not sys._is_gil_enabled():  # pyright: ignore[reportPrivateUsage]
            output_filename = case.output_filename.replace(".png", "_nogil.png")
        all_plot_data.append((case.name, output_filename, all_results_for_case))

    for title, output_filename, results in all_plot_data:
        plot_results(title, results, output_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level.",
    )
    parser.add_argument(
        "-w",
        "--max-workers",
        type=int,
        default=os.cpu_count(),
        help="Maximum number of workers.",
    )
    parser.add_argument(
        "-t",
        "--num-tasks",
        type=int,
        default=os.cpu_count(),
        help="Number of tasks to divide the work into.",
    )
    parser.add_argument(
        "-r",
        "--runs",
        type=int,
        default=3,
        help="Number of times to run the benchmark.",
    )
    parser.add_argument(
        "-s",
        "--start-method",
        type=str,
        choices=mp.get_all_start_methods(),
        default=mp.get_start_method(allow_none=True),
        help="Multiprocessing start method.",
    )
    args = parser.parse_args(namespace=Args)

    mp.set_start_method(args.start_method)
    setup_logging(args.log_level)
    main(args.max_workers, args.num_tasks, args.runs)
