from datetime import datetime
from importlib import import_module
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
import typer


class BenchmarkCase(NamedTuple):
    id: str
    name: str
    problem_size: int
    run_function: Callable[[int, int, int, int], list[Result]]


CASES_DIR = Path(__file__).parent / "benchmark" / "cases"
MODULES = [
    (file_path.stem, import_module(f"benchmark.cases.{file_path.stem}"))
    for file_path in CASES_DIR.glob("*.py")
    if not file_path.stem.startswith("_")
]
BENCHMARK_CASES = [
    BenchmarkCase(
        id=id,
        name=getattr(module, "BENCHMARK_NAME"),
        problem_size=getattr(module, "DEFAULT_PROBLEM_SIZE"),
        run_function=getattr(module, "main"),
    )
    for id, module in MODULES
]
MAX_WORKERS_LIST: list[int] = [
    2**i for i in range(int(math.log2(os.cpu_count() or 1) + 1))
]


def main(runs: int = 1, log_level: str = "INFO") -> None:
    setup_logging(log_level)
    all_plot_data: list[tuple[str, Path, list[Result]]] = []
    python_version = sys.version.partition("(")[0].strip()

    base_results_dir = Path("results")
    base_results_dir.mkdir(exist_ok=True)

    results_dir = base_results_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir.mkdir(exist_ok=True)

    for case in BENCHMARK_CASES:
        all_results_for_case: list[Result] = []
        for max_workers in MAX_WORKERS_LIST:
            results = case.run_function(
                max_workers, max_workers, runs, case.problem_size
            )
            display_results(f"{case.name}\n({python_version})", results)
            for r in results:
                all_results_for_case.append(
                    Result(r.method, r.avg_time, r.std_time, max_workers)
                )

        base_filename = case.id
        if not sys._is_gil_enabled():  # pyright: ignore[reportPrivateUsage]
            base_filename += "_nogil"
        png_path = results_dir / f"{base_filename}.png"
        csv_path = results_dir / f"{base_filename}.csv"

        save_results_to_csv(all_results_for_case, csv_path)
        all_plot_data.append(
            (f"{case.name} ({python_version})", png_path, all_results_for_case)
        )

    for title, output_filename, results in all_plot_data:
        plot_results(title, results, output_filename)


if __name__ == "__main__":
    typer.run(main)
