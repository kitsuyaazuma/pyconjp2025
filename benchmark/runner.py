import logging
import time
from typing import Callable
from rich.logging import RichHandler
from rich.progress import track
import numpy as np

from .models import Result


def setup_logging(level: str = "INFO") -> None:
    """Configures logging using RichHandler for clean output."""

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(markup=True, rich_tracebacks=True)],
    )


def run_benchmark(benchmarks: dict[str, Callable[[], None]], runs: int) -> list[Result]:
    """Runs a set of benchmark methods for a specified number of runs and collects execution statistics."""

    execution_times: dict[str, list[float]] = {name: [] for name in benchmarks.keys()}

    for _ in track(range(runs), description="Benchmark Progress"):
        for name, task in benchmarks.items():
            start_time = time.perf_counter()
            task()
            end_time = time.perf_counter()
            execution_times[name].append(end_time - start_time)

    benchmark_results: list[Result] = []
    for name, times in execution_times.items():
        avg = float(np.mean(times))
        std = float(np.std(times))
        benchmark_results.append(Result(name, avg, std))
    return benchmark_results
