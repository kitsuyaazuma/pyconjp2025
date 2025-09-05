from collections import defaultdict
import logging
import time
from typing import Callable, NamedTuple
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.progress import track
import numpy as np
import matplotlib.pyplot as plt


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


class Result(NamedTuple):
    method: str
    avg_time: float
    std_time: float
    problem_size: int | None = None


def display_results(title: str, results: list[Result]) -> None:
    """Displays benchmark results in a formatted table using Rich."""
    console = Console()

    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Method", style="dim")
    table.add_column("Execution Time (s)", justify="right")

    for result in results:
        table.add_row(result.method, f"{result.avg_time:.4f} ± {result.std_time:.4f}")

    console.print(table)


def plot_results(title: str, results: list[Result], output_filename: str) -> None:
    grouped_results: defaultdict[str, list[tuple[int, float, float]]] = defaultdict(
        list
    )
    for result in results:
        assert result.problem_size is not None
        grouped_results[result.method].append(
            (result.problem_size, result.avg_time, result.std_time)
        )

    plt.figure(figsize=(12, 8))
    fig, ax = plt.subplots(figsize=(12, 8))

    for method, data in grouped_results.items():
        data.sort(key=lambda x: x[0])
        problem_sizes, avg_times, std_times = zip(*data)

        ax.errorbar(
            problem_sizes,
            avg_times,
            yerr=std_times,
            label=method,
            marker="o",
            capsize=3,
        )

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Problem Size", fontsize=12)
    ax.set_ylabel("Execution Time (s)", fontsize=12)
    ax.set_xscale("log")
    # ax.set_yscale('log')
    ax.grid(True, which="both", ls="--", linewidth=0.5)
    ax.legend(title="Method")

    fig.tight_layout()
    fig.savefig(output_filename)
    plt.close(fig)

    logging.info(f"Saved line graph to {output_filename}")
