from collections import defaultdict
import logging
from pathlib import Path

from .models import Result


def display_results(title: str, results: list[Result]) -> None:
    """Displays benchmark results in a formatted table using Rich."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Method", style="dim")
    table.add_column("Execution Time (s)", justify="right")

    for result in results:
        table.add_row(result.method, f"{result.avg_time:.4f} ± {result.std_time:.4f}")

    console.print(table)


def plot_results(
    title: str, results: list[Result], output_filename: str | Path
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator

    grouped_results: defaultdict[str, list[tuple[int, float, float]]] = defaultdict(
        list
    )
    for result in results:
        assert result.problem_size is not None
        grouped_results[result.method].append(
            (result.problem_size, result.avg_time, result.std_time)
        )

    plt.rcParams["font.size"] = 14
    plt.rcParams["axes.labelsize"] = 14
    fig, ax = plt.subplots(figsize=(8, 6))

    for method, data in grouped_results.items():
        data.sort(key=lambda x: x[0])
        problem_sizes, avg_times, std_times = zip(*data)

        ax.errorbar(
            problem_sizes,
            avg_times,
            yerr=std_times,
            label=method,
            marker="o",
            markersize=7,
            linewidth=2,
            capsize=5,
            elinewidth=1.5,
        )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Problem Size")
    ax.set_ylabel("Execution Time (s)")
    ax.set_xscale("log")
    ax.tick_params(axis="both", which="major")
    ax.tick_params(axis="both", which="minor")
    ax.grid(True, which="major", ls="--", linewidth=0.8, alpha=0.6)
    ax.grid(True, which="minor", ls=":", linewidth=0.6, alpha=0.4)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.legend(fontsize=14)

    fig.tight_layout()
    fig.savefig(output_filename, dpi=200)
    plt.close(fig)

    logging.info(f"Saved line graph to {output_filename}")


def save_results_to_csv(results: list[Result], output_filename: str | Path) -> None:
    """Saves benchmark results to a CSV file."""
    import csv

    with open(output_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ["Method", "Average Time (s)", "Standard Deviation (s)", "Problem Size"]
        )
        for result in results:
            writer.writerow(
                [result.method, result.avg_time, result.std_time, result.problem_size]
            )
    logging.info(f"Saved results to {output_filename}")
