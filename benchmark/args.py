# pyright: reportUnusedCallResult=false
import argparse
import os
import multiprocessing as mp
from dataclasses import dataclass


@dataclass
class CommonArgs:
    log_level: str
    max_workers: int
    num_tasks: int
    runs: int
    start_method: str


def create_parser() -> argparse.ArgumentParser:
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
        "-m",
        "--start-method",
        type=str,
        choices=mp.get_all_start_methods(),
        default=mp.get_start_method(allow_none=True),
        help="Multiprocessing start method.",
    )
    return parser
