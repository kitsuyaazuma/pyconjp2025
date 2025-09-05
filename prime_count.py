# pyright: reportUnusedCallResult=false
import argparse
from dataclasses import dataclass
import logging
import math
import sys
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

from utils import Result, display_results, run_benchmark, setup_logging


def is_prime(n: int) -> bool:
    """Checks if a number is prime."""
    if n <= 1:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


def count_primes_in_range(start: int, end: int) -> int:
    """Counts the number of primes within a given range."""
    return sum(1 for n in range(start, end) if is_prime(n))


def count_primes_sieve(limit: int) -> int:
    """Counts primes p where 0 <= p < limit using a simple Sieve of Eratosthenes."""
    if limit <= 2:
        return 0

    sieve = bytearray([1]) * limit
    sieve[0] = 0
    sieve[1] = 0

    upper = int(math.isqrt(limit - 1))
    for p in range(2, upper + 1):
        if sieve[p]:
            num_zeros = len(range(p * p, limit, p))
            sieve[p * p : limit : p] = bytearray(num_zeros)

    return int(sum(sieve))


def main(
    problem_size: int, max_workers: int, num_tasks: int, runs: int
) -> list[Result]:
    """Sets up and runs the prime counting benchmark."""
    logging.info(f"Checking primes up to {problem_size:,} using {max_workers} workers.")

    chunk_size = problem_size // num_tasks

    expected_count = count_primes_sieve(problem_size)

    def run_with_threading() -> None:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    count_primes_in_range,
                    i * chunk_size,
                    (i + 1) * chunk_size if i < num_tasks - 1 else problem_size,
                )
                for i in range(num_tasks)
            ]
            results = [future.result() for future in futures]
            count = sum(results)
            assert count == expected_count

    def run_with_multiprocessing() -> None:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    count_primes_in_range,
                    i * chunk_size,
                    (i + 1) * chunk_size if i < num_tasks - 1 else problem_size,
                )
                for i in range(num_tasks)
            ]
            results = [future.result() for future in futures]
            count = sum(results)
            assert count == expected_count

    gil_enabled = sys._is_gil_enabled()  # pyright: ignore[reportPrivateUsage]
    results = run_benchmark(
        {
            f"Threading ({'w/' if gil_enabled else 'w/o'} GIL)": run_with_threading,
            "Multiprocessing": run_with_multiprocessing,
        },
        runs=runs,
    )
    return results


@dataclass
class Args:
    log_level: str
    problem_size: int
    max_workers: int
    num_tasks: int
    runs: int
    start_method: str


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
        "-s",
        "--size",
        dest="problem_size",
        type=int,
        default=1_000_000,
        help="Problem size (upper limit for the prime countdown).",
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
    )
    args = parser.parse_args(namespace=Args)

    mp.set_start_method(args.start_method)
    setup_logging(args.log_level)

    results = main(args.problem_size, args.max_workers, args.num_tasks, args.runs)
    python_version = sys.version.partition("(")[0].strip()
    display_results(f"Prime Number Benchmark\n({python_version})", results)
