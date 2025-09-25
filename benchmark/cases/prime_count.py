# pyright: reportUnusedCallResult=false
import logging
import math
import sys
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait
import multiprocessing as mp

from ..args import CommonArgs, create_parser
from ..models import Result
from ..report import display_results
from ..runner import run_benchmark, setup_logging

BENCHMARK_NAME = "Prime Counting Benchmark"
DEFAULT_PROBLEM_SIZE = 10_000_000


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
    max_workers: int,
    num_tasks: int,
    runs: int,
    problem_size: int,
    filter: str | None = None,
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
            wait(futures)
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
            wait(futures)
            results = [future.result() for future in futures]
            count = sum(results)
            assert count == expected_count

    benchmarks = {
        "threading": run_with_threading,
        "multiprocessing": run_with_multiprocessing,
    }
    if filter is not None:
        benchmarks = {filter: benchmarks[filter]}
    results = run_benchmark(benchmarks, runs)
    return results


if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument(
        "-s",
        "--size",
        dest="problem_size",
        type=int,
        default=DEFAULT_PROBLEM_SIZE,
        help="Problem size (upper limit for the prime countdown).",
    )
    parser.add_argument(
        "-f",
        "--filter",
        dest="filter",
        type=str,
        choices=["threading", "multiprocessing"],
        default=None,
        help="Run only the specified benchmark.",
    )
    args = parser.parse_args(namespace=CommonArgs)

    mp.set_start_method(args.start_method)
    setup_logging(args.log_level)

    results = main(
        args.max_workers, args.num_tasks, args.runs, args.problem_size, args.filter
    )
    python_version = sys.version.partition("(")[0].strip()
    display_results(f"{BENCHMARK_NAME}\n({python_version})", results)
