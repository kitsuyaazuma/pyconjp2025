# pyright: reportUnusedCallResult=false
import argparse
from dataclasses import dataclass
import logging
import sys
import os
import multiprocessing as mp
from multiprocessing import shared_memory
from concurrent.futures import Future, ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
import numpy.typing as npt
from utils import Result, setup_logging, run_benchmark, display_results


def sum_array_chunk(data: npt.NDArray[np.int64]) -> np.int64:
    """Calculates the sum of a given NumPy array chunk."""
    total = np.int64(0)
    # NOTE: Intentionally not using np.sum(data) to simulate a CPU-bound task.
    for value in data.flatten():
        total += value
    return total


def sum_array_shm(
    shm_name: str, shape: tuple[int], dtype: np.dtype, start: int, end: int
) -> np.int64:
    """Calculates the sum of a segment of a NumPy array in shared memory."""
    shm = None
    try:
        shm = shared_memory.SharedMemory(name=shm_name)
        data: npt.NDArray[np.int64] = np.ndarray(shape, dtype=dtype, buffer=shm.buf)[
            start:end
        ]
        # NOTE: Intentionally not using np.sum(data) to simulate a CPU-bound task.
        total = np.int64(0)
        for value in data.flatten():
            total += value
    finally:
        if shm:
            shm.close()
    return total


def main(
    problem_size: int, max_workers: int, num_tasks: int, runs: int
) -> list[Result]:
    """Sets up and runs the NumPy array summation benchmark."""
    logging.info(
        f"Summing array with {problem_size:,} elements using {max_workers} workers."
    )

    data = np.arange(problem_size, dtype=np.int64)
    chunk_size = problem_size // num_tasks

    expected_total = np.int64(problem_size * (problem_size - 1) // 2)

    def run_with_threading() -> None:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures: list[Future[np.int64]] = []
            for i in range(num_tasks):
                start = i * chunk_size
                end = (i + 1) * chunk_size if i < num_tasks - 1 else problem_size
                futures.append(executor.submit(sum_array_chunk, data[start:end]))
            total = sum(future.result() for future in futures)
            assert total == expected_total

    def run_with_multiprocessing() -> None:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures: list[Future[np.int64]] = []
            for i in range(num_tasks):
                start = i * chunk_size
                end = (i + 1) * chunk_size if i < num_tasks - 1 else problem_size
                futures.append(executor.submit(sum_array_chunk, data[start:end]))
            total = sum(future.result() for future in futures)
            assert total == expected_total

    def run_with_multiprocessing_shm() -> None:
        shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
        try:
            shm_array = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
            shm_array[:] = data[:]
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures: list[Future[np.int64]] = []
                for i in range(num_tasks):
                    start = i * chunk_size
                    end = (i + 1) * chunk_size if i < num_tasks - 1 else problem_size
                    futures.append(
                        executor.submit(
                            sum_array_shm, shm.name, data.shape, data.dtype, start, end
                        )
                    )
                total = sum(future.result() for future in futures)
                assert total == expected_total
        finally:
            shm.close()
            shm.unlink()

    gil_enabled = sys._is_gil_enabled()  # pyright: ignore[reportPrivateUsage]
    results = run_benchmark(
        {
            f"Threading ({'w/' if gil_enabled else 'w/o'} GIL)": run_with_threading,
            "Multiprocessing": run_with_multiprocessing,
            "Multiprocessing (w/ Shared Memory)": run_with_multiprocessing_shm,
        },
        runs,
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
        default=10_000_000,
        help="Problem size (number of elements in the NumPy array).",
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
    args = parser.parse_args(namespace=Args)

    setup_logging()
    mp.set_start_method(args.start_method)

    results = main(
        problem_size=args.problem_size,
        max_workers=args.max_workers,
        num_tasks=args.num_tasks,
        runs=args.runs,
    )
    python_version = sys.version.partition("(")[0].strip()
    display_results(f"Array Summation Benchmark\n({python_version})", results)
