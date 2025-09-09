# pyright: reportUnusedCallResult=false
from dataclasses import dataclass
import logging
import sys
import multiprocessing as mp
from multiprocessing import shared_memory
from concurrent.futures import Future, ThreadPoolExecutor, ProcessPoolExecutor, wait
import numpy as np
import numpy.typing as npt

from ..args import CommonArgs, create_parser
from ..models import Result
from ..report import display_results
from ..runner import run_benchmark, setup_logging


def sum_array_chunk(data: npt.NDArray[np.int64]) -> np.int64:
    """Calculates the sum of a given NumPy array chunk."""
    total = np.int64(0)
    # NOTE: Intentionally not using np.sum(data) to simulate a CPU-bound task.
    for value in data.reshape(-1):
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
        for value in data.reshape(-1):
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
            wait(futures)
            total = sum(future.result() for future in futures)
            assert total == expected_total

    def run_with_multiprocessing() -> None:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures: list[Future[np.int64]] = []
            for i in range(num_tasks):
                start = i * chunk_size
                end = (i + 1) * chunk_size if i < num_tasks - 1 else problem_size
                futures.append(executor.submit(sum_array_chunk, data[start:end]))
            wait(futures)
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
                wait(futures)
                total = sum(future.result() for future in futures)
                assert total == expected_total
        finally:
            shm.close()
            shm.unlink()

    results = run_benchmark(
        {
            "threading": run_with_threading,
            "multiprocessing": run_with_multiprocessing,
            "multiprocessing (w/ shm)": run_with_multiprocessing_shm,
        },
        runs,
    )
    return results


@dataclass
class Args(CommonArgs):
    problem_size: int


if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument(
        "-s",
        "--size",
        dest="problem_size",
        type=int,
        default=10_000_000,
        help="Problem size (number of elements in the NumPy array).",
    )
    args = parser.parse_args(namespace=Args)

    mp.set_start_method(args.start_method)
    setup_logging(args.log_level)

    results = main(
        problem_size=args.problem_size,
        max_workers=args.max_workers,
        num_tasks=args.num_tasks,
        runs=args.runs,
    )
    python_version = sys.version.partition("(")[0].strip()
    display_results(f"Array Summation Benchmark\n({python_version})", results)
