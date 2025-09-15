# pyconjp2025

This repository contains the demonstration code for the PyConJP 2025 talk: "[Beyond Multiprocessing: A Real-World ML Workload Speedup with Python 3.13+ Free-Threading](https://2025.pycon.jp/en/timetable/talk/HADBDX)".

It provides a suite of benchmarks designed to compare the performance of standard CPython (with the Global Interpreter Lock) against the free-threading build of CPython 3.14.

## Benchmarks

The following benchmarks are included:

- **Prime Counting**: A CPU-bound task that counts prime numbers up to a given limit, implemented with both `threading` and `multiprocessing`.

- **Array Summation**: Another CPU-bound task that calculates the sum of a large NumPy array using different concurrency models: `threading`, `multiprocessing`, and `multiprocessing` with shared memory.

## Getting Started

### Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager

### Setup

#### 1. Clone the repository

```bash
git clone https://github.com/kitsuyaazuma/pyconjp2025.git
cd pyconjp2025
```

#### 2. Install dependencies:

```bash
uv sync
```

#### 3. Set up Python environment:

- With GIL (Standard CPython):

  ```bash
  make gil
  ```

- Without GIL (free-threading CPython):

  ```bash
  make nogil
  ```

### Running the Benchmarks

To run the full suite of benchmarks:

```bash
uv run python main.py
```

The script will execute each benchmark with a range of worker counts, displaying the results in the console and saving them as CSV files and PNG graphs in the `results/` directory.

You can customize the number of runs for each benchmark using the `--runs` option:

```
uv run python main.py --runs 5
```

### Running Individual Benchmarks

You can also run each benchmark as a Python module to have more control over the parameters.

- Array Summation Benchmark:

    ```bash
    uv run python -m benchmark.cases.array_sum --max-workers 8 --runs 10 --size 100000000
    ```

- Prime Counting Benchmark:

    ```bash
    uv run python -m benchmark.cases.array_sum --max-workers 8 --runs 10 --size 100000000
    ```

