import threading

counter = 0


def worker(iterations: int) -> None:
    global counter
    for _ in range(iterations):
        counter += 1


if __name__ == "__main__":
    ITERATIONS_PER_THREAD = 100_000

    t1 = threading.Thread(target=worker, args=(ITERATIONS_PER_THREAD,))
    t2 = threading.Thread(target=worker, args=(ITERATIONS_PER_THREAD,))

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    expected = 2 * ITERATIONS_PER_THREAD
    print(f"Expected: {expected}, Actual: {counter}")
