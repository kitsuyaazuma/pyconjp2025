from typing import NamedTuple


class Result(NamedTuple):
    method: str
    avg_time: float
    std_time: float
    max_workers: int | None = None
