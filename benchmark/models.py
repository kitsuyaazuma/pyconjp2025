from typing import NamedTuple


class Result(NamedTuple):
    method: str
    avg_time: float
    std_time: float
    problem_size: int | None = None
