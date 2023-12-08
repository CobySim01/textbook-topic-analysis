from typing import Callable
from inspect import signature


def num_arguments(fn: Callable) -> int:
    return len(signature(fn).parameters)
