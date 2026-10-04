"""Compute backend selection helpers for CPU, threaded, and WebGPU execution."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")

_VALID_BACKENDS = {"auto", "cpu", "threaded", "webgpu"}


def get_compute_backend() -> str:
    """Return the active backend selection from the environment.

    Supported values are "auto", "cpu", "threaded", and "webgpu".
    The default is "auto", which prefers the GPU backend when available
    and otherwise falls back to the CPU path.
    """
    backend = os.environ.get("INKSTITCH_COMPUTE_BACKEND", "auto").strip().lower()
    if backend not in _VALID_BACKENDS:
        return "auto"
    return backend


def use_threaded_cpu() -> bool:
    return get_compute_backend() == "threaded"


def use_webgpu() -> bool:
    return get_compute_backend() in {"auto", "webgpu"}


def parallel_map(func: Callable[[T], R], items: Sequence[T], max_workers: int | None = None) -> list[R]:
    """Apply a function to a sequence using a small worker pool when threaded execution is requested."""
    try:
        item_count = len(items)
    except TypeError:
        items = tuple(items)
        item_count = len(items)

    if item_count == 0:
        return []

    workers = max_workers or min(item_count, (os.cpu_count() or 1))
    if workers <= 1:
        return [func(item) for item in items]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(func, items))
