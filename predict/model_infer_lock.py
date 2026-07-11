#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""按模型权重路径 + task 的 GPU 推理互斥锁（与 ``model_yolo_cache`` 键一致）。"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

_LOCKS: dict[tuple[str, str], threading.Lock] = {}
_META_LOCK = threading.Lock()


def infer_lock_key(model_path: str, *, task: str | None = None) -> tuple[str, str]:
    path = str(Path(model_path).expanduser().resolve())
    return (path, str(task or ""))


def get_model_infer_lock(model_path: str, *, task: str | None = None) -> threading.Lock:
    key = infer_lock_key(model_path, task=task)
    lock = _LOCKS.get(key)
    if lock is not None:
        return lock
    with _META_LOCK:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _LOCKS[key] = lock
        return lock


@contextmanager
def model_infer_guard(model_path: str, *, task: str | None = None) -> Iterator[None]:
    with get_model_infer_lock(model_path, task=task):
        yield
