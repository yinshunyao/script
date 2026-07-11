#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""YOLO / TensorRT engine 权重进程内缓存，避免同一路径重复反序列化 engine。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_YOLO_CACHE: dict[tuple[str, str], Any] = {}


def get_cached_yolo(model_path: str, *, task: str | None = None) -> Any:
    """
    按 ``(绝对路径, task)`` 缓存 ``ultralytics.YOLO`` 实例。

    嵌套分类多路由、预处理参数不同但共用同一 ``.engine`` 时，只加载一次 TensorRT。
    """
    from ultralytics import YOLO

    path = str(Path(model_path).expanduser().resolve())
    task_key = str(task or "")
    key = (path, task_key)
    if key in _YOLO_CACHE:
        return _YOLO_CACHE[key]

    log.info("加载 YOLO 权重: %s%s", path, f" task={task_key}" if task_key else "")
    model = YOLO(path, task=task) if task_key else YOLO(path)
    _YOLO_CACHE[key] = model
    return model


def clear_yolo_cache() -> None:
    """释放进程内 YOLO / engine 缓存（``InsectPredictAll.release`` 时调用）。"""
    if _YOLO_CACHE:
        log.info("释放 YOLO 权重缓存 %d 项", len(_YOLO_CACHE))
    _YOLO_CACHE.clear()


def evict_yolo_cache(model_path: str, *, task: str | None = None) -> None:
    """从进程内 YOLO 缓存移除指定权重（模型加载失败或权重更新后调用）。"""
    path = str(Path(model_path).expanduser().resolve())
    task_key = str(task or "")
    key = (path, task_key)
    if key in _YOLO_CACHE:
        log.info("清除 YOLO 权重缓存: %s%s", path, f" task={task_key}" if task_key else "")
        del _YOLO_CACHE[key]


def is_model_load_error(exc: BaseException) -> bool:
    """判断异常是否由模型权重缺失/无法加载引起（用于失败后丢弃管线缓存）。"""
    if isinstance(exc, FileNotFoundError):
        return True
    if isinstance(exc, OSError) and getattr(exc, "errno", None) in (2,):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        if "no such file" in msg or "cannot load" in msg or "找不到" in msg:
            return True
    cause = exc.__cause__
    if cause is not None and cause is not exc and is_model_load_error(cause):
        return True
    context = exc.__context__
    if context is not None and context is not exc and is_model_load_error(context):
        return True
    return False
