#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""YOLO 嵌套分类 crop 批量推理：同权重、同预处理参数的多 crop 拼 batch（.pt / TensorRT .engine）。"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from script.predict.model_cls import ModelCls
from script.predict.model_cls_crop import (
    cls_infer_pad_square,
    cls_infer_to_gray,
    crop_cls_instance_bgr,
    resolve_from_bbox,
)
from script.predict.model_gpu_crop import cls_job_can_defer_gpu_crop, get_gpu_crop_session
from script.predict.model_cls_factory import cls_cache_key, create_classifier, resolve_cls_backend
from script.config_paths import get_predict_cfg
from script.predict.model_trt import is_trt_engine_path, read_trt_engine_max_batch, resolve_inference_model_path
from script.predict_seg_lib import (
    resolve_cls_crop_background,
    resolve_cls_pad_color,
)

log = logging.getLogger(__name__)

DEFAULT_CLS_BATCH_SIZE = 32

_yolo_cls_batch_cache: dict[tuple[str, str, str], bool] = {}


def _cls_yolo_batch_cache_key(cfg: dict[str, Any]) -> tuple[str, str, str] | None:
    pt = str(cfg.get("model") or "").strip()
    if not pt:
        return None
    return (
        pt,
        str(cfg.get("cls_backend") or cfg.get("model_backend") or "auto").strip().lower(),
        str(cfg.get("timm_model") or "").strip(),
    )


def cfg_uses_yolo_cls_batch(cfg: dict[str, Any]) -> bool:
    """
    嵌套 ``models.cls`` 是否走 YOLO 批量预填（``ModelCls.predict_batch``）。

    - backend 为 ``yolo``（非 ConvNeXt/timm）
    - ``trt_switch=true`` 且 CUDA 可用时须存在可加载的 ``.engine`` 才批量
    - 否则使用 ``.pt``，YOLO11 classify 同样支持 list batch 推理

    同一 ``models.cls`` 配置在进程内只解析一次 backend（P1）。
    """
    cache_key = _cls_yolo_batch_cache_key(cfg)
    if cache_key is None:
        return False
    cached = _yolo_cls_batch_cache.get(cache_key)
    if cached is not None:
        return cached
    uses = resolve_cls_backend(cfg, cache_key[0]) == "yolo"
    _yolo_cls_batch_cache[cache_key] = uses
    return uses


def cfg_uses_trt_yolo_cls(cfg: dict[str, Any]) -> bool:
    """兼容旧名；等价于 ``cfg_uses_yolo_cls_batch``。"""
    return cfg_uses_yolo_cls_batch(cfg)


def resolve_cls_batch_size(
    cfg: dict[str, Any] | None,
    *,
    root_cfg: dict[str, Any] | None = None,
    global_cfg: dict[str, Any] | None = None,
) -> int:
    """
    解析 YOLO 嵌套分类 crop 批量大小。

    优先级：``models.cls`` 节点 > 根 detect/seg > ``predict_cfg.cls_batch_size``；默认 32。
    兼容旧顶层 ``cls_trt_batch_size`` / ``trt_batch_size``。
    TensorRT 运行时还会被 engine metadata 中的 max batch 限制。
    """
    lookup_keys = ("cls_batch_size", "cls_trt_batch_size", "trt_batch_size")
    for src in (cfg, root_cfg, global_cfg):
        if not isinstance(src, dict):
            continue
        for layer in (src, get_predict_cfg(src)):
            if not layer:
                continue
            for key in lookup_keys:
                v = layer.get(key)
                if v is None:
                    continue
                try:
                    n = int(v)
                    if n > 0:
                        return n
                except (TypeError, ValueError):
                    continue
    return DEFAULT_CLS_BATCH_SIZE


def resolve_cls_trt_batch_size(
    cfg: dict[str, Any] | None,
    *,
    root_cfg: dict[str, Any] | None = None,
    global_cfg: dict[str, Any] | None = None,
) -> int:
    """兼容旧名；等价于 ``resolve_cls_batch_size``。"""
    return resolve_cls_batch_size(cfg, root_cfg=root_cfg, global_cfg=global_cfg)


def cls_batch_group_key(cfg: dict[str, Any]) -> str:
    """批量分组：同模型 + 同预处理。"""
    pad_clr = resolve_cls_pad_color(cfg.get("cls_pad_color"))
    return (
        f"{cls_cache_key(cfg)}:"
        f"sq={bool(cfg.get('to_square', True))}:"
        f"gb={bool(cfg.get('gray_binarize', False))}:"
        f"tg={bool(cfg.get('to_gray', False))}:"
        f"pad={pad_clr}:"
        f"poly={not resolve_from_bbox(cfg)}:"
        f"cpr={float(cfg.get('crop_pad_ratio', 0.05))}:"
        f"bg={resolve_cls_crop_background(cfg.get('cls_crop_background'))}"
    )


def make_cls_crop(
    image_bgr: np.ndarray,
    row: dict[str, Any],
    cfg: dict[str, Any],
) -> np.ndarray | None:
    """与 ``_ClsRunner.classify_row`` 一致的 crop 逻辑。"""
    return crop_cls_instance_bgr(image_bgr, row, cfg)


@dataclass
class ClsBatchJob:
    """单条待批量分类任务。"""

    row: dict[str, Any]
    crop: np.ndarray | None
    cfg: dict[str, Any]
    route_label: str
    group_key: str
    batch_size: int
    row_token: int | None = None

    def cache_key(self) -> tuple[int, str]:
        token = self.row_token if self.row_token is not None else id(self.row)
        return (token, self.route_label)


TrtClsJob = ClsBatchJob


def run_cls_job_batches(
    jobs: list[ClsBatchJob],
    *,
    cls_cache: dict[str, Any],
    device: str | None,
    cache_lock: threading.Lock | None = None,
) -> dict[tuple[int, str], dict[str, Any]]:
    """
    按 ``group_key`` 与 ``batch_size`` 切块执行 ``ModelCls.predict_batch``。

    返回 ``{(id(row), route_label): cls_result}``（与 ``ModelCls.predict`` 结构一致）。
    """
    if not jobs:
        return {}

    grouped: dict[str, list[ClsBatchJob]] = {}
    for job in jobs:
        grouped.setdefault(job.group_key, []).append(job)

    out: dict[tuple[int, str], dict[str, Any]] = {}
    for group_key, items in grouped.items():
        cfg = items[0].cfg
        model = _get_or_create_yolo_cls(
            cfg, cls_cache, device, cache_lock=cache_lock
        )
        if model is None:
            raise RuntimeError(
                f"无法加载 YOLO 嵌套分类模型: {cfg.get('model')!r}"
            )
        requested_batch = max(1, int(items[0].batch_size))
        engine_path = str(getattr(model, "model_path", "") or "")
        if is_trt_engine_path(engine_path):
            max_batch = read_trt_engine_max_batch(engine_path)
            batch_size = max(1, min(requested_batch, max_batch))
            if batch_size < requested_batch:
                log.warning(
                    "TRT engine %s 最大 batch=%d，已将 cls_batch_size %d 限制为 %d；"
                    "若需更大批量请用 acc_tensorRT.py 以 DYNAMIC=True, BATCH>=N 重新导出",
                    Path(engine_path).name,
                    max_batch,
                    requested_batch,
                    batch_size,
                )
        else:
            batch_size = requested_batch
        pad_clr = resolve_cls_pad_color(cfg.get("cls_pad_color"))
        pad_square = bool(cfg.get("to_square", True))
        gray_binarize = bool(cfg.get("gray_binarize", False))
        to_gray = cls_infer_to_gray(cfg)
        dev = getattr(model, "device", None) or device
        gpu_session = get_gpu_crop_session()
        t_group0 = time.perf_counter()

        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            if gpu_session is not None and all(
                cls_job_can_defer_gpu_crop(j.row, j.cfg) for j in chunk
            ):
                crops = gpu_session.batch_crop_pad_from_rows(
                    [j.row for j in chunk],
                    cfg=cfg,
                    pad_square=pad_square,
                    pad_color_bgr=pad_clr,
                )
                infer_pad_square = cls_infer_pad_square(cfg, crops[0] if crops else None)
            else:
                crops = []
                for j in chunk:
                    if j.crop is not None:
                        crops.append(j.crop)
                    elif gpu_session is not None:
                        crops.append(
                            make_cls_crop(gpu_session._image_bgr, j.row, j.cfg)
                        )
                    else:
                        crops.append(None)
                infer_pad_square = cls_infer_pad_square(cfg, crops[0] if crops else None)
            results = model.predict_batch(
                crops,
                device=dev,
                pad_square=infer_pad_square,
                gray_binarize=gray_binarize,
                pad_color_bgr=pad_clr,
                to_gray=to_gray,
                max_batch=batch_size,
            )
            for job, cls_result in zip(chunk, results):
                if cls_result is None:
                    continue
                out[job.cache_key()] = cls_result

        backend_tag = "TRT" if is_trt_engine_path(engine_path) else "YOLO"
        elapsed_s = time.perf_counter() - t_group0
        n_jobs = len(items)
        n_batches = (n_jobs + batch_size - 1) // batch_size if n_jobs else 0
        per_job_ms = (elapsed_s * 1000.0 / n_jobs) if n_jobs else 0.0
        log.info(
            "%s 分类批量推理完成: group=%s jobs=%d batch_size=%d batches=%d "
            "耗时=%.3fs (%.2fms/job) model=%s",
            backend_tag,
            group_key[:80],
            n_jobs,
            batch_size,
            n_batches,
            elapsed_s,
            per_job_ms,
            Path(str(cfg.get("model") or "")).name,
        )
    return out


def run_trt_cls_job_batches(
    jobs: list[ClsBatchJob],
    *,
    cls_cache: dict[str, Any],
    device: str | None,
    cache_lock: threading.Lock | None = None,
) -> dict[tuple[int, str], dict[str, Any]]:
    """兼容旧名；等价于 ``run_cls_job_batches``。"""
    return run_cls_job_batches(
        jobs, cls_cache=cls_cache, device=device, cache_lock=cache_lock
    )


def _get_or_create_yolo_cls(
    cfg: dict[str, Any],
    cls_cache: dict[str, Any],
    device: str | None,
    *,
    cache_lock: threading.Lock | None = None,
) -> ModelCls | None:
    pad_clr = resolve_cls_pad_color(cfg.get("cls_pad_color"))
    key = cls_cache_key({**cfg, "cls_pad_color": pad_clr})
    infer_path = resolve_inference_model_path(cfg, quiet=True) or str(
        cfg.get("model") or ""
    )
    lock_ctx = cache_lock if cache_lock is not None else nullcontext()
    with lock_ctx:
        cached = cls_cache.get(key)
        if isinstance(cached, ModelCls):
            cached_path = str(getattr(cached, "model_path", "") or "")
            if cached_path == infer_path and Path(infer_path).is_file():
                return cached
            log.warning(
                "分类模型缓存失效（path=%s exists=%s），将重新加载: %s",
                cached_path,
                Path(cached_path).is_file() if cached_path else False,
                infer_path,
            )
            cls_cache.pop(key, None)
        elif cached is not None:
            cls_cache.pop(key, None)
        try:
            log.info("加载 YOLO 嵌套分类模型: %s", infer_path)
            cls_cache[key] = create_classifier(
                str(cfg["model"]),
                device=device,
                pad_square=bool(cfg.get("to_square", True)),
                gray_binarize=bool(cfg.get("gray_binarize", False)),
                pad_color_bgr=pad_clr,
                to_gray=bool(cfg.get("to_gray", False)),
                cfg=cfg,
            )
        except Exception:
            cls_cache.pop(key, None)
            raise
    model = cls_cache.get(key)
    if not isinstance(model, ModelCls):
        return None
    return model
