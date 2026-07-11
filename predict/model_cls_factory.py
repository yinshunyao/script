#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""分类模型工厂：YOLO（``ModelCls``）与 timm ConvNeXt（``ModelClsTimm``）统一创建。"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from script.predict.model_cls import ModelCls
from script.predict.model_cls_timm import ModelClsTimm, is_timm_cls_checkpoint
from script.predict.model_trt import resolve_inference_model_path

log = logging.getLogger(__name__)

ClsModel = ModelCls | ModelClsTimm


@lru_cache(maxsize=128)
def _resolve_cls_backend_cached(model_path: str, backend_hint: str) -> str:
    """``auto`` / 空 hint 时按 checkpoint 结构识别；结果按 (path, hint) 全局缓存。"""
    if backend_hint in ("convnext", "timm"):
        return "convnext"
    if backend_hint in ("yolo", "ultralytics"):
        return "yolo"
    if backend_hint not in ("auto", ""):
        raise ValueError(
            f"未知 cls_backend={backend_hint!r}，支持 yolo / convnext / auto"
        )
    if not model_path:
        return "yolo"
    return "convnext" if is_timm_cls_checkpoint(model_path) else "yolo"


def resolve_cls_backend(cfg: dict[str, Any] | None, model_path: str) -> str:
    """
    解析分类后端：``yolo`` | ``convnext``。

    - 显式 ``cls_backend`` / ``model_backend``：``convnext``、``timm`` → convnext；``yolo``、``ultralytics`` → yolo
    - ``auto`` 或空：按 checkpoint 结构自动识别（每个权重路径仅检测一次）
    """
    raw = ""
    if cfg:
        raw = str(cfg.get("cls_backend") or cfg.get("model_backend") or "auto").strip().lower()
    if raw not in ("convnext", "timm", "yolo", "ultralytics", "auto", ""):
        raise ValueError(f"未知 cls_backend={raw!r}，支持 yolo / convnext / auto")
    pt = str(model_path or "").strip()
    hint = "auto" if raw in ("auto", "") else raw
    backend = _resolve_cls_backend_cached(pt, hint)

    timm_hint = str((cfg or {}).get("timm_model") or "").strip()
    if timm_hint and backend == "yolo":
        log.warning(
            "配置含 timm_model=%r 但 checkpoint 非 timm 格式，已按 YOLO 加载: %s",
            timm_hint,
            model_path,
        )
    return backend


def create_classifier(
    model_path: str,
    *,
    device: str | None = None,
    pad_square: bool = False,
    gray_binarize: bool = False,
    pad_color_bgr: tuple[int, int, int] = (255, 255, 255),
    to_gray: bool = False,
    cls_backend: str | None = None,
    timm_model: str | None = None,
    image_size: int | None = None,
    cfg: dict[str, Any] | None = None,
) -> ClsModel:
    """按配置或 checkpoint 创建分类器，供 ``predict_all`` / ``PredictSeg`` 等复用。"""
    merged: dict[str, Any] = dict(cfg or {})
    if cls_backend is not None:
        merged["cls_backend"] = cls_backend
    backend = resolve_cls_backend(merged, model_path)
    load_path = (
        resolve_inference_model_path(merged, model_path=model_path, quiet=True)
        if backend == "yolo"
        else model_path
    )
    if backend == "convnext":
        tm = timm_model or merged.get("timm_model")
        imgsz = image_size if image_size is not None else merged.get("image_size")
        if imgsz is not None:
            imgsz = int(imgsz) if int(imgsz) > 0 else None
        return ModelClsTimm(
            model_path=load_path,
            device=device,
            pad_square=pad_square,
            gray_binarize=gray_binarize,
            pad_color_bgr=pad_color_bgr,
            to_gray=to_gray,
            timm_model=str(tm).strip() if tm else None,
            image_size=imgsz,
        )
    return ModelCls(
        model_path=load_path,
        device=device,
        pad_square=pad_square,
        gray_binarize=gray_binarize,
        pad_color_bgr=pad_color_bgr,
        to_gray=to_gray,
    )


def cls_cache_key(cfg: dict[str, Any]) -> str:
    """``InsectPredictAll`` 分类器缓存键：按权重路径区分，预处理参数在 ``predict()`` 调用时传入。"""
    pt = str(cfg.get("model") or "")
    backend = resolve_cls_backend(cfg, pt) if pt else "yolo"
    if backend == "yolo":
        path = resolve_inference_model_path(cfg, quiet=True) or pt
    else:
        path = pt
    return f"cls:{backend}:{path}:{cfg.get('timm_model')}:{cfg.get('image_size')}"
