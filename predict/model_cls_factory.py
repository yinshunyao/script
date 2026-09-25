#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""分类模型工厂：YOLO / timm ConvNeXt / 分类 ONNX 统一创建。"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from script.predict.model_cls import ModelCls
from script.predict.model_cls_onnx import ModelClsOnnx, is_onnx_cls_path
from script.predict.model_cls_timm import ModelClsTimm, is_timm_cls_checkpoint
from script.predict.model_trt import resolve_inference_model_path

log = logging.getLogger(__name__)

ClsModel = ModelCls | ModelClsTimm | ModelClsOnnx
_ONNX_HINTS = ("onnx", "ort")
_CONVNEXT_HINTS = ("convnext", "timm")
_YOLO_HINTS = ("yolo", "ultralytics")
_ALLOWED_HINTS = _CONVNEXT_HINTS + _YOLO_HINTS + _ONNX_HINTS + ("auto", "")


@lru_cache(maxsize=128)
def _resolve_cls_backend_cached(model_path: str, backend_hint: str) -> str:
    """``auto`` / 空 hint 时按路径或 checkpoint 结构识别；结果按 (path, hint) 全局缓存。"""
    onnx_path = is_onnx_cls_path(model_path)
    if backend_hint in _ONNX_HINTS:
        if model_path and not onnx_path:
            raise ValueError(f"cls_backend=onnx 但权重不是 .onnx: {model_path}")
        return "onnx"
    if onnx_path:
        if backend_hint in _CONVNEXT_HINTS:
            raise ValueError(
                f"cls_backend={backend_hint} 不能加载 .onnx，请改 onnx/auto 或换 timm .pt: {model_path}"
            )
        return "onnx"
    if backend_hint in _CONVNEXT_HINTS:
        return "convnext"
    if backend_hint in _YOLO_HINTS:
        return "yolo"
    if backend_hint not in ("auto", ""):
        raise ValueError(
            f"未知 cls_backend={backend_hint!r}，支持 yolo / convnext / onnx / auto"
        )
    if not model_path:
        return "yolo"
    return "convnext" if is_timm_cls_checkpoint(model_path) else "yolo"


def resolve_cls_backend(cfg: dict[str, Any] | None, model_path: str) -> str:
    """
    解析分类后端：``yolo`` | ``convnext`` | ``onnx``。

    - 路径以 ``.onnx`` 结尾一律走 ONNX Runtime（即使 ``cls_backend`` 仍写 ``yolo``）
    - 显式 ``onnx`` / ``ort`` → onnx；``convnext`` / ``timm`` → convnext；``yolo`` / ``ultralytics`` → yolo
    - ``auto`` 或空：``.onnx`` → onnx，否则按 checkpoint 是否含 ``model_state``
    """
    raw = ""
    if cfg:
        raw = str(cfg.get("cls_backend") or cfg.get("model_backend") or "auto").strip().lower()
    if raw not in _ALLOWED_HINTS:
        raise ValueError(f"未知 cls_backend={raw!r}，支持 yolo / convnext / onnx / auto")
    pt = str(model_path or "").strip()
    hint = "auto" if raw in ("auto", "") else raw
    backend = _resolve_cls_backend_cached(pt, hint)
    if raw in _YOLO_HINTS and backend == "onnx":
        log.info(
            "model 为 .onnx，已按 ONNX Runtime 分类加载（忽略 cls_backend=yolo）: %s",
            model_path,
        )

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
    imgsz = image_size if image_size is not None else merged.get("image_size")
    if imgsz is not None:
        imgsz = int(imgsz) if int(imgsz) > 0 else None
    if backend == "onnx":
        return ModelClsOnnx(
            model_path=load_path,
            device=device,
            pad_square=pad_square,
            gray_binarize=gray_binarize,
            pad_color_bgr=pad_color_bgr,
            to_gray=to_gray,
            image_size=imgsz,
        )
    if backend == "convnext":
        tm = timm_model or merged.get("timm_model")
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
