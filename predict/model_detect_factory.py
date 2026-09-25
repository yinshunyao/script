#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检测模型工厂：YOLO（``ModelDetector``）与 ONNX（``ModelDetectorOnnx``）。"""

from __future__ import annotations

from typing import Any

from script.predict.model_detect import ModelDetector

DETECT_PIPELINE_TYPES = frozenset({"detect", "yolo", "onnx", "ort"})
ONNX_MODEL_TYPES = frozenset({"onnx", "ort"})


def normalize_model_type(model_type: str | None) -> str:
    m = str(model_type or "detect").strip().lower()
    return "detect" if m == "yolo" else m


def is_detect_pipeline_type(model_type: str | None) -> bool:
    """是否走 PredictSize 检测管线（含 YOLO detect 与 ONNX）。"""
    return normalize_model_type(model_type) in DETECT_PIPELINE_TYPES


def is_onnx_model_type(model_type: str | None) -> bool:
    return normalize_model_type(model_type) in ONNX_MODEL_TYPES


def is_onnx_detect_backend(
    model_type: str | None = None,
    model_path: str | None = None,
) -> bool:
    """显式 ``onnx``，或权重路径为 ``.onnx``。"""
    if is_onnx_model_type(model_type):
        return True
    return str(model_path or "").strip().lower().endswith(".onnx")


def pipeline_type_of(model_type: str | None) -> str:
    """``detect`` / ``segment`` / 空（未知）。onnx 归一为 detect。"""
    mtype = normalize_model_type(model_type)
    if mtype == "segment":
        return "segment"
    if is_detect_pipeline_type(mtype):
        return "detect"
    return ""


def create_detector(
    model_path: str,
    *,
    model_type: str | None = "detect",
    class_names: list[str] | str | None = None,
    **kwargs: Any,
) -> ModelDetector:
    """按 ``model_type`` / 扩展名创建检测器，滑窗/merge 接口与 ``ModelDetector`` 一致。"""
    if is_onnx_detect_backend(model_type, model_path):
        from script.predict.model_detect_onnx import ModelDetectorOnnx

        return ModelDetectorOnnx(
            model_path,
            class_names=class_names,
            **kwargs,
        )
    return ModelDetector(model_path, **kwargs)
