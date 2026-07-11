#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Detail  : 读取分类/检测/分割模型元信息（YOLO 与 timm ConvNeXt checkpoint）。

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

_FILE = Path(__file__).resolve()
_INSECT_ROOT = _FILE.parents[2]

# torchvision / torch._dynamo 在 import 时会调用 os.getcwd()；若 shell 所在目录已删除会报错。
try:
    os.getcwd()
except FileNotFoundError:
    os.chdir(_FILE.parent)

if str(_INSECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_INSECT_ROOT))

import torch

from script.predict.model_channel import detect_model_input_channels
from script.predict.model_cls_timm import is_timm_cls_checkpoint, load_timm_cls_checkpoint

log = logging.getLogger(__name__)

_YOLO_TASK_BY_MODEL_CLS = {
    "ClassificationModel": "classify",
    "DetectionModel": "detect",
    "SegmentationModel": "segment",
}


def _normalize_names(names: Any) -> dict[int, str]:
    if names is None:
        return {}
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, (list, tuple)):
        return {i: str(v) for i, v in enumerate(names)}
    return {}


def _pick_yolo_inner_model(ckpt: dict[str, Any]) -> Any | None:
    for key in ("model", "ema"):
        obj = ckpt.get(key)
        if obj is not None:
            return obj
    return None


def _task_from_inner_model(inner: Any) -> str | None:
    cls_name = type(inner).__name__
    return _YOLO_TASK_BY_MODEL_CLS.get(cls_name)


def _channels_from_yaml(yaml: dict[str, Any] | None, default: int = 3) -> int:
    if not isinstance(yaml, dict):
        return default
    for key in ("channels", "ch"):
        raw = yaml.get(key)
        if raw:
            ch = int(raw)
            if ch > 0:
                return ch
    return default


def _imgsz_from_train_args(train_args: dict[str, Any] | None) -> int | None:
    if not isinstance(train_args, dict):
        return None
    raw = train_args.get("imgsz")
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        if not raw:
            return None
        return int(raw[0])
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return None
    return val if val > 0 else None


def _load_yolo_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def _get_timm_model_info(path: Path) -> dict[str, Any]:
    ckpt = load_timm_cls_checkpoint(path)
    raw_cti = ckpt.get("class_to_idx")
    if isinstance(raw_cti, dict) and raw_cti:
        class_to_idx = {str(k): int(v) for k, v in raw_cti.items()}
    elif isinstance(ckpt.get("classes"), (list, tuple)) and ckpt["classes"]:
        classes = [str(c) for c in ckpt["classes"]]
        class_to_idx = {name: i for i, name in enumerate(classes)}
    else:
        class_to_idx = {}

    names = {idx: name for name, idx in class_to_idx.items()}
    image_size = ckpt.get("image_size")
    image_size_int = int(image_size) if image_size else None

    train_args = {
        k: ckpt.get(k)
        for k in ("epoch", "val_acc", "timm_model", "image_size")
        if ckpt.get(k) is not None
    }

    return {
        "path": str(path),
        "exists": True,
        "file_size_bytes": path.stat().st_size,
        "backend": "convnext",
        "task": "classify",
        "input_channels": 3,
        "imgsz": image_size_int,
        "image_size": image_size_int,
        "num_classes": len(names),
        "names": names,
        "timm_model": str(ckpt.get("timm_model") or "").strip() or None,
        "train_date": None,
        "ultralytics_version": None,
        "yaml_file": None,
        "train_args": train_args,
    }


def _get_yolo_model_info_from_ckpt(path: Path, ckpt: dict[str, Any]) -> dict[str, Any] | None:
    inner = _pick_yolo_inner_model(ckpt)
    train_args = ckpt.get("train_args") if isinstance(ckpt.get("train_args"), dict) else {}

    task = _task_from_inner_model(inner) if inner is not None else None
    if not task:
        raw_task = str(train_args.get("task") or "").strip().lower()
        if raw_task in ("classify", "detect", "segment"):
            task = raw_task

    names: dict[int, str] = {}
    yaml: dict[str, Any] | None = None
    yaml_file: str | None = None
    if inner is not None:
        names = _normalize_names(getattr(inner, "names", None))
        raw_yaml = getattr(inner, "yaml", None)
        if isinstance(raw_yaml, dict):
            yaml = raw_yaml
            yaml_file = raw_yaml.get("yaml_file")

    input_channels = _channels_from_yaml(yaml)
    imgsz = _imgsz_from_train_args(train_args)
    if imgsz is None and isinstance(yaml, dict) and yaml.get("imgsz"):
        try:
            imgsz = int(yaml["imgsz"])
        except (TypeError, ValueError):
            imgsz = None

    if inner is None and not task and not names:
        return None

    train_args_out = {
        k: train_args.get(k)
        for k in ("task", "imgsz", "data", "epochs", "batch", "model")
        if train_args.get(k) is not None
    }

    return {
        "path": str(path),
        "exists": True,
        "file_size_bytes": path.stat().st_size,
        "backend": "yolo",
        "task": task,
        "input_channels": input_channels,
        "imgsz": imgsz,
        "image_size": imgsz,
        "num_classes": len(names) if names else int((yaml or {}).get("nc") or 0),
        "names": names,
        "timm_model": None,
        "train_date": ckpt.get("date"),
        "ultralytics_version": ckpt.get("version"),
        "yaml_file": yaml_file,
        "train_args": train_args_out,
    }


def _get_yolo_model_info_via_load(path: Path, task: str | None = None) -> dict[str, Any]:
    from ultralytics import YOLO

    yolo_task = task
    if yolo_task is None:
        yolo_task = "classify"
    yolo = YOLO(str(path), task=yolo_task)
    names = _normalize_names(yolo.names)
    model_ch = detect_model_input_channels(yolo)
    inner = getattr(yolo, "model", None)
    yaml = getattr(inner, "yaml", None) if inner is not None else None
    yaml_file = yaml.get("yaml_file") if isinstance(yaml, dict) else None

    ckpt = _load_yolo_checkpoint(path)
    train_args = ckpt.get("train_args") if isinstance(ckpt.get("train_args"), dict) else {}
    imgsz = _imgsz_from_train_args(train_args)

    return {
        "path": str(path),
        "exists": True,
        "file_size_bytes": path.stat().st_size,
        "backend": "yolo",
        "task": str(yolo.task or task or train_args.get("task") or "").strip() or None,
        "input_channels": model_ch,
        "imgsz": imgsz,
        "image_size": imgsz,
        "num_classes": len(names),
        "names": names,
        "timm_model": None,
        "train_date": ckpt.get("date"),
        "ultralytics_version": ckpt.get("version"),
        "yaml_file": yaml_file,
        "train_args": {
            k: train_args.get(k)
            for k in ("task", "imgsz", "data", "epochs", "batch", "model")
            if train_args.get(k) is not None
        },
    }


def get_model_info(
    model_path: str | Path,
    *,
    task: str | None = None,
) -> dict[str, Any]:
    """
    读取模型元信息，自动识别 YOLO（classify/detect/segment）与 timm 分类 checkpoint。

    返回字段含：backend、task、input_channels、imgsz、num_classes、names、train_args 等。
    """
    path = Path(model_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"模型文件不存在: {path}")

    if is_timm_cls_checkpoint(path):
        return _get_timm_model_info(path)

    ckpt = _load_yolo_checkpoint(path)
    if not isinstance(ckpt, dict):
        raise ValueError(f"无法解析模型 checkpoint: {path}")

    info = _get_yolo_model_info_from_ckpt(path, ckpt)
    if info is not None and info.get("names") and info.get("task"):
        inner = _pick_yolo_inner_model(ckpt)
        if inner is not None:
            try:
                from ultralytics import YOLO

                yolo = YOLO(str(path), task=info["task"])
                info["input_channels"] = detect_model_input_channels(yolo)
            except Exception:
                log.debug("YOLO 加载失败，保留 yaml 通道数", exc_info=True)
        return info

    return _get_yolo_model_info_via_load(path, task=task)


def format_model_info(info: dict[str, Any]) -> str:
    """将 ``get_model_info`` 结果格式化为可读文本。"""
    lines = [
        f"path: {info.get('path')}",
        f"backend: {info.get('backend')}",
        f"task: {info.get('task')}",
        f"input_channels: {info.get('input_channels')}",
        f"imgsz: {info.get('imgsz')}",
        f"num_classes: {info.get('num_classes')}",
    ]
    if info.get("timm_model"):
        lines.append(f"timm_model: {info.get('timm_model')}")
    if info.get("yaml_file"):
        lines.append(f"yaml_file: {info.get('yaml_file')}")
    if info.get("train_date"):
        lines.append(f"train_date: {info.get('train_date')}")
    if info.get("ultralytics_version"):
        lines.append(f"ultralytics_version: {info.get('ultralytics_version')}")
    if info.get("file_size_bytes"):
        mb = info["file_size_bytes"] / (1024 * 1024)
        lines.append(f"file_size_mb: {mb:.2f}")

    names = info.get("names") or {}
    if names:
        lines.append("names:")
        for idx in sorted(names):
            lines.append(f"  {idx}: {names[idx]}")

    train_args = info.get("train_args") or {}
    if train_args:
        lines.append("train_args:")
        for key, val in train_args.items():
            lines.append(f"  {key}: {val}")
    return "\n".join(lines)


if __name__ == "__main__":
    # /Users/shunyaoyin/miniconda310/miniconda3/envs/yolo11/bin/python3 /Users/shunyaoyin/Documents/code/ai-company/insect/script/predict/get_model_info.py
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    MODEL_PATH = "/Volumes/shunyao-h1/models-test/small-cls-v3.0/daofeishi-cls-3.0.3.pt"
    TASK = None  # 可选：classify / detect / segment；None 表示自动识别
    AS_JSON = False

    result = get_model_info(MODEL_PATH, task=TASK)
    if AS_JSON:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    else:
        print(format_model_info(result))
