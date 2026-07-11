#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SAM2 图像分割推理封装（点提示 → mask）。"""
from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

# 多目标 mask 叠加色（RGB）
OBJECT_COLORS: list[tuple[int, int, int]] = [
    (30, 180, 255),
    (255, 120, 50),
    (80, 220, 100),
    (220, 80, 200),
    (255, 220, 60),
    (140, 120, 255),
    (255, 100, 140),
    (60, 200, 200),
]


def object_color_rgb(object_id: int) -> tuple[int, int, int]:
    return OBJECT_COLORS[(max(object_id, 1) - 1) % len(OBJECT_COLORS)]

MODEL_PRESETS: dict[str, dict[str, str]] = {
    "sam2.1_tiny": {
        "config": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "checkpoint": "sam2.1_hiera_tiny.pt",
        "label": "SAM 2.1 Hiera Tiny（默认，速度快）",
    },
    "sam2.1_small": {
        "config": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "checkpoint": "sam2.1_hiera_small.pt",
        "label": "SAM 2.1 Hiera Small",
    },
    "sam2.1_base_plus": {
        "config": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "checkpoint": "sam2.1_hiera_base_plus.pt",
        "label": "SAM 2.1 Hiera Base+",
    },
    "sam2.1_large": {
        "config": "configs/sam2.1/sam2.1_hiera_l.yaml",
        "checkpoint": "sam2.1_hiera_large.pt",
        "label": "SAM 2.1 Hiera Large（精度高，较慢）",
    },
}


def resolve_device(device: str | None = None) -> torch.device:
    if device and device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def infer_model_key_from_checkpoint(checkpoint_path: Path) -> str | None:
    """根据权重文件名推断 model_key。"""
    name = checkpoint_path.name
    for key, preset in MODEL_PRESETS.items():
        if preset["checkpoint"] == name:
            return key
    return None


def resolve_checkpoint(checkpoint_path: Path, filename: str) -> Path:
    """
    解析权重路径。

    ``checkpoint_path`` 可以是：
    - 直接的 ``.pt`` 文件路径；
    - 包含权重文件的目录（按 preset 文件名查找）。
    """
    path = checkpoint_path.expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() != ".pt":
            raise ValueError(f"权重路径须为 .pt 文件: {path}")
        return path
    if path.is_dir():
        candidate = path / filename
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(
            f"在目录 {path} 中未找到权重 {filename!r}。"
            f"请确认文件名，或直接传入 .pt 文件路径。"
        )
    raise FileNotFoundError(f"权重路径不存在: {path}")


@contextlib.contextmanager
def _inference_context(device: torch.device):
    if device.type == "cuda":
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            yield
    else:
        with torch.inference_mode():
            yield


class Sam2Engine:
    """懒加载 SAM2 模型，缓存当前图像 embedding。"""

    def __init__(
        self,
        *,
        model_key: str = "sam2.1_tiny",
        checkpoint_path: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        if model_key not in MODEL_PRESETS:
            raise ValueError(f"未知模型: {model_key!r}，可选: {list(MODEL_PRESETS)}")
        self.model_key = model_key
        self.checkpoint_path = Path(
            checkpoint_path or Path(__file__).parent / "checkpoints"
        )
        self.device = resolve_device(device)
        self._predictor: Any = None
        self._loaded_key: str | None = None
        self._loaded_ckpt: Path | None = None
        self._image_rgb: np.ndarray | None = None

    def set_checkpoint_path(self, checkpoint_path: str | Path) -> None:
        path = Path(checkpoint_path)
        if path != self.checkpoint_path:
            self.checkpoint_path = path
            self._predictor = None
            self._loaded_key = None
            self._loaded_ckpt = None
            self._image_rgb = None

    def _ensure_predictor(self) -> None:
        preset = MODEL_PRESETS[self.model_key]
        ckpt_path = resolve_checkpoint(self.checkpoint_path, preset["checkpoint"])
        if (
            self._predictor is not None
            and self._loaded_key == self.model_key
            and self._loaded_ckpt == ckpt_path
        ):
            return

        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        logger.info(
            "加载 SAM2 模型 %s，device=%s，checkpoint=%s",
            self.model_key,
            self.device.type,
            ckpt_path,
        )
        sam_model = build_sam2(
            preset["config"],
            ckpt_path=str(ckpt_path),
            device=self.device.type,
        )
        self._predictor = SAM2ImagePredictor(sam_model)
        self._loaded_key = self.model_key
        self._loaded_ckpt = ckpt_path
        self._image_rgb = None

    def set_model(self, model_key: str) -> None:
        if model_key != self.model_key:
            self.model_key = model_key
            self._predictor = None
            self._loaded_key = None
            self._image_rgb = None

    def set_image(self, image_rgb: np.ndarray | None) -> None:
        if image_rgb is None:
            self._image_rgb = None
            return
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("输入图像须为 HWC RGB 格式")
        self._ensure_predictor()
        image_rgb = np.ascontiguousarray(image_rgb, dtype=np.uint8)
        with _inference_context(self.device):
            self._predictor.set_image(image_rgb)
        self._image_rgb = image_rgb

    def predict(
        self,
        points: list[list[int]],
        labels: list[int],
    ) -> tuple[np.ndarray, float]:
        if self._image_rgb is None:
            raise RuntimeError("请先上传图片")
        if not points:
            raise RuntimeError("请先在图片上点击添加提示点")

        point_coords = np.array(points, dtype=np.float32)
        point_labels = np.array(labels, dtype=np.int32)
        multimask = len(points) == 1

        with _inference_context(self.device):
            masks, scores, _ = self._predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=multimask,
            )

        best_idx = int(np.argmax(scores))
        return masks[best_idx], float(scores[best_idx])

    def predict_objects(
        self,
        object_prompts: dict[int, tuple[list[list[int]], list[int]]],
    ) -> dict[int, tuple[np.ndarray, float]]:
        """按目标分组推理，每个目标独立生成 mask。"""
        results: dict[int, tuple[np.ndarray, float]] = {}
        for obj_id in sorted(object_prompts):
            points, labels = object_prompts[obj_id]
            if not points:
                continue
            results[obj_id] = self.predict(points, labels)
        return results

    @staticmethod
    def group_prompts(
        points: list[list[int]],
        labels: list[int],
        object_ids: list[int],
    ) -> dict[int, tuple[list[list[int]], list[int]]]:
        grouped: dict[int, tuple[list[list[int]], list[int]]] = {}
        for pt, lb, obj_id in zip(points, labels, object_ids):
            pts, lbs = grouped.setdefault(int(obj_id), ([], []))
            pts.append(pt)
            lbs.append(lb)
        return grouped

    @staticmethod
    def draw_points(
        image_rgb: np.ndarray,
        points: list[list[int]],
        labels: list[int],
        object_ids: list[int] | None = None,
    ) -> np.ndarray:
        canvas = image_rgb.copy()
        if object_ids is None:
            object_ids = [1] * len(points)
        multi_object = len(set(object_ids)) > 1
        for (x, y), label, obj_id in zip(points, labels, object_ids):
            if label == 1:
                color = object_color_rgb(obj_id)
            else:
                color = (231, 76, 60)
            cv2.circle(canvas, (int(x), int(y)), 7, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, (int(x), int(y)), 8, (255, 255, 255), 1, lineType=cv2.LINE_AA)
            if multi_object:
                cv2.putText(
                    canvas,
                    str(obj_id),
                    (int(x) + 8, int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    lineType=cv2.LINE_AA,
                )
        return canvas

    @staticmethod
    def overlay_mask(
        image_rgb: np.ndarray,
        mask: np.ndarray,
        *,
        alpha: float = 0.45,
        color: tuple[int, int, int] | None = None,
    ) -> np.ndarray:
        overlay = image_rgb.copy().astype(np.float32)
        mask_bool = mask.astype(bool)
        tint = np.array(color or (30, 180, 255), dtype=np.float32)
        overlay[mask_bool] = overlay[mask_bool] * (1.0 - alpha) + tint * alpha
        return overlay.astype(np.uint8)

    @staticmethod
    def overlay_masks(
        image_rgb: np.ndarray,
        masks_by_object: dict[int, np.ndarray],
        *,
        alpha: float = 0.45,
    ) -> np.ndarray:
        overlay = image_rgb.copy().astype(np.float32)
        for obj_id, mask in sorted(masks_by_object.items()):
            mask_bool = mask.astype(bool)
            tint = np.array(object_color_rgb(obj_id), dtype=np.float32)
            overlay[mask_bool] = overlay[mask_bool] * (1.0 - alpha) + tint * alpha
        return overlay.astype(np.uint8)
