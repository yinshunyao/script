#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""timm ConvNeXt（及同训练脚本导出的 checkpoint）分类推理，接口与 ``ModelCls`` 对齐。"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from script.predict.model_infer_lock import model_infer_guard

from script.predict.model_cls import ModelCls

log = logging.getLogger(__name__)


def _normalize_cls_checkpoint_path(path: str | Path) -> str:
    p = Path(path).expanduser()
    if not p.is_file():
        return str(p)
    try:
        return str(p.resolve())
    except OSError:
        return str(p)


@lru_cache(maxsize=64)
def _is_timm_cls_checkpoint_cached(path: str) -> bool:
    """按权重路径缓存 timm 判定（配置加载后不变，避免热路径重复 ``torch.load``）。"""
    p = Path(path)
    if not p.is_file():
        return False
    try:
        try:
            obj = torch.load(str(p), map_location="cpu", weights_only=False)
        except TypeError:
            obj = torch.load(str(p), map_location="cpu")
    except Exception as e:
        log.debug("非 timm checkpoint 或无法读取 %s: %s", p, e)
        return False
    return (
        isinstance(obj, dict)
        and "model_state" in obj
        and ("class_to_idx" in obj or "classes" in obj)
    )


def is_timm_cls_checkpoint(path: str | Path) -> bool:
    """是否为 ``train_cls_convnext`` 保存的 ``best_convnext.pt`` 类 checkpoint。"""
    return _is_timm_cls_checkpoint_cached(_normalize_cls_checkpoint_path(path))


def load_timm_cls_checkpoint(path: str | Path) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"分类 checkpoint 不存在: {p}")
    try:
        try:
            obj = torch.load(str(p), map_location="cpu", weights_only=False)
        except TypeError:
            obj = torch.load(str(p), map_location="cpu")
    except Exception as e:
        size_mb = p.stat().st_size / (1024 * 1024)
        raise ValueError(
            f"无法读取 timm 分类 checkpoint（文件可能损坏、不完整或非 PyTorch 格式）: {p} "
            f"(size={size_mb:.2f}MB, err={e})"
        ) from e
    if not isinstance(obj, dict) or "model_state" not in obj:
        raise ValueError(f"非 timm 分类 checkpoint（缺少 model_state）: {p}")
    return obj


class ModelClsTimm:
    """
    timm 分类推理；预处理链与 ``ModelCls`` 一致（灰度二值化 → 补方 → to_gray）。

    checkpoint 须含 ``model_state``、``class_to_idx``（或 ``classes``）、可选 ``timm_model`` / ``image_size``。
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str | None = None,
        pad_square: bool = False,
        gray_binarize: bool = False,
        pad_color_bgr: tuple[int, int, int] = (255, 255, 255),
        to_gray: bool = False,
        *,
        timm_model: str | None = None,
        image_size: int | None = None,
    ):
        import timm

        self.model_path = str(model_path)
        self._infer_task = "timm"
        ckpt = load_timm_cls_checkpoint(model_path)
        self.timm_model_name = str(timm_model or ckpt.get("timm_model") or "").strip()
        if not self.timm_model_name:
            raise ValueError(
                f"checkpoint 未含 timm_model，请在配置中设置 timm_model: {self.model_path}"
            )

        raw_cti = ckpt.get("class_to_idx")
        if isinstance(raw_cti, dict) and raw_cti:
            self.class_to_idx = {str(k): int(v) for k, v in raw_cti.items()}
        elif isinstance(ckpt.get("classes"), (list, tuple)) and ckpt["classes"]:
            classes = [str(c) for c in ckpt["classes"]]
            self.class_to_idx = {name: i for i, name in enumerate(classes)}
        else:
            raise ValueError(f"checkpoint 缺少 class_to_idx / classes: {self.model_path}")

        self.names = {idx: name for name, idx in self.class_to_idx.items()}
        self.image_size = int(image_size or ckpt.get("image_size") or 224)
        self.pad_square = bool(pad_square)
        self.gray_binarize = bool(gray_binarize)
        self.to_gray = bool(to_gray)
        self.pad_color_bgr = tuple(int(x) for x in pad_color_bgr)
        self.model_ch = 3

        if device is None:
            self.device = self._auto_detect_device()
        else:
            self.device = device

        num_classes = len(self.class_to_idx)
        self.model = timm.create_model(
            self.timm_model_name, pretrained=False, num_classes=num_classes
        )
        state = ckpt["model_state"]
        if isinstance(state, dict) and state and next(iter(state)).startswith("module."):
            state = {k[len("module.") :]: v for k, v in state.items()}
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.model.to(self.device)

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        sz = self.image_size
        self._transform = transforms.Compose(
            [
                transforms.Resize(int(sz * 256 / 224)),
                transforms.CenterCrop(sz),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        log.info(
            "已加载 timm 分类: %s arch=%s classes=%d imgsz=%d device=%s",
            self.model_path,
            self.timm_model_name,
            num_classes,
            sz,
            self.device,
        )

    @staticmethod
    def _auto_detect_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _preprocess_for_predict(
        self,
        image: np.ndarray,
        pad_square: bool | None,
        gray_binarize: bool | None,
        pad_color_bgr: tuple[int, int, int] | None,
        to_gray: bool | None,
    ) -> np.ndarray:
        use_bin = self.gray_binarize if gray_binarize is None else bool(gray_binarize)
        if use_bin:
            image = ModelCls.bgr_gray_clahe_otsu_to_bgr(image)
        use_pad = self.pad_square if pad_square is None else bool(pad_square)
        if use_pad:
            color = self.pad_color_bgr if pad_color_bgr is None else tuple(int(x) for x in pad_color_bgr)
            image = ModelCls.pad_bgr_to_square(image, pad_value=color)
        use_gray = self.to_gray if to_gray is None else bool(to_gray)
        if use_gray:
            image = ModelCls.bgr_to_gray_3ch(image)
        return image

    def _bgr_to_tensor(self, image_bgr: np.ndarray) -> torch.Tensor:
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("空图像")
        if image_bgr.ndim == 2:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        from PIL import Image

        pil = Image.fromarray(rgb)
        return self._transform(pil)

    def _probs_to_result(self, probs: torch.Tensor) -> dict[str, Any]:
        confs, ids = torch.topk(probs, k=min(5, probs.numel()))
        topk: list[dict[str, Any]] = []
        for i in range(ids.numel()):
            cid = int(ids[i].item())
            topk.append(
                {
                    "class_id": cid,
                    "class_name": self.names.get(cid, str(cid)),
                    "conf": float(confs[i].item()),
                }
            )
        top1 = topk[0]
        return {
            "class_id": top1["class_id"],
            "class_name": top1["class_name"],
            "conf": top1["conf"],
            "top3": topk[:3],
            "topk": topk,
        }

    def predict(
        self,
        image: np.ndarray,
        device: str | None = None,
        pad_square: bool | None = None,
        gray_binarize: bool | None = None,
        pad_color_bgr: tuple[int, int, int] | None = None,
        to_gray: bool | None = None,
    ) -> dict[str, Any] | None:
        try:
            image = self._preprocess_for_predict(
                image, pad_square, gray_binarize, pad_color_bgr, to_gray
            )
            tensor = self._bgr_to_tensor(image).unsqueeze(0)
            dev = device or self.device
            tensor = tensor.to(dev)
            with torch.no_grad():
                with model_infer_guard(self.model_path, task=self._infer_task):
                    logits = self.model(tensor)
                probs = F.softmax(logits, dim=1)[0].detach().cpu()
            return self._probs_to_result(probs)
        except Exception as e:
            log.error("timm 分类推理异常 %s: %s", self.model_path, e, exc_info=True)
            return None

    def predictTop2(
        self,
        image: np.ndarray,
        device: str | None = None,
        pad_square: bool | None = None,
        gray_binarize: bool | None = None,
        pad_color_bgr: tuple[int, int, int] | None = None,
        to_gray: bool | None = None,
    ) -> dict[str, Any] | None:
        try:
            image = self._preprocess_for_predict(
                image, pad_square, gray_binarize, pad_color_bgr, to_gray
            )
            tensor = self._bgr_to_tensor(image).unsqueeze(0)
            dev = device or self.device
            tensor = tensor.to(dev)
            with torch.no_grad():
                with model_infer_guard(self.model_path, task=self._infer_task):
                    logits = self.model(tensor)
                probs = F.softmax(logits, dim=1)[0].detach().cpu()
            confs, ids = torch.topk(probs, k=min(2, probs.numel()))
            if ids.numel() < 2:
                return None
            return {
                "1": {
                    "class_id": int(ids[0].item()),
                    "class_name": self.names.get(int(ids[0].item()), str(ids[0].item())),
                    "conf": float(confs[0].item()),
                },
                "2": {
                    "class_id": int(ids[1].item()),
                    "class_name": self.names.get(int(ids[1].item()), str(ids[1].item())),
                    "conf": float(confs[1].item()),
                },
            }
        except Exception as e:
            log.error("timm 分类 Top2 推理异常 %s: %s", self.model_path, e, exc_info=True)
            return None
