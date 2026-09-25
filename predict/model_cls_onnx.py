#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""分类 ONNX 推理：加载导出的 ``.onnx`` + sidecar ``.meta.json``，接口与 ``ModelCls`` 对齐。"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torchvision import transforms

from script.predict.model_cls import ModelCls, CLS_INFER_KEEP_TOPK
from script.predict.model_infer_lock import model_infer_guard

log = logging.getLogger(__name__)

_DEFAULT_IMAGE_SIZE = 512
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_IMAGE_INPUT = "images"
_LOGITS_OUTPUT = "logits"
_ORT_DTYPE = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
}


def is_onnx_cls_path(model_path: str | None) -> bool:
    return str(model_path or "").strip().lower().endswith(".onnx")


def _ort_numpy_dtype(type_str: str) -> np.dtype:
    return np.dtype(_ORT_DTYPE.get(str(type_str), np.float32))


def _spatial_size_from_shape(shape) -> int:
    if shape is None or len(shape) < 4:
        return 0
    h, w = shape[2], shape[3]
    try:
        side = int(h)
    except (TypeError, ValueError):
        side = 0
    if side <= 0:
        try:
            side = int(w)
        except (TypeError, ValueError):
            side = 0
    return side if side > 0 else 0


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float64, copy=False)
    x = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(x)
    return (exp / np.sum(exp, axis=-1, keepdims=True)).astype(np.float32)


def load_cls_onnx_meta(onnx_path: Path) -> dict[str, Any]:
    meta_path = onnx_path.with_suffix(".meta.json")
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"分类 ONNX 缺少 sidecar: {meta_path}（与 {onnx_path.name} 同目录）"
        )
    try:
        obj = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"无法读取分类 ONNX sidecar {meta_path}: {e}") from e
    if not isinstance(obj, dict):
        raise ValueError(f"sidecar 不是 JSON 对象: {meta_path}")
    return obj


def _classes_from_meta(meta: dict[str, Any]) -> list[str]:
    raw_classes = meta.get("classes")
    if isinstance(raw_classes, (list, tuple)) and raw_classes:
        return [str(c) for c in raw_classes]
    raw_cti = meta.get("class_to_idx")
    if isinstance(raw_cti, dict) and raw_cti:
        pairs = [(str(k), int(v)) for k, v in raw_cti.items()]
        pairs.sort(key=lambda kv: kv[1])
        return [name for name, _ in pairs]
    raise ValueError("sidecar 缺少 classes / class_to_idx")


def _onnx_providers(device: str) -> list:
    import onnxruntime as ort

    avail = list(ort.get_available_providers())
    if device.startswith("cuda") and "CUDAExecutionProvider" in avail:
        idx = 0
        if ":" in device:
            try:
                idx = int(device.split(":")[-1])
            except ValueError:
                idx = 0
        return [
            ("CUDAExecutionProvider", {"device_id": idx}),
            "CPUExecutionProvider",
        ]
    if device.startswith("cuda"):
        log.warning("ONNX 分类请求 CUDA 但当前 Runtime 无 CUDA EP，回退 CPU: %s", avail)
    elif device == "mps":
        log.info("ONNX 分类在 MPS 上回退 CPU")
    return ["CPUExecutionProvider"]


class ModelClsOnnx:
    """导出分类 ONNX 的 Runtime 封装；预处理链与 ``ModelClsTimm`` 一致。"""

    def __init__(
        self,
        model_path: str | Path,
        device: str | None = None,
        pad_square: bool = False,
        gray_binarize: bool = False,
        pad_color_bgr: tuple[int, int, int] = (255, 255, 255),
        to_gray: bool = False,
        *,
        image_size: int | None = None,
    ):
        self.model_path = str(Path(model_path).expanduser().resolve())
        self._infer_task = "onnx-cls"
        onnx_path = Path(self.model_path)
        if not onnx_path.is_file():
            raise FileNotFoundError(f"分类 ONNX 不存在: {onnx_path}")
        meta = load_cls_onnx_meta(onnx_path)
        classes = _classes_from_meta(meta)
        self.names = {i: name for i, name in enumerate(classes)}
        self.class_to_idx = {name: i for i, name in enumerate(classes)}

        self.pad_square = bool(pad_square)
        self.gray_binarize = bool(gray_binarize)
        self.to_gray = bool(to_gray)
        self.pad_color_bgr = tuple(int(x) for x in pad_color_bgr)
        self.model_ch = 3
        if device is None:
            self.device = self._auto_detect_device()
        else:
            self.device = device

        cfg_size = int(image_size) if image_size is not None and int(image_size) > 0 else 0
        meta_size = int(meta["image_size"]) if meta.get("image_size") not in (None, "") else 0
        self.image_size = cfg_size or meta_size or _DEFAULT_IMAGE_SIZE
        mean = tuple(float(x) for x in (meta.get("mean") or _IMAGENET_MEAN))
        std = tuple(float(x) for x in (meta.get("std") or _IMAGENET_STD))
        if len(mean) != 3:
            mean = _IMAGENET_MEAN
        if len(std) != 3:
            std = _IMAGENET_STD
        self._input_name = str(meta.get("input_name") or _IMAGE_INPUT)
        self._output_name = str(meta.get("output_name") or _LOGITS_OUTPUT)
        self._session = self._build_session()
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
            "已加载 ONNX 分类: %s classes=%d imgsz=%d device=%s providers=%s",
            self.model_path,
            len(classes),
            self.image_size,
            self.device,
            self._session.get_providers(),
        )

    @staticmethod
    def _auto_detect_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _build_session(self):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "ONNX 分类需要 onnxruntime，请安装 script/requirments.txt 中的对应包"
            ) from exc
        providers = _onnx_providers(self.device)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(
            self.model_path, sess_options=so, providers=providers
        )
        inputs = {str(i.name): i for i in session.get_inputs()}
        if self._input_name not in inputs:
            if len(inputs) == 1:
                self._input_name = next(iter(inputs))
            else:
                raise ValueError(
                    f"ONNX 分类找不到输入 {self._input_name!r}，实际: {list(inputs)}"
                )
        spec = inputs[self._input_name]
        graph_size = _spatial_size_from_shape(spec.shape)
        if graph_size > 0:
            self.image_size = graph_size
        self._input_dtype = _ort_numpy_dtype(spec.type)
        out_names = [str(o.name) for o in session.get_outputs()]
        if self._output_name not in out_names:
            if len(out_names) == 1:
                self._output_name = out_names[0]
            else:
                raise ValueError(
                    f"ONNX 分类找不到输出 {self._output_name!r}，实际: {out_names}"
                )
        return session

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

    def _bgr_to_nchw(self, image_bgr: np.ndarray) -> np.ndarray:
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("空图像")
        if image_bgr.ndim == 2:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        from PIL import Image

        pil = Image.fromarray(rgb)
        tensor = self._transform(pil)
        return tensor.unsqueeze(0).numpy().astype(self._input_dtype, copy=False)

    def _infer_probs(self, batch_nchw: np.ndarray) -> np.ndarray:
        with model_infer_guard(self.model_path, task=self._infer_task):
            logits = self._session.run(
                [self._output_name], {self._input_name: batch_nchw}
            )[0]
        return _softmax_rows(np.asarray(logits))

    def _probs_to_result(self, probs: np.ndarray) -> dict[str, Any]:
        flat = np.asarray(probs).reshape(-1)
        k = min(int(CLS_INFER_KEEP_TOPK), int(flat.size))
        ids = np.argsort(-flat)[:k]
        topk: list[dict[str, Any]] = []
        for cid in ids:
            idx = int(cid)
            topk.append(
                {
                    "class_id": idx,
                    "class_name": self.names.get(idx, str(idx)),
                    "conf": float(flat[idx]),
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
        del device
        try:
            image = self._preprocess_for_predict(
                image, pad_square, gray_binarize, pad_color_bgr, to_gray
            )
            probs = self._infer_probs(self._bgr_to_nchw(image))
            return self._probs_to_result(probs[0])
        except Exception as e:
            log.error("ONNX 分类推理异常 %s: %s", self.model_path, e, exc_info=True)
            return None

    def predict_batch(
        self,
        images: list[np.ndarray],
        device: str | None = None,
        pad_square: bool | None = None,
        gray_binarize: bool | None = None,
        pad_color_bgr: tuple[int, int, int] | None = None,
        to_gray: bool | None = None,
        max_batch: int | None = None,
    ) -> list[dict | None]:
        del device
        if not images:
            return []
        chunk_limit = int(max_batch) if max_batch is not None and int(max_batch) > 0 else len(images)
        if chunk_limit <= 0:
            chunk_limit = len(images)
        out: list[dict | None] = []
        for start in range(0, len(images), max(1, chunk_limit)):
            chunk = images[start : start + max(1, chunk_limit)]
            out.extend(
                self._predict_batch_chunk(
                    chunk,
                    pad_square=pad_square,
                    gray_binarize=gray_binarize,
                    pad_color_bgr=pad_color_bgr,
                    to_gray=to_gray,
                )
            )
        return out[: len(images)]

    def _predict_batch_chunk(
        self,
        images: list[np.ndarray],
        *,
        pad_square: bool | None,
        gray_binarize: bool | None,
        pad_color_bgr: tuple[int, int, int] | None,
        to_gray: bool | None,
    ) -> list[dict | None]:
        if not images:
            return []
        try:
            valid_idx: list[int] = []
            tensors: list[np.ndarray] = []
            for i, image in enumerate(images):
                if image is None or getattr(image, "size", 1) == 0:
                    continue
                img = self._preprocess_for_predict(
                    image, pad_square, gray_binarize, pad_color_bgr, to_gray
                )
                tensors.append(self._bgr_to_nchw(img))
                valid_idx.append(i)
            results: list[dict | None] = [None] * len(images)
            if not tensors:
                return results
            batch = np.concatenate(tensors, axis=0)
            probs = self._infer_probs(batch)
            for j, i in enumerate(valid_idx):
                results[i] = self._probs_to_result(probs[j])
            return results
        except Exception as e:
            log.error("ONNX 分类批量推理异常 %s: %s", self.model_path, e, exc_info=True)
            return [None] * len(images)

    def predictTop2(
        self,
        image: np.ndarray,
        device: str | None = None,
        pad_square: bool | None = None,
        gray_binarize: bool | None = None,
        pad_color_bgr: tuple[int, int, int] | None = None,
        to_gray: bool | None = None,
    ) -> dict[str, Any] | None:
        result = self.predict(
            image,
            device=device,
            pad_square=pad_square,
            gray_binarize=gray_binarize,
            pad_color_bgr=pad_color_bgr,
            to_gray=to_gray,
        )
        if not result:
            return None
        topk = result.get("topk") or []
        if len(topk) < 2:
            return None
        return {"1": dict(topk[0]), "2": dict(topk[1])}
