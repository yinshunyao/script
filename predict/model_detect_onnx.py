#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ONNX 检测推理：加载 ``.onnx``，输出与 ``ModelDetector._predict`` 对齐的框。"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from script.predict.model_channel import (
    preprocess_yolo_input,
    scale_xyxy,
    yolo_input_coord_scale,
)
from script.predict.model_detect import (
    ModelDetector,
    coerce_detect_class_names,
    resolve_detect_class_names,
)
from script.predict.model_infer_lock import model_infer_guard

log = logging.getLogger(__name__)

_DEFAULT_CLASS_NAMES = ["insect"]
_IMAGE_INPUT = "images"
_SIZE_INPUT = "orig_target_sizes"
_OUTPUT_NAMES = ("labels", "boxes", "scores")

_ORT_DTYPE = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
}

_ENGINE_CACHE: dict[tuple[str, str], "OnnxDetectEngine"] = {}


def is_onnx_model_path(model_path: str | None) -> bool:
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


class OnnxDetectEngine:
    """进程内可缓存的 ONNX 检测会话。"""

    def __init__(
        self,
        model_path: str,
        *,
        class_names: list[str] | str | None = None,
        device: str = "cpu",
    ) -> None:
        self.model_path = str(Path(model_path).expanduser().resolve())
        self.device = str(device)
        names = resolve_detect_class_names(
            class_names, relative_to=self.model_path
        ) or list(_DEFAULT_CLASS_NAMES)
        self.names = coerce_detect_class_names(names)
        self.eval_spatial_size = 640
        self._image_name = _IMAGE_INPUT
        self._size_name = _SIZE_INPUT
        self._image_dtype = np.dtype(np.float32)
        self._size_dtype = np.dtype(np.float32)
        self._output_names: list[str] | None = None
        self._session = self._build()

    def _providers(self) -> list:
        import onnxruntime as ort

        avail = list(ort.get_available_providers())
        device = self.device
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
            log.warning("ONNX 请求 CUDA 但当前 Runtime 无 CUDA EP，回退 CPU: %s", avail)
        elif device == "mps":
            log.info("ONNX 在 MPS 上回退 CPU")
        return ["CPUExecutionProvider"]

    def _bind_io(self, session) -> None:
        inputs = {str(i.name): i for i in session.get_inputs()}
        if _IMAGE_INPUT not in inputs or _SIZE_INPUT not in inputs:
            raise ValueError(
                "ONNX 检测需要输入 %s 与 %s，实际: %s"
                % (_IMAGE_INPUT, _SIZE_INPUT, list(inputs))
            )
        self._image_name = _IMAGE_INPUT
        self._size_name = _SIZE_INPUT
        self._image_dtype = _ort_numpy_dtype(inputs[_IMAGE_INPUT].type)
        self._size_dtype = _ort_numpy_dtype(inputs[_SIZE_INPUT].type)
        side = _spatial_size_from_shape(inputs[_IMAGE_INPUT].shape)
        if side > 0:
            self.eval_spatial_size = side
        out_names = [str(o.name) for o in session.get_outputs()]
        if all(n in out_names for n in _OUTPUT_NAMES):
            self._output_names = list(_OUTPUT_NAMES)
        elif len(out_names) >= 3:
            self._output_names = out_names[:3]
        else:
            raise ValueError(
                "ONNX 检测需要输出 labels/boxes/scores，实际: %s" % out_names
            )

    def _build(self):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "ONNX 检测需要 onnxruntime，请安装 script/requirments.txt 中的对应包"
            ) from exc
        if not Path(self.model_path).is_file():
            raise FileNotFoundError(f"ONNX 权重不存在: {self.model_path}")
        providers = self._providers()
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(
            self.model_path, sess_options=so, providers=providers
        )
        self._bind_io(session)
        log.info(
            "加载 ONNX 检测: path=%s device=%s providers=%s imgsz=%s names=%s",
            self.model_path,
            self.device,
            session.get_providers(),
            self.eval_spatial_size,
            self.names,
        )
        return session

    def preprocess_bgr(self, image: np.ndarray, *, imgsz: int) -> np.ndarray:
        """BGR HWC uint8 → ``1x3xS×S``，值域 0~1。"""
        work = image
        if work.ndim == 2:
            work = cv2.cvtColor(work, cv2.COLOR_GRAY2BGR)
        elif work.ndim == 3 and work.shape[2] == 1:
            work = cv2.cvtColor(work[:, :, 0], cv2.COLOR_GRAY2BGR)
        rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
        side = max(1, int(imgsz))
        if rgb.shape[0] != side or rgb.shape[1] != side:
            rgb = cv2.resize(rgb, (side, side), interpolation=cv2.INTER_LINEAR)
        ten = np.ascontiguousarray(rgb).transpose(2, 0, 1).astype(np.float32) / 255.0
        return ten.astype(self._image_dtype, copy=False)

    def infer(
        self,
        images: np.ndarray,
        orig_sizes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        feeds = {
            self._image_name: np.ascontiguousarray(images, dtype=self._image_dtype),
            self._size_name: np.ascontiguousarray(orig_sizes, dtype=self._size_dtype),
        }
        labels, boxes, scores = self._session.run(self._output_names, feeds)
        return np.asarray(labels), np.asarray(boxes), np.asarray(scores)


def _onnx_class_names_cache_token(class_names: list[str] | str | None) -> str:
    if class_names is None:
        return ""
    if isinstance(class_names, str):
        return class_names.strip()
    return ",".join(str(x) for x in class_names)


def get_cached_onnx_engine(
    model_path: str,
    *,
    class_names: list[str] | str | None = None,
    device: str = "cpu",
) -> OnnxDetectEngine:
    path = str(Path(model_path).expanduser().resolve())
    key = (path, str(device), _onnx_class_names_cache_token(class_names))
    engine = _ENGINE_CACHE.get(key)
    if engine is not None:
        return engine
    engine = OnnxDetectEngine(path, class_names=class_names, device=device)
    _ENGINE_CACHE[key] = engine
    return engine


def clear_onnx_detect_cache() -> None:
    if _ENGINE_CACHE:
        log.info("释放 ONNX 检测缓存 %d 项", len(_ENGINE_CACHE))
    _ENGINE_CACHE.clear()


class ModelDetectorOnnx(ModelDetector):
    """复用滑窗/merge，仅替换 YOLO 前向为 ONNX。"""

    def __init__(
        self,
        model_path,
        conf_thresh=0.5,
        conf_merge=0.3,
        conf_merge_draw=0.01,
        iou_threshold=0.3,
        ior_threshold=0.5,
        device=None,
        augment=False,
        half=False,
        *,
        nms_iou: float | None = None,
        max_det: int | None = None,
        nms_agnostic: bool | None = None,
        gray_contrast_enhance: bool = False,
        gray_clahe_clip: float = 2.0,
        gray_clahe_tile: int = 8,
        class_names: list[str] | str | None = None,
    ):
        self._class_names_cfg = class_names
        self.engine: OnnxDetectEngine | None = None
        self.eval_spatial_size = 640
        super().__init__(
            model_path,
            conf_thresh=conf_thresh,
            conf_merge=conf_merge,
            conf_merge_draw=conf_merge_draw,
            iou_threshold=iou_threshold,
            ior_threshold=ior_threshold,
            device=device,
            augment=augment,
            half=half,
            nms_iou=nms_iou,
            max_det=max_det,
            nms_agnostic=nms_agnostic,
            gray_contrast_enhance=gray_contrast_enhance,
            gray_clahe_clip=gray_clahe_clip,
            gray_clahe_tile=gray_clahe_tile,
        )
        if self.augment:
            log.info("ONNX 检测忽略 YOLO augment/TTA: %s", model_path)

    def _load_weights(self, model_path) -> None:
        if not is_onnx_model_path(model_path):
            raise ValueError(f"ONNX 检测需要 .onnx 权重: {model_path}")
        if not Path(str(model_path)).expanduser().is_file():
            raise FileNotFoundError(f"ONNX 权重不存在: {model_path}")
        self.engine = get_cached_onnx_engine(
            str(model_path),
            class_names=self._class_names_cfg,
            device=self.device,
        )
        self.model_ch = 3
        self.names = dict(self.engine.names)
        self.eval_spatial_size = int(self.engine.eval_spatial_size)
        self.model = SimpleNamespace(names=self.names)
        log.info(
            "ONNX 检测已就绪 eval_spatial_size=%d names=%s: %s",
            self.eval_spatial_size,
            self.names,
            model_path,
        )

    def _resolve_infer_imgsz(self, imgsz: int | None) -> int:
        raw = int(imgsz or 0)
        return raw if raw > 0 else int(self.eval_spatial_size or 640)

    def _prepare_inputs(
        self,
        images: list[np.ndarray],
        imgsz: int,
    ) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float]]]:
        assert self.engine is not None
        tensors: list[np.ndarray] = []
        orig_wh: list[list[int]] = []
        scales: list[tuple[float, float]] = []
        for image in images:
            source_shape = image.shape[:2]
            work = preprocess_yolo_input(
                image,
                3,
                gray_contrast_enhance=self.gray_contrast_enhance,
                clahe_clip=self.gray_clahe_clip,
                clahe_tile=self.gray_clahe_tile,
                target_imgsz=imgsz if self.gray_contrast_enhance else 0,
            )
            if work is None:
                work = image
            oh, ow = int(work.shape[0]), int(work.shape[1])
            tensors.append(self.engine.preprocess_bgr(work, imgsz=imgsz))
            orig_wh.append([ow, oh])
            scales.append(yolo_input_coord_scale(source_shape, work))
        batch = np.stack(tensors, axis=0)
        orig = np.asarray(orig_wh, dtype=np.float32)
        return batch, orig, scales

    def _decode_outputs(
        self,
        labels: np.ndarray,
        boxes: np.ndarray,
        scores: np.ndarray,
        detect_ids: list[str],
        scales: list[tuple[float, float]],
        conf: float,
        max_det: int | None,
    ) -> list[list]:
        batch_out: list[list] = []
        limit = int(max_det) if max_det is not None and int(max_det) > 0 else 0
        n = int(labels.shape[0]) if labels.ndim >= 1 else 0
        for idx in range(n):
            detect_id = detect_ids[idx] if idx < len(detect_ids) else detect_ids[-1]
            sx, sy = scales[idx] if idx < len(scales) else (1.0, 1.0)
            rows: list = []
            lab = np.asarray(labels[idx]).reshape(-1)
            box = np.asarray(boxes[idx]).reshape(-1, 4)
            sco = np.asarray(scores[idx]).reshape(-1)
            keep = sco > float(conf)
            if not bool(np.any(keep)):
                batch_out.append(rows)
                continue
            lab = lab[keep]
            box = box[keep]
            sco = sco[keep]
            order = np.argsort(-sco)
            if limit:
                order = order[:limit]
            for j in order.tolist():
                x1, y1, x2, y2 = (float(v) for v in box[j].tolist())
                if sx != 1.0 or sy != 1.0:
                    x1, y1, x2, y2 = scale_xyxy(x1, y1, x2, y2, sx, sy)
                else:
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                rows.append(
                    [
                        x1,
                        y1,
                        x2,
                        y2,
                        float(sco[j]),
                        int(lab[j]),
                        detect_id,
                    ]
                )
            batch_out.append(rows)
        return batch_out

    def _predict(
        self,
        image,
        conf=0.01,
        detect_id: str = "0-0",
        device=None,
        imgsz=0,
        overlap=0,
        *,
        nms_iou: float | None = None,
        max_det: int | None = None,
        nms_agnostic: bool | None = None,
        augment: bool | None = None,
    ):
        batch = self._predict_batch(
            [image],
            [detect_id],
            conf=conf,
            device=device,
            imgsz=imgsz,
            overlap=overlap,
            nms_iou=nms_iou,
            max_det=max_det,
            nms_agnostic=nms_agnostic,
            augment=augment,
        )
        return batch[0] if batch else []

    def _predict_batch(
        self,
        images: list[np.ndarray],
        detect_ids: list[str],
        conf=0.01,
        device=None,
        imgsz=0,
        overlap=0,
        *,
        nms_iou: float | None = None,
        max_det: int | None = None,
        nms_agnostic: bool | None = None,
        augment: bool | None = None,
    ) -> list[list]:
        del overlap, nms_iou, nms_agnostic, augment, device
        if not images:
            return []
        assert self.engine is not None
        use_imgsz = self._resolve_infer_imgsz(imgsz)
        use_max = max_det if max_det is not None else self.max_det
        batch, orig, scales = self._prepare_inputs(images, use_imgsz)
        with model_infer_guard(self.model_path, task=self._infer_task):
            labels, boxes, scores = self.engine.infer(batch, orig)
        return self._decode_outputs(
            labels, boxes, scores, detect_ids, scales, float(conf), use_max
        )


if __name__ == "__main__":
    IMAGE_PATH = ""
    MODEL_PATH = ""
    CONF = 0.3

    if not IMAGE_PATH or not MODEL_PATH:
        raise SystemExit("在 __main__ 中填写 IMAGE_PATH / MODEL_PATH 后直接运行")

    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(IMAGE_PATH)
    det = ModelDetectorOnnx(
        MODEL_PATH,
        conf_thresh=CONF,
        conf_merge=CONF,
        class_names=["insect"],
    )
    out = det.predict(img, clip_size=0, overlap_size=0)
    print(f"boxes={len(out)}")
    for row in out[:20]:
        print(row)
