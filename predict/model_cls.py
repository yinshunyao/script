#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/05/21
# @Author  : ysy
# @Email   : xxx@qq.com 
# @Detail  : 
# @Software: PyCharm
import logging
from typing import Any

import cv2
import numpy as np
import torch
import torchvision
from script.predict.model_infer_lock import model_infer_guard
from script.predict.model_trt import read_trt_engine_max_batch
from script.predict.model_yolo_cache import get_cached_yolo
from script.predict.model_channel import (
    adapt_image_to_model_channels,
    detect_model_input_channels,
    mps_safe_device,
)


class ModelCls:
    """YOLO 分类封装；可选灰度+CLAHE+Otsu 二值化；可选白边补方；可选推理前最后一步转灰度。"""

    def __init__(
        self,
        model_path,
        device=None,
        pad_square=False,
        gray_binarize=False,
        pad_color_bgr=(255, 255, 255),
        to_gray=False,
    ):
        self.model_path = model_path
        self._infer_task = "classify"
        self.pad_square = bool(pad_square)
        self.gray_binarize = bool(gray_binarize)
        self.to_gray = bool(to_gray)
        self.pad_color_bgr = tuple(int(x) for x in pad_color_bgr)
        # 自动检测设备
        if device is None:
            self.device = self._auto_detect_device()
        else:
            self.device = device
        # Ultralytics 任务名为 classify（非 classification）；.engine 导出权重不会从 ckpt 推断 task，须显式传对
        self.model = get_cached_yolo(self.model_path, task="classify")
        self.model_ch = detect_model_input_channels(self.model)
        self.trt_max_batch = (
            read_trt_engine_max_batch(self.model_path) if self.uses_trt else None
        )
        if self.trt_max_batch == 1:
            logging.info(
                "分类 TRT engine 为固定 batch=1: %s（crop 将逐张推理；"
                "批量加速需 DYNAMIC=True 且 BATCH>1 重新导出）",
                self.model_path,
            )
        if self.model_ch == 1:
            logging.info(f"分类模型为单通道(ch=1)，推理将以单通道灰度输入: {self.model_path}")
            mps_safe_device(self.device, self.model_ch)
        # 切换到推理模式
        self.names = self.model.names

    @property
    def uses_trt(self) -> bool:
        return str(self.model_path or "").lower().endswith(".engine")

    @staticmethod
    def pad_bgr_to_square(image, pad_value=(255, 255, 255)):
        """
        将 BGR 图居中补成正方形，短边方向填充 pad_value（与训练侧白边正方形几何一致）。

        :param image: numpy BGR, shape (H, W, 3)
        :return: 正方形图或原图（已方或非三通道时原样返回）
        """
        if image is None or image.size == 0:
            return image
        if image.ndim != 3 or image.shape[2] != 3:
            return image
        h, w = image.shape[:2]
        if h == w:
            return image
        s = max(h, w)
        out = np.full((s, s, 3), pad_value, dtype=image.dtype)
        y0 = (s - h) // 2
        x0 = (s - w) // 2
        out[y0 : y0 + h, x0 : x0 + w] = image
        return out

    @staticmethod
    def bgr_gray_clahe_otsu_to_bgr(image, clahe_clip=2.0, clahe_tile=(8, 8)):
        """
        BGR 裁剪 → 灰度 → CLAHE → Otsu 二值化 → 三通道 BGR（与训练侧 cls_gray_binarize 管线一致）。
        """
        if image is None or image.size == 0:
            return image
        if image.ndim != 3 or image.shape[2] != 3:
            return image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(
            clipLimit=float(clahe_clip),
            tileGridSize=(int(clahe_tile[0]), int(clahe_tile[1])),
        )
        enhanced = clahe.apply(gray)
        _, binary = cv2.threshold(
            enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return cv2.merge([binary, binary, binary])

    @staticmethod
    def bgr_to_gray_3ch(image):
        """
        BGR → 灰度 → 三通道 BGR（R=G=B）。

        亮度系数与训练侧 PIL ``convert("L")`` 等价（均为 ITU-R 601-2 / cv2 默认）。
        非三通道或空图原样返回。
        """
        if image is None or image.size == 0:
            return image
        if image.ndim != 3 or image.shape[2] != 3:
            return image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def _maybe_gray_binarize(self, image, gray_binarize):
        use = self.gray_binarize if gray_binarize is None else bool(gray_binarize)
        if not use:
            return image
        return self.bgr_gray_clahe_otsu_to_bgr(image)

    def _maybe_to_gray(self, image, to_gray):
        use = self.to_gray if to_gray is None else bool(to_gray)
        if not use:
            return image
        return self.bgr_to_gray_3ch(image)

    def _maybe_pad(self, image, pad_square, pad_color_bgr):
        use = self.pad_square if pad_square is None else bool(pad_square)
        if not use:
            return image
        color = self.pad_color_bgr if pad_color_bgr is None else tuple(int(x) for x in pad_color_bgr)
        return self.pad_bgr_to_square(image, pad_value=color)

    def _preprocess_for_predict(
        self, image, pad_square, gray_binarize, pad_color_bgr, to_gray
    ):
        image = self._maybe_gray_binarize(image, gray_binarize)
        image = self._maybe_pad(image, pad_square, pad_color_bgr)
        # to_gray 作为推理前最后一步
        image = self._maybe_to_gray(image, to_gray)
        return image

    def _auto_detect_device(self):
        """
        自动检测可用的设备
        优先级: CUDA > MPS > CPU

        :return: 设备名称 ('cuda', 'mps', 'cpu')
        """
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'

    def predict(self, image, device=None, pad_square=None, gray_binarize=None, pad_color_bgr=None, to_gray=None):
        try:
            image = self._preprocess_for_predict(
                image, pad_square, gray_binarize, pad_color_bgr, to_gray
            )
            image = adapt_image_to_model_channels(image, self.model_ch)
            with model_infer_guard(self.model_path, task=self._infer_task):
                results = self.model.predict(
                    image, verbose=False, device=mps_safe_device(device or self.device, self.model_ch)
                )
            if not results:
                logging.error(f"模型{self.model_path}推理结果为空")
                return None

            probe = results[0].probs
            return self._parse_cls_result(probe)
        except Exception as e:
            logging.error(f"模型{self.model_path}推理异常:{e}", exc_info=True)
            return None

    def _parse_cls_result(self, probe) -> dict[str, Any]:
        class_id = probe.top1
        top_ids = (
            list(probe.top5)
            if hasattr(probe, "top5") and probe.top5 is not None
            else [class_id]
        )
        top_confs = (
            list(probe.top5conf)
            if hasattr(probe, "top5conf") and probe.top5conf is not None
            else [probe.top1conf]
        )
        topk: list[dict] = []
        for i in range(min(5, len(top_ids), len(top_confs))):
            cid = int(top_ids[i])
            topk.append(
                {
                    "class_id": cid,
                    "class_name": self.names[cid],
                    "conf": float(top_confs[i].item()),
                }
            )
        top3 = topk[:3]
        return {
            "class_id": class_id,
            "class_name": self.names[class_id],
            "conf": float(probe.top1conf.item()),
            "top3": top3,
            "topk": topk,
        }

    def predict_batch(
        self,
        images: list[np.ndarray],
        device=None,
        pad_square=None,
        gray_binarize=None,
        pad_color_bgr=None,
        to_gray=None,
        max_batch: int | None = None,
    ) -> list[dict | None]:
        """多 crop 批量分类；``max_batch`` 默认 TRT 取 engine 上限，``.pt`` 取整批长度。"""
        if not images:
            return []
        if max_batch is not None and int(max_batch) > 0:
            chunk_limit = int(max_batch)
        elif self.uses_trt:
            chunk_limit = self.trt_max_batch if self.trt_max_batch else len(images)
        else:
            chunk_limit = len(images)
        if chunk_limit is None or chunk_limit <= 0:
            chunk_limit = len(images)
        out: list[dict | None] = []
        for start in range(0, len(images), max(1, int(chunk_limit))):
            chunk = images[start : start + max(1, int(chunk_limit))]
            out.extend(
                self._predict_batch_chunk(
                    chunk,
                    device=device,
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
        device=None,
        pad_square=None,
        gray_binarize=None,
        pad_color_bgr=None,
        to_gray=None,
    ) -> list[dict | None]:
        if not images:
            return []
        try:
            preprocessed = []
            for image in images:
                img = self._preprocess_for_predict(
                    image, pad_square, gray_binarize, pad_color_bgr, to_gray
                )
                preprocessed.append(adapt_image_to_model_channels(img, self.model_ch))
            with model_infer_guard(self.model_path, task=self._infer_task):
                results = self.model.predict(
                    preprocessed if len(preprocessed) > 1 else preprocessed[0],
                    verbose=False,
                    device=mps_safe_device(device or self.device, self.model_ch),
                )
            if not isinstance(results, list):
                results = [results]
            if not results:
                logging.error("模型%s批量推理结果为空", self.model_path)
                return [None] * len(images)
            out: list[dict | None] = []
            for r in results:
                if r is None or getattr(r, "probs", None) is None:
                    out.append(None)
                else:
                    out.append(self._parse_cls_result(r.probs))
            if len(out) < len(images):
                out.extend([None] * (len(images) - len(out)))
            return out[: len(images)]
        except (AssertionError, RuntimeError) as e:
            msg = str(e)
            if self.uses_trt and len(images) > 1 and (
                "max model size" in msg or "input size" in msg
            ):
                logging.warning(
                    "模型%s TRT batch 不匹配，回退逐张推理: %s",
                    self.model_path,
                    msg,
                )
                return [
                    self.predict(
                        image,
                        device=device,
                        pad_square=pad_square,
                        gray_binarize=gray_binarize,
                        pad_color_bgr=pad_color_bgr,
                        to_gray=to_gray,
                    )
                    for image in images
                ]
            logging.error("模型%s批量推理异常:%s", self.model_path, e, exc_info=True)
            return [None] * len(images)
        except Exception as e:
            logging.error("模型%s批量推理异常:%s", self.model_path, e, exc_info=True)
            return [None] * len(images)

    def predictTop2(self, image, device=None, pad_square=None, gray_binarize=None, pad_color_bgr=None, to_gray=None):
        try:
            image = self._preprocess_for_predict(
                image, pad_square, gray_binarize, pad_color_bgr, to_gray
            )
            image = adapt_image_to_model_channels(image, self.model_ch)
            with model_infer_guard(self.model_path, task=self._infer_task):
                results = self.model.predict(
                    image, verbose=False, device=mps_safe_device(device or self.device, self.model_ch)
                )
            if not results:
                logging.error(f"模型{self.model_path}推理结果为空")
                return None

            probe = results[0].probs
            class_id = probe.top5
            top_conf = probe.top5conf
            result = {
                "1": {"class_id": class_id[0], "class_name": self.names[class_id[0]], "conf": float(top_conf[0].item()), },
                "2": {"class_id": class_id[1], "class_name": self.names[class_id[1]], "conf": float(top_conf[1].item()), }
            }
            return result
        except Exception as e:
            logging.error(f"模型{self.model_path}推理异常:{e}", exc_info=True)
            return None