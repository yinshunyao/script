#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/05/21
# @Author  : ysy
# @Email   : xxx@qq.com 
# @Detail  : 
# @Software: PyCharm
import logging

import cv2
import numpy as np
import torch
import torchvision
from ultralytics import YOLO


class ModelCls:
    """YOLO 分类封装；可选灰度+CLAHE+Otsu 二值化；可选将非正方形 BGR 裁剪白边补成正方形。"""

    def __init__(self, model_path, device=None, pad_square=False, gray_binarize=False):
        self.model_path = model_path
        self.pad_square = bool(pad_square)
        self.gray_binarize = bool(gray_binarize)
        # 自动检测设备
        if device is None:
            self.device = self._auto_detect_device()
        else:
            self.device = device
        self.model = YOLO(self.model_path, task="classification")
        # 切换到推理模式
        self.names = self.model.names

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

    def _maybe_gray_binarize(self, image, gray_binarize):
        use = self.gray_binarize if gray_binarize is None else bool(gray_binarize)
        if not use:
            return image
        return self.bgr_gray_clahe_otsu_to_bgr(image)

    def _maybe_pad(self, image, pad_square):
        use = self.pad_square if pad_square is None else bool(pad_square)
        if not use:
            return image
        return self.pad_bgr_to_square(image)

    def _preprocess_for_predict(self, image, pad_square, gray_binarize):
        image = self._maybe_gray_binarize(image, gray_binarize)
        image = self._maybe_pad(image, pad_square)
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

    def predict(self, image, device=None, pad_square=None, gray_binarize=None):
        try:
            image = self._preprocess_for_predict(image, pad_square, gray_binarize)
            results = self.model.predict(image, verbose=False, device=device or self.device)
            if not results:
                logging.error(f"模型{self.model_path}推理结果为空")
                return None

            probe = results[0].probs
            class_id = probe.top1
            # 兼容「top3 置信度」使用场景：优先取 top5 的前 3 项（若不足则按实际长度返回）
            top_ids = list(probe.top5) if hasattr(probe, "top5") and probe.top5 is not None else [class_id]
            top_confs = list(probe.top5conf) if hasattr(probe, "top5conf") and probe.top5conf is not None else [probe.top1conf]
            top3 = []
            for i in range(min(3, len(top_ids), len(top_confs))):
                cid = int(top_ids[i])
                top3.append(
                    {
                        "class_id": cid,
                        "class_name": self.names[cid],
                        "conf": float(top_confs[i].item()),
                    }
                )
            return {
                "class_id": class_id,
                "class_name": self.names[class_id],
                "conf": float(probe.top1conf.item()),
                "top3": top3,
            }
        except Exception as e:
            logging.error(f"模型{self.model_path}推理异常:{e}", exc_info=True)
            return None

    def predictTop2(self, image, device=None, pad_square=None, gray_binarize=None):
        try:
            image = self._preprocess_for_predict(image, pad_square, gray_binarize)
            results = self.model.predict(image, verbose=False, device=device or self.device)
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