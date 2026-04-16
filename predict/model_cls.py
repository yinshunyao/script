#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/05/21
# @Author  : ysy
# @Email   : xxx@qq.com 
# @Detail  : 
# @Software: PyCharm
import logging
import torchvision
import torch
from ultralytics import YOLO

class ModelCls:

    def __init__(self, model_path, device=None):
        self.model_path = model_path
        # 自动检测设备
        if device is None:
            self.device = self._auto_detect_device()
        else:
            self.device = device
        self.model = YOLO(self.model_path, task="classification")
        # 切换到推理模式
        self.names = self.model.names

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

    def predict(self, image, device=None):
        try:
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

    def predictTop2(self, image, device=None):
        try:
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