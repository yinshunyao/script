#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/05/21
# @Author  : ysy
# @Email   : xxx@qq.com 
# @Detail  : 
# @Software: PyCharm
import logging
from ultralytics import YOLO

class ModelCls:

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = YOLO(self.model_path, task="classification")
        # 切换到推理模式
        self.names = self.model.names

    def predict(self, image, device=None):
        try:
            results = self.model.predict(image, device=device)
            if not results:
                logging.error(f"模型{self.model_path}推理结果为空")
                return None

            probe = results[0].probs
            class_id = probe.top1
            return {
                "class_id": class_id,
                "class_name": self.names[class_id],
                "conf": float(probe.top1conf.item()),
            }
        except Exception as e:
            logging.error(f"模型{self.model_path}推理异常:{e}", exc_info=True)
            return None