#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/05/15
# @Author  : ysy
# @Email   : xxx@qq.com 
# @Detail  : 
# @Software: PyCharm
import logging
import torchvision
import torch
from ultralytics import YOLO
import numpy as np

def get_clip(w, h, clip_size, overlap_size):
    # clip_size = 10, overlap_size = 2为例
    # 0, 10; 8, 18; 16, 26
    step = clip_size - overlap_size
    if step <= 0:
        step = 1  # 避免无限循环
    for i in range(0, w, step):
        for j in range(0, h, step):
            yield i, j, min(w, i + clip_size), min(h, j + clip_size)


def ior(box1, box2):
    x1, y1, x2, y2 = box1[:4]
    x3, y3, x4, y4 = box2[:4]
    x_overlap = max(0, min(x2, x4) - max(x1, x3))
    y_overlap = max(0, min(y2, y4) - max(y1, y3))
    overlap_area = x_overlap * y_overlap
    box1_s = (x2-x1) * (y2-y1)
    box2_s = (x4-x3) * (y4-y3)
    if box1_s == 0 or box2_s == 0:
        return 0
    return overlap_area / min(box1_s, box2_s)


class ModelDetector:
    def __init__(self, model_path, conf_thresh=0.5, conf_merge=0.3, iou_threshold=0.3, ior_threshold=0.5, device=None):
        """

        :param model_path:
        :param conf_thresh: 整体输出置信度
        :param conf_merge:  merge之前置信度
        :param iou_threshold:
        :param ior_threshold:
        :param device: 设备类型 ('cuda', 'mps', 'cpu')，如果为None则自动检测
        """
        # 自动检测设备
        if device is None:
            self.device = self._auto_detect_device()
        else:
            self.device = device

        logging.info(f"使用设备: {self.device}")

        # 加载模型（设备在predict时自动处理）
        self.model = YOLO(model_path)

        # 整体输出置信度
        self.conf_thresh = conf_thresh
        # merge之前置信度
        self.conf_merge = conf_merge
        self.iou_threshold = iou_threshold
        self.ior_threshold = ior_threshold

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

    def _predict(self, image, conf=0.01, detect_id: str='0-0', device=None):
        pred = self.model.predict(image, verbose=True, device=device or self.device)
        results = []
        for detection in pred[0].boxes:
            try:
                box_conf = detection.conf.item()
                if box_conf < conf:
                    continue

                x1, y1, x2, y2 = detection.xyxy.tolist()[0]
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
                results.append([x1, y1, x2, y2, box_conf, int(detection.cls.item()), detect_id])
            except Exception as e:
                logging.error(f"parse the box {detection} error: {e}")
                continue

        return results

    def merge_iou(self, boxes):
        boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
        # NMS操作在CPU上更稳定，特别是对于MPS设备
        # 如果需要在GPU上加速，可以使用CUDA，但MPS可能不支持某些操作
        boxes_xyxy = torch.tensor([box[:4] for box in boxes], dtype=torch.float)
        boxes_conf = torch.tensor([box[4] for box in boxes], dtype=torch.float)
        keep_indices = torchvision.ops.nms(boxes_xyxy, boxes_conf, self.iou_threshold)

        # 筛选保留的框
        filtered_boxes = boxes_xyxy[keep_indices]
        filtered_scores = boxes_conf[keep_indices]
        # 获取保留框的类别信息
        filtered_cls = [boxes[idx][5] for idx in keep_indices]
        return [
            {
                "x1": int(box[0]),
                "y1": int(box[1]),
                "x2": int(box[2]),
                "y2": int(box[3]),
                "conf": float(filtered_scores[idx].item()),
                "cls_id": int(filtered_cls[idx]),
                "class_name": self.model.names[int(filtered_cls[idx])],
            }
            for idx, box in enumerate(filtered_boxes)
        ]

    def merge_ior(self, boxes):
        # 按置信度降序排序，优先保留高置信度的框
        boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
        filtered_boxes = []
        merged_indices = set()  # 记录已被合并的框的索引

        for i in range(len(boxes)):
            if i in merged_indices:
                continue

            box = boxes[i]
            merged = False

            for j in range(i + 1, len(boxes)):
                if j in merged_indices:
                    continue

                # 如果 detect_id相同，不合并
                if boxes[i][6] == boxes[j][6]:
                    continue

                # cls不同，不合并
                if boxes[i][5] != boxes[j][5]:
                    continue

                # 计算IoR
                if ior(box, boxes[j]) > self.ior_threshold:
                    # 根据框的大小来保留大框
                    s_i = (box[2] - box[0]) * (box[3] - box[1])
                    s_j = (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1])

                    if s_i > s_j:
                        # 当前框更大，保留当前框，标记j为已合并
                        merged_indices.add(j)
                    else:
                        # j框更大，用j框替换当前框，标记i为已合并
                        box = boxes[j]
                        merged_indices.add(i)
                        merged = True
                        break

            if not merged:
                filtered_boxes.append(box)

        return [
            {
                "x1": int(box[0]),
                "y1": int(box[1]),
                "x2": int(box[2]),
                "y2": int(box[3]),
                "conf": float(box[4]),
                "cls_id": int(box[5]),
                "class_name": self.model.names[int(box[5])],
                "detect_id": box[6],
            }
            for box in filtered_boxes
        ]

    def _convert_result(self, results):
        return [
            {
                "x1": int(box[0]),
                "y1": int(box[1]),
                "x2": int(box[2]),
                "y2": int(box[3]),
                "conf": float(box[4]),
                "cls_id": int(box[5]),
                "class_name": self.model.names[int(box[5])],
                "detect_id": box[6],
            }
            for idx, box in enumerate(results)
        ]

    @staticmethod
    def draw(img, results):
        import cv2
        # 调试展示图片
        for result in results:
            x1, y1, x2, y2 = result["x1"], result["y1"], result["x2"], result["y2"]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{result['class_name']}-{result['conf']:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

        cv2.imshow("img", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def _is_near_clip_edge(box, clip_w, clip_h, edge_reject_distance):
        """
        判断检测框是否过于靠近切片边缘。

        :param box: 检测框，至少包含 [x1, y1, x2, y2]
        :param clip_w: 切片有效宽度
        :param clip_h: 切片有效高度
        :param edge_reject_distance: 边缘过滤阈值（像素），<=0 时关闭
        :return: True 表示应过滤
        """
        if edge_reject_distance is None or edge_reject_distance <= 0:
            return False

        x1, y1, x2, y2 = box[:4]
        min_edge_distance = min(x1, y1, clip_w - x2, clip_h - y2)
        return min_edge_distance < edge_reject_distance

    def predict(self, image, clip_size=2500, overlap_size=800,
                padding=False, debug=False, debug_clip=False,
                min_size=5, max_size=None, edge_reject_distance=0,
                device=None,
                ):
        """
        推理
        :param device:
        :param image:
        :param clip_size: 按照正方形切片，如果大于或者等于全图，不切
        :param overlap_size: 多个切片重叠区域像素大小
        :param padding: 当为True时，边缘切片不足正方形会往右或往下填充全黑像素扩展为正方形
        :param edge_reject_distance: 切片边缘过滤阈值（像素），<=0 表示关闭
        :param debug: 调试用
        :return:
        """
        w = image.shape[1]
        h = image.shape[0]
        all_box = []
        if not clip_size or not overlap_size and (clip_size >= w and clip_size >= h or clip_size <= overlap_size <= 1):
            results = self._predict(image, self.conf_thresh, device=device)
            results = self._convert_result(results)
            if debug:
                self.draw(image, results)

            return results

        # 切片
        for clip_x1, clip_y1, clip_x2, clip_y2 in get_clip(w, h, clip_size, overlap_size):
            clip = image[clip_y1:clip_y2, clip_x1:clip_x2]

            # padding: 边缘切片不足正方形时，往右或往下填充全黑像素扩展为正方形
            actual_clip_w = clip_x2 - clip_x1  # 真实切片宽度（padding前）
            actual_clip_h = clip_y2 - clip_y1  # 真实切片高度（padding前）
            if padding and (actual_clip_w < clip_size or actual_clip_h < clip_size):
                pad_h = max(clip_size, actual_clip_h)
                pad_w = max(clip_size, actual_clip_w)
                if len(clip.shape) == 3:
                    padded = np.zeros((pad_h, pad_w, clip.shape[2]), dtype=clip.dtype)
                else:
                    padded = np.zeros((pad_h, pad_w), dtype=clip.dtype)
                padded[:actual_clip_h, :actual_clip_w] = clip
                clip = padded

            # results = self._predict(clip, conf=self.conf_merge, detect_id=f"{clip_x1}-{clip_y1}", device=device)
            results = self._predict(clip, conf=self.conf_merge, detect_id=f"{clip_x1}-{clip_y1}", device=device)
            filtered_clip_results = []

            # 校准坐标
            for result in results:
                local_result = result.copy()
                # 置信度过滤
                if local_result[4] < self.conf_thresh:
                    continue

                # 如果有 padding，限制检测框在真实图像范围内，丢弃完全落在填充区域的框
                if padding:
                    # 框完全在填充区域内，丢弃
                    if local_result[0] >= actual_clip_w or local_result[1] >= actual_clip_h:
                        continue
                    # 限制 x2, y2 不超过真实切片边界
                    local_result[2] = min(local_result[2], actual_clip_w)
                    local_result[3] = min(local_result[3], actual_clip_h)

                # 如果有尺寸阈值，则过滤
                if min_size and ((local_result[2] - local_result[0]) < min_size or (local_result[3] - local_result[1]) < min_size):
                    continue

                if max_size and ((local_result[2] - local_result[0]) > max_size or (local_result[3] - local_result[1]) > max_size):
                    continue

                # 切片模式下过滤靠近切片边缘的框，避免边缘截断框干扰后续 merge
                if self._is_near_clip_edge(local_result, actual_clip_w, actual_clip_h, edge_reject_distance):
                    continue

                filtered_clip_results.append(local_result.copy())

                local_result[0] += clip_x1
                local_result[1] += clip_y1
                local_result[2] += clip_x1
                local_result[3] += clip_y1

                all_box.append(local_result)

            if debug_clip and results:
                import cv2
                clip_debug = clip.copy()

                # 原始检出框：灰色
                for result in results:
                    bx1, by1, bx2, by2 = result[:4]
                    cv2.rectangle(clip_debug, (bx1, by1), (bx2, by2), (180, 180, 180), 2)
                    cv2.putText(
                        clip_debug, f"{result[5]}-{result[4]:.2f}", (bx1, by1 - 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(180, 180, 180), thickness=1
                    )

                # 过滤后保留框：红色
                for result in filtered_clip_results:
                    bx1, by1, bx2, by2 = result[:4]
                    cv2.rectangle(clip_debug, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
                    cv2.putText(
                        clip_debug, f"{result[5]}-{result[4]:.2f}", (bx1, by1 - 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=1
                    )

                window_name = (
                    f"clip x:{clip_x1}-{clip_x2} y:{clip_y1}-{clip_y2} "
                    f"size:{actual_clip_w}x{actual_clip_h}"
                )
                cv2.imshow(window_name, clip_debug)
                cv2.waitKey(0)
                cv2.destroyWindow(window_name)

        results = self.merge_ior( all_box)

        if debug:
            self.draw(image, results)
        return results

