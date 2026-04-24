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
import cv2

def _pad_background_value(clip):
    """与 clip 同 dtype 的「白底」填充标量：uint8 为 255；浮点按 0~1 / 0~255 推断。"""
    if clip.dtype == np.uint8:
        return 255
    if np.issubdtype(clip.dtype, np.floating):
        if clip.size == 0:
            return np.float32(1.0)
        mx = float(np.max(clip))
        return np.float32(1.0 if mx <= 1.0 else 255.0)
    if np.issubdtype(clip.dtype, np.integer):
        return int(np.iinfo(clip.dtype).max)
    return 255


def _pad_tile_to_clip_square(clip, actual_clip_w, actual_clip_h, clip_size):
    """
    将宽高均不超过 clip_size 的切片补成 clip_size×clip_size，原内容居中，空缺填白底。

    :return: (padded_clip, off_x, off_y, pad_w, pad_h)；无需补全时返回 (clip, 0, 0, aw, ah)。
    """
    if actual_clip_w >= clip_size and actual_clip_h >= clip_size:
        return clip, 0, 0, actual_clip_w, actual_clip_h
    # 与分片边长一致：画布恒为 clip_size×clip_size（get_clip 保证单窗宽高 ≤ clip_size）
    pad_w = clip_size
    pad_h = clip_size
    off_x = (pad_w - actual_clip_w) // 2
    off_y = (pad_h - actual_clip_h) // 2
    fill = _pad_background_value(clip)
    if len(clip.shape) == 3:
        padded = np.full((pad_h, pad_w, clip.shape[2]), fill, dtype=clip.dtype)
    else:
        padded = np.full((pad_h, pad_w), fill, dtype=clip.dtype)
    padded[off_y : off_y + actual_clip_h, off_x : off_x + actual_clip_w] = clip
    return padded, off_x, off_y, pad_w, pad_h


def get_clip(w, h, clip_size, overlap_size):
    """
    生成切片滑窗坐标（左上、右下，右下为开区间）。

    规则：
    - **只有**当某个方向的边长 > clip_size 时，该方向才需要滑窗并使用 overlap_size。
    - 当某个方向边长 <= clip_size 时，该方向只取起点 0（单窗），由上层 padding 补成正方形，
      不再在该方向引入 overlap/多窗。

    例：
    - w > clip_size, h <= clip_size：x 方向滑窗，y 方向仅 j=0 一次（长边切片 + 短边 padding）
    - w <= clip_size, h <= clip_size：仅一窗 (0,0,w,h)
    """
    # clip_size = 10, overlap_size = 2 为例
    # x: 0,10; 8,18; 16,26 ...
    def _step(length: int) -> int:
        if length <= clip_size:
            return clip_size  # range(0, length, clip_size) 只会产生起点 0
        s = clip_size - overlap_size
        return 1 if s <= 0 else s

    step_x = _step(w)
    step_y = _step(h)

    for i in range(0, w, step_x):
        for j in range(0, h, step_y):
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

def enhance_contrast(img):
    # 方法1: CLAHE（自适应直方图均衡，对局部纹理友好）
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

class ModelDetector:
    def __init__(self, model_path, conf_thresh=0.5, conf_merge=0.3, iou_threshold=0.3, ior_threshold=0.5, device=None, augment=False):
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

        self.augment = augment

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

    def _predict(self, image, conf=0.01, detect_id: str='0-0', device=None, imgsz=0, overlap=0):
        kwargs = {}
        if imgsz:
            kwargs['imgsz'] = imgsz
        if overlap:
            kwargs['overlap'] = overlap
        pred = self.model.predict(image, verbose=False, device=device or self.device, augment=self.augment, **kwargs)
        results = []
        if not pred or not pred[0].boxes:
            return results
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
        """
        同类、不同切片 detect_id、IoR 超过阈值时合并为一条：
        外接框取组内坐标并集（x1/y1 最小，x2/y2 最大），置信度取组内最高；
        detect_id 随组内置信度最高框保留。
        """
        n = len(boxes)
        if n == 0:
            return []

        parent = list(range(n))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i in range(n):
            for j in range(i + 1, n):
                if self._box_row_filtered(boxes[i]) or self._box_row_filtered(boxes[j]):
                    continue
                if boxes[i][6] == boxes[j][6]:
                    continue
                if boxes[i][5] != boxes[j][5]:
                    continue
                if ior(boxes[i], boxes[j]) > self.ior_threshold:
                    union(i, j)

        clusters = {}
        for i in range(n):
            r = find(i)
            clusters.setdefault(r, []).append(i)

        filtered_boxes = []
        for _root, idxs in clusters.items():
            group = [boxes[i] for i in idxs]
            max_conf = max(b[4] for b in group)
            pick = max(group, key=lambda b: b[4])
            ux1 = min(b[0] for b in group)
            uy1 = min(b[1] for b in group)
            ux2 = max(b[2] for b in group)
            uy2 = max(b[3] for b in group)
            flt = any(self._box_row_filtered(b) for b in group)
            # edge_min_dist：沿用置信度最高框（pick）的值（若存在）
            # edge_min_dist = pick[8] if len(pick) > 8 else None
            # 使用最大值
            edge_min_dist = max(b[8] for b in group)
            merged = [ux1, uy1, ux2, uy2, max_conf, pick[5], pick[6], flt, edge_min_dist]
            filtered_boxes.append(merged)

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
                "filter": bool(box[7]) if len(box) > 7 else False,
                "edge_min_dist": (None if len(box) <= 8 or box[8] is None else float(box[8])),
            }
            for box in filtered_boxes
        ]

    @staticmethod
    def _box_row_filtered(box):
        """列表格式检测框第 8 维为 True 时表示已标记过滤（不参与 merge 等）。"""
        return len(box) > 7 and bool(box[7])

    def _slice_local_box_prefilter(
        self, result, actual_clip_w, actual_clip_h, padding, min_size, max_size,
    ):
        """
        切片局部坐标下应用置信度、padding 越界、尺寸阈值，
        并计算该框到切片有效边界的最小距离（edge_min_dist，像素）。

        返回 (local_row, filtered, edge_min_dist)。
        local_row 为长度 7 的列表 [x1,y1,x2,y2,conf,cls,detect_id]。
        """
        lr = list(result[:7])
        if lr[4] < self.conf_thresh:
            return lr, True, None

        if padding:
            # 局部坐标系原点为「真实切片」左上角；与有效像素矩形求交，去掉填充边上的模型输出
            x1, y1, x2, y2 = lr[0], lr[1], lr[2], lr[3]
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            ix1 = max(0, x1)
            iy1 = max(0, y1)
            ix2 = min(actual_clip_w, x2)
            iy2 = min(actual_clip_h, y2)
            if ix1 >= ix2 or iy1 >= iy2:
                return lr, True, None
            lr[0], lr[1], lr[2], lr[3] = ix1, iy1, ix2, iy2

        w_b = lr[2] - lr[0]
        h_b = lr[3] - lr[1]
        if min_size and (w_b < min_size or h_b < min_size):
            return lr, True, None
        if max_size and (w_b > max_size or h_b > max_size):
            return lr, True, None

        edge_min_dist = float(
            self._edge_min_dist_local(
                lr[0], lr[1], lr[2], lr[3],
                clip_w=actual_clip_w,
                clip_h=actual_clip_h,
            )
        )
        return lr, False, edge_min_dist

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
                "filter": False,
            }
            for idx, box in enumerate(results)
        ]

    @staticmethod
    def draw(img, results):
        import cv2
        # 调试：全部框绘制；filter=True 灰色，filter=False 红色
        for result in results:
            x1, y1, x2, y2 = result["x1"], result["y1"], result["x2"], result["y2"]
            color = (180, 180, 180) if result.get("filter") else (0, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                img, f"{result['class_name']}-{result['conf']:.2f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2,
            )

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

    @staticmethod
    def _parse_detect_clip_origin(detect_id):
        if not detect_id or not isinstance(detect_id, str):
            return None
        parts = detect_id.split("-", 1)
        if len(parts) != 2:
            return None
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            return None

    @staticmethod
    def _min_dist_box_to_clip_rect(x1, y1, x2, y2, clip_x1, clip_y1, clip_x2, clip_y2):
        """全图坐标下，框到切片矩形四边的最小距离（像素，下限为 0）。"""
        return min(
            max(0, x1 - clip_x1),
            max(0, clip_x2 - x2),
            max(0, y1 - clip_y1),
            max(0, clip_y2 - y2),
        )

    @staticmethod
    def _edge_min_dist_local(x1, y1, x2, y2, clip_w, clip_h):
        """切片局部坐标下，框到切片有效矩形四边的最小距离（像素，下限为 0）。"""
        return min(
            max(0, x1),
            max(0, y1),
            max(0, clip_w - x2),
            max(0, clip_h - y2),
        )

    def _edge_min_dist_for_detection(self, det, w, h, clip_size):
        """
        切片推理结果（全图坐标）到所属切片有效边界的最近距离；无法解析时返回 None。
        """
        origin = self._parse_detect_clip_origin(det.get("detect_id", ""))
        if origin is None or not clip_size:
            return None
        clip_x1, clip_y1 = origin
        clip_x2 = min(w, clip_x1 + clip_size)
        clip_y2 = min(h, clip_y1 + clip_size)
        return self._min_dist_box_to_clip_rect(
            det["x1"], det["y1"], det["x2"], det["y2"],
            clip_x1, clip_y1, clip_x2, clip_y2,
        )

    def _apply_post_merge_edge_filter(
        self, results, w, h, clip_size,
        edge_reject_distance, edge_reject_conf_threshold,
    ):
        """
        merge 之后：为每条结果写入 edge_min_dist；当距离阈值生效且满足滤除条件时
        将 det['filter'] 置为 True（不删除条目，由 predict 对外返回前剥离）。
        """
        if not results:
            return results

        out = []
        for det in results:
            det = dict(det)
            det.setdefault("filter", False)

            dm = det.get("edge_min_dist", None)
            if dm is None:
                dm = self._edge_min_dist_for_detection(det, w, h, clip_size)
            if dm is not None:
                dm = float(max(0.0, float(dm)))
                det["edge_min_dist"] = dm
            else:
                det["edge_min_dist"] = None

            if det["filter"]:
                out.append(det)
                continue

            if edge_reject_distance is None or edge_reject_distance <= 0:
                out.append(det)
                continue

            if dm is None:
                out.append(det)
                continue

            # 阈值高于切片阈值
            if edge_reject_conf_threshold and det["conf"] > edge_reject_conf_threshold and dm > 2:
                out.append(det)
                continue

            # 距离大于等于边缘距离
            if dm >= edge_reject_distance:
                out.append(det)
                continue

            det["filter"] = True
            out.append(det)

        return out

    def predict(self, image, clip_size=2500, overlap_size=800,
                padding=True, debug=False, debug_clip=False,
                min_size=5, max_size=None, edge_reject_distance=0,
                edge_reject_conf_threshold=None,
                device=None,
                ):
        """
        推理
        :param device:
        :param image:
        :param clip_size: 按照正方形切片，如果大于或者等于全图，不切
        :param overlap_size: 多个切片重叠区域像素大小
        :param padding: 默认 True。边缘切片任一边小于 clip_size 则补成与分片边长一致的
            clip_size×clip_size 正方形；原图居中、空缺白底；再将检测框坐标移回切片局部坐标。
        :param edge_reject_distance: merge 之后切片边缘距离阈值（像素），<=0 表示不按距离滤除
        :param edge_reject_conf_threshold: merge 之后与距离联合过滤的置信度上限（不含）；
            None 时使用 self.conf_thresh。若需「仅按距离、忽略置信度」可传入大于 1 的值。
        :param debug: 调试用
        :return:
        """
        w = image.shape[1]
        h = image.shape[0]
        all_box = []
        kwargs = {}
        if not clip_size or not overlap_size or (clip_size >= w and clip_size >= h or clip_size <= overlap_size <= 1):
            # kwargs["imgsz"] = clip_size
            # kwargs["overlap"] = overlap_size/clip_size

            results = self._predict(image, self.conf_thresh, device=device, **kwargs)
            results = self._convert_result(results)
            if debug:
                self.draw(image, results)

            return [r for r in results if not r.get("filter")]

        # 切片
        for clip_x1, clip_y1, clip_x2, clip_y2 in get_clip(w, h, clip_size, overlap_size):
            clip = image[clip_y1:clip_y2, clip_x1:clip_x2]
            # 自适应直方图均衡
            # clip = enhance_contrast(clip)
            actual_clip_w = clip_x2 - clip_x1  # 真实切片宽度（padding 前）
            actual_clip_h = clip_y2 - clip_y1  # 真实切片高度（padding 前）
            pad_off_x = pad_off_y = 0
            if padding and (actual_clip_w < clip_size or actual_clip_h < clip_size):
                clip, pad_off_x, pad_off_y, _pw, _ph = _pad_tile_to_clip_square(
                    clip, actual_clip_w, actual_clip_h, clip_size
                )

            results = self._predict(clip, conf=self.conf_merge, detect_id=f"{clip_x1}-{clip_y1}", device=device)

            if pad_off_x or pad_off_y:
                for r in results:
                    r[0] -= pad_off_x
                    r[1] -= pad_off_y
                    r[2] -= pad_off_x
                    r[3] -= pad_off_y

            # 校准坐标并带上 filter 标记（第 8 维）
            for result in results:
                lr, flt, edge_min_dist = self._slice_local_box_prefilter(
                    result, actual_clip_w, actual_clip_h, padding, min_size, max_size,
                )
                global_row = lr[:7] + [flt, edge_min_dist]
                global_row[0] += clip_x1
                global_row[1] += clip_y1
                global_row[2] += clip_x1
                global_row[3] += clip_y1
                all_box.append(global_row)

            if debug_clip:
                import cv2
                # clip 已为填充后画布（与分片边长一致）；框坐标在「真实窗局部」系，需平移到画布上绘制
                clip_debug = clip.copy()
                dx, dy = pad_off_x, pad_off_y

                for result in results:
                    lr, flt, _edge_min_dist = self._slice_local_box_prefilter(
                        result, actual_clip_w, actual_clip_h, padding, min_size, max_size,
                    )
                    bx1, by1, bx2, by2 = lr[:4]
                    px1, py1, px2, py2 = bx1 + dx, by1 + dy, bx2 + dx, by2 + dy
                    color = (180, 180, 180) if flt else (0, 0, 255)
                    cv2.rectangle(clip_debug, (px1, py1), (px2, py2), color, 2)
                    cv2.putText(
                        clip_debug, f"{lr[5]}-{lr[4]:.2f}", (px1, py1 - 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=color, thickness=1,
                    )

                ph, pw = clip_debug.shape[:2]
                window_name = (
                    f"clip x:{clip_x1}-{clip_x2} y:{clip_y1}-{clip_y2} "
                    f"padded:{pw}x{ph} tile:{actual_clip_w}x{actual_clip_h}"
                )
                cv2.imshow(window_name, clip_debug)
                cv2.waitKey(0)
                cv2.destroyWindow(window_name)

        results = self.merge_ior(all_box)
        results = self._apply_post_merge_edge_filter(
            results, w, h, clip_size,
            edge_reject_distance=edge_reject_distance,
            edge_reject_conf_threshold=edge_reject_conf_threshold,
        )

        if debug:
            self.draw(image, results)
        return [r for r in results if not r.get("filter")]

