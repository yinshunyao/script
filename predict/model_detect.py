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
import numpy as np
import cv2
from typing import Any, NamedTuple
from script.predict.model_infer_lock import model_infer_guard
from script.predict.model_yolo_cache import get_cached_yolo
from script.predict.model_channel import (
    detect_model_input_channels,
    mps_safe_device,
    preprocess_yolo_input,
    scale_xyxy,
    yolo_input_coord_scale,
)

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


class ClipProfile(NamedTuple):
    """单套滑窗切片参数。"""

    clip_size: int
    overlap_size: int
    clip_start: int = 0
    seg_imgsz: int = 0


class _DetectClipTile(NamedTuple):
    """单张 detect 滑窗 tile（batch 推理单元）。"""

    clip: np.ndarray
    detect_id: str
    clip_x1: int
    clip_y1: int
    clip_x2: int
    clip_y2: int
    actual_clip_w: int
    actual_clip_h: int
    pad_off_x: int
    pad_off_y: int
    clip_size: int


def _prepare_detect_clip_tile(
    image: np.ndarray,
    clip_x1: int,
    clip_y1: int,
    clip_x2: int,
    clip_y2: int,
    clip_size: int,
    padding: bool,
    profile_idx: int = 0,
) -> _DetectClipTile:
    clip = image[clip_y1:clip_y2, clip_x1:clip_x2]
    actual_clip_w = clip_x2 - clip_x1
    actual_clip_h = clip_y2 - clip_y1
    pad_off_x = pad_off_y = 0
    if padding and (actual_clip_w < clip_size or actual_clip_h < clip_size):
        clip, pad_off_x, pad_off_y, _pw, _ph = _pad_tile_to_clip_square(
            clip, actual_clip_w, actual_clip_h, clip_size
        )
    detect_id = make_clip_detect_id(profile_idx, clip_x1, clip_y1)
    return _DetectClipTile(
        clip,
        detect_id,
        clip_x1,
        clip_y1,
        clip_x2,
        clip_y2,
        actual_clip_w,
        actual_clip_h,
        pad_off_x,
        pad_off_y,
        int(clip_size),
    )


def uses_clip_inference_path(
    w: int, h: int, clip_size: int, overlap_size: int,
) -> bool:
    """是否对该图尺寸启用滑窗切片（与 ModelDetector.predict 判定一致）。"""
    return bool(
        clip_size
        and overlap_size
        and not (clip_size >= w and clip_size >= h)
        and not (clip_size <= overlap_size <= 1)
    )


def resolve_clip_profiles(
    *,
    clip_size: int = 0,
    overlap_size: int = 0,
    clip_start: int = 0,
    clip_profiles: list | None = None,
) -> list[ClipProfile]:
    """
    解析滑窗切片配置：``clip_profiles`` 非空时优先；否则单套 ``clip_size``/``overlap_size``。
    ``clip_start``：滑窗网格起始像素（x、y 同值）；``0`` 表示从 ``(0,0)`` 起。
    ``clip_profiles[].seg_imgsz``：该套 YOLO ``imgsz``；``<=0`` 或未写表示推理时用外层 ``seg_imgsz``。
    ``overlap_size: 0`` 表示整图单窗（不滑窗）；``overlap_size > 0`` 为滑窗重叠像素。
    ``clip_size: 0`` 表示不切片、整图推理（``overlap_size`` 忽略，规范为 0）。
    跳过 ``enable: false`` 或尺寸无效的项；若全部无效则回退顶层字段。
    """
    if clip_profiles:
        out: list[ClipProfile] = []
        for item in clip_profiles:
            if not isinstance(item, dict):
                continue
            if item.get("enable") is False:
                continue
            cs = int(item.get("clip_size", 0) or 0)
            os = int(item.get("overlap_size", 0) or 0)
            st = int(item.get("clip_start", 0) or 0)
            si = max(0, int(item.get("seg_imgsz", 0) or 0))
            if cs == 0:
                out.append(ClipProfile(0, 0, max(0, st), si))
                continue
            slide_ok = cs > 0 and os > 0
            whole_ok = cs > 0 and os == 0
            if slide_ok or whole_ok:
                out.append(ClipProfile(cs, os, max(0, st), si))
        if out:
            return out
    return [
        ClipProfile(
            int(clip_size or 0),
            int(overlap_size or 0),
            max(0, int(clip_start or 0)),
            0,
        )
    ]


def resolve_profile_imgsz(
    profile: ClipProfile | tuple,
    default_imgsz: int | None,
    *,
    model_default: int = 0,
) -> int | None:
    """单套 profile 生效的 ``imgsz``：profile 内 ``seg_imgsz`` → 调用方 default → 模型默认。"""
    raw = 0
    if isinstance(profile, ClipProfile):
        raw = int(profile.seg_imgsz or 0)
    elif len(profile) >= 4:
        raw = max(0, int(profile[3] or 0))
    if raw > 0:
        return raw
    if default_imgsz is not None and int(default_imgsz) > 0:
        return int(default_imgsz)
    md = int(model_default or 0)
    return md if md > 0 else None


def unpack_clip_profile(profile: ClipProfile | tuple) -> tuple[int, int, int]:
    """``(clip_size, overlap_size, clip_start)``。"""
    if isinstance(profile, ClipProfile):
        return profile.clip_size, profile.overlap_size, profile.clip_start
    if len(profile) >= 3:
        return int(profile[0]), int(profile[1]), max(0, int(profile[2] or 0))
    return int(profile[0]), int(profile[1]), 0


def make_clip_detect_id(profile_idx: int, clip_x1: int, clip_y1: int) -> str:
    return f"{int(profile_idx)}:{int(clip_x1)}-{int(clip_y1)}"


def parse_clip_detect_id(detect_id: str | None) -> tuple[int, int, int] | None:
    """
    解析 ``detect_id`` → ``(profile_idx, clip_x1, clip_y1)``。
    兼容旧格式 ``{x}-{y}``（profile_idx=0）。
    """
    if not detect_id or not isinstance(detect_id, str):
        return None
    s = detect_id.strip()
    profile_idx = 0
    if ":" in s:
        head, s = s.split(":", 1)
        try:
            profile_idx = int(head)
        except ValueError:
            return None
    parts = s.split("-", 1)
    if len(parts) != 2:
        return None
    try:
        return profile_idx, int(parts[0]), int(parts[1])
    except ValueError:
        return None


def format_clip_slice_label_suffix(row: dict) -> str:
    """
    绘图标签分片序号后缀（1-based），与 ``get_clip`` 遍历顺序一致。
    多套滑窗时为 ``#套-片``（如 ``#2-5``），单套时为 ``#5``。
    """
    if "clip_tile_seq" in row:
        seq = int(row["clip_tile_seq"]) + 1
        prof = int(row.get("clip_profile_idx", 0) or 0)
        total = int(row.get("clip_profile_total", 1) or 1)
        if total > 1:
            return f" #{prof + 1}-{seq}"
        return f" #{seq}"
    parsed = parse_clip_detect_id(row.get("detect_id"))
    if not parsed:
        return ""
    profile_idx, clip_x1, clip_y1 = parsed
    if profile_idx > 0:
        return f" #{profile_idx + 1}@{clip_x1}-{clip_y1}"
    if clip_x1 or clip_y1:
        return f" @{clip_x1}-{clip_y1}"
    return ""


def get_clip(w, h, clip_size, overlap_size, clip_start: int = 0):
    """
    生成切片滑窗坐标（左上、右下，右下为开区间）。

    规则：
    - **只有**当某个方向的边长 > clip_size 时，该方向才需要滑窗并使用 overlap_size。
    - 当某个方向边长 <= clip_size 时，该方向只取起点 0（单窗），由上层 padding 补成正方形，
      不再在该方向引入 overlap/多窗。
    - ``clip_start``：该方向需滑窗时，首个窗左上角从 ``clip_start`` 起（x、y 同值）；``0`` 为历史默认。

    例：
    - w > clip_size, h <= clip_size：x 方向滑窗，y 方向仅 j=0 一次（长边切片 + 短边 padding）
    - w <= clip_size, h <= clip_size：仅一窗 (0,0,w,h)
    """
    clip_start = max(0, int(clip_start or 0))

    def _step(length: int) -> int:
        if length <= clip_size:
            return clip_size  # range(0, length, clip_size) 只会产生起点 0
        s = clip_size - overlap_size
        return 1 if s <= 0 else s

    def _starts(length: int) -> list[int]:
        if length <= clip_size:
            return [0]
        step = _step(length)
        origin = clip_start if clip_start < length else 0
        out: list[int] = []
        i = origin
        while i < length:
            out.append(i)
            i += step
        return out

    for i in _starts(w):
        for j in _starts(h):
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
    ):
        """

        :param model_path:
        :param conf_thresh: 整体输出置信度
        :param conf_merge:  merge之前置信度
        :param iou_threshold:
        :param ior_threshold:
        :param device: 设备类型 ('cuda', 'mps', 'cpu')，如果为None则自动检测
        :param gray_contrast_enhance: 送入 YOLO 前对灰度图做 CLAHE 对比度增强（默认关）
        """
        # 自动检测设备
        if device is None:
            self.device = self._auto_detect_device()
        else:
            self.device = device

        self.augment = augment
        self.half = half

        logging.info(f"使用设备: {self.device}")

        self.model_path = str(model_path)
        self._infer_task: str | None = None
        # 加载模型（设备在predict时自动处理）
        self.model = get_cached_yolo(model_path)
        self.model_ch = detect_model_input_channels(self.model)
        if self.model_ch == 1:
            logging.info(f"检测模型为单通道(ch=1)，推理将以单通道灰度输入: {model_path}")
            mps_safe_device(self.device, self.model_ch)

        # 整体输出置信度
        self.conf_thresh = conf_thresh
        # merge之前置信度
        self.conf_merge = conf_merge
        self.conf_merge_draw = float(conf_merge_draw)
        self.iou_threshold = iou_threshold
        self.ior_threshold = ior_threshold
        # Ultralytics 内置 NMS 参数（与本文件里的 merge_iou iou_threshold 不同）
        self.nms_iou = nms_iou
        self.max_det = max_det
        # Ultralytics class-agnostic NMS（跨类别抑制）
        self.nms_agnostic = nms_agnostic
        self.gray_contrast_enhance = bool(gray_contrast_enhance)
        self.gray_clahe_clip = float(gray_clahe_clip)
        self.gray_clahe_tile = int(gray_clahe_tile)
        if self.gray_contrast_enhance:
            logging.info(
                "检测推理已开启灰度 CLAHE 对比度增强 (clip=%.2f, tile=%d): %s",
                self.gray_clahe_clip,
                self.gray_clahe_tile,
                model_path,
            )

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

    def _infer_conf(self, return_all_rows: bool) -> float:
        """``return_all_rows`` 时用更低 ``conf_merge_draw`` 供 DRAW_FILTER 绘制。"""
        return self.conf_merge_draw if return_all_rows else self.conf_merge

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
    ):
        kwargs = {}
        if imgsz:
            kwargs['imgsz'] = imgsz
        if overlap:
            kwargs['overlap'] = overlap
        if nms_iou is None:
            nms_iou = self.nms_iou
        if max_det is None:
            max_det = self.max_det
        if nms_agnostic is None:
            nms_agnostic = self.nms_agnostic
        if nms_iou is not None:
            kwargs["iou"] = float(nms_iou)
        if max_det is not None:
            kwargs["max_det"] = int(max_det)
        if nms_agnostic is not None:
            kwargs["agnostic_nms"] = bool(nms_agnostic)
        use_imgsz = int(imgsz or 0)
        source_shape = image.shape[:2]
        yolo_in = preprocess_yolo_input(
            image,
            self.model_ch,
            gray_contrast_enhance=self.gray_contrast_enhance,
            clahe_clip=self.gray_clahe_clip,
            clahe_tile=self.gray_clahe_tile,
            target_imgsz=use_imgsz,
        )
        sx, sy = yolo_input_coord_scale(source_shape, yolo_in)
        with model_infer_guard(self.model_path, task=self._infer_task):
            pred = self.model.predict(
                yolo_in,
                verbose=False,
                device=mps_safe_device(device or self.device, self.model_ch),
                augment=self.augment,
                half=self.half,
                conf=float(conf),
                **kwargs,
            )
        results = []
        if not pred or not pred[0].boxes:
            return results
        for detection in pred[0].boxes:
            try:
                box_conf = detection.conf.item()
                if box_conf < conf:
                    continue

                x1, y1, x2, y2 = detection.xyxy.tolist()[0]
                if sx != 1.0 or sy != 1.0:
                    x1, y1, x2, y2 = scale_xyxy(x1, y1, x2, y2, sx, sy)
                else:
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                results.append([x1, y1, x2, y2, box_conf, int(detection.cls.item()), detect_id])
            except Exception as e:
                logging.error(f"parse the box {detection} error: {e}")
                continue

        return results

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
    ) -> list[list]:
        """多 tile YOLO detect batch；单张时退化为 ``_predict``。"""
        if not images:
            return []
        if len(images) == 1:
            return [
                self._predict(
                    images[0],
                    conf,
                    detect_ids[0],
                    device=device,
                    imgsz=imgsz,
                    overlap=overlap,
                    nms_iou=nms_iou,
                    max_det=max_det,
                    nms_agnostic=nms_agnostic,
                )
            ]
        kwargs: dict[str, Any] = {}
        if imgsz:
            kwargs["imgsz"] = imgsz
        if overlap:
            kwargs["overlap"] = overlap
        if nms_iou is None:
            nms_iou = self.nms_iou
        if max_det is None:
            max_det = self.max_det
        if nms_agnostic is None:
            nms_agnostic = self.nms_agnostic
        if nms_iou is not None:
            kwargs["iou"] = float(nms_iou)
        if max_det is not None:
            kwargs["max_det"] = int(max_det)
        if nms_agnostic is not None:
            kwargs["agnostic_nms"] = bool(nms_agnostic)
        use_imgsz = int(imgsz or 0)
        yolo_inputs = []
        scales: list[tuple[float, float]] = []
        for image in images:
            source_shape = image.shape[:2]
            yolo_in = preprocess_yolo_input(
                image,
                self.model_ch,
                gray_contrast_enhance=self.gray_contrast_enhance,
                clahe_clip=self.gray_clahe_clip,
                clahe_tile=self.gray_clahe_tile,
                target_imgsz=use_imgsz,
            )
            sx, sy = yolo_input_coord_scale(source_shape, yolo_in)
            yolo_inputs.append(yolo_in)
            scales.append((sx, sy))
        with model_infer_guard(self.model_path, task=self._infer_task):
            pred = self.model.predict(
                yolo_inputs,
                verbose=False,
                device=mps_safe_device(device or self.device, self.model_ch),
                augment=self.augment,
                half=self.half,
                conf=float(conf),
                **kwargs,
            )
        if not isinstance(pred, list):
            pred = [pred]
        batch_out: list[list] = []
        for idx, r in enumerate(pred):
            detect_id = detect_ids[idx] if idx < len(detect_ids) else detect_ids[-1]
            sx, sy = scales[idx] if idx < len(scales) else scales[-1]
            rows: list = []
            if r is not None and getattr(r, "boxes", None) is not None:
                for detection in r.boxes:
                    try:
                        box_conf = detection.conf.item()
                        if box_conf < conf:
                            continue
                        x1, y1, x2, y2 = detection.xyxy.tolist()[0]
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
                                box_conf,
                                int(detection.cls.item()),
                                detect_id,
                            ]
                        )
                    except Exception as e:
                        logging.error("parse batch box %s error: %s", detection, e)
            batch_out.append(rows)
        if len(batch_out) < len(images):
            batch_out.extend([[]] * (len(images) - len(batch_out)))
        return batch_out[: len(images)]

    def _infer_detect_sliding_tiles(
        self,
        tiles: list[_DetectClipTile],
        *,
        infer_conf: float,
        clip_batch_size: int,
        padding: bool,
        min_size: int,
        max_size: int | None,
        device,
        nms_iou: float | None,
        max_det: int | None,
        nms_agnostic: bool | None,
        all_box: list,
        debug_clip: bool = False,
    ) -> None:
        bs = max(1, int(clip_batch_size or 1))
        n_batches = (len(tiles) + bs - 1) // bs if tiles else 0
        logging.debug(
            "detect sliding tiles: count=%d batch_size=%d batches=%d",
            len(tiles),
            bs,
            n_batches,
        )
        for start in range(0, len(tiles), bs):
            chunk = tiles[start : start + bs]
            if bs <= 1 or len(chunk) == 1:
                batch_results = [
                    self._predict(
                        t.clip,
                        conf=infer_conf,
                        detect_id=t.detect_id,
                        device=device,
                        nms_iou=nms_iou,
                        max_det=max_det,
                        nms_agnostic=nms_agnostic,
                    )
                    for t in chunk
                ]
            else:
                batch_results = self._predict_batch(
                    [t.clip for t in chunk],
                    [t.detect_id for t in chunk],
                    conf=infer_conf,
                    device=device,
                    nms_iou=nms_iou,
                    max_det=max_det,
                    nms_agnostic=nms_agnostic,
                )
            for tile, results in zip(chunk, batch_results):
                if tile.pad_off_x or tile.pad_off_y:
                    for r in results:
                        r[0] -= tile.pad_off_x
                        r[1] -= tile.pad_off_y
                        r[2] -= tile.pad_off_x
                        r[3] -= tile.pad_off_y
                for result in results:
                    lr, flt, edge_min_dist = self._slice_local_box_prefilter(
                        result,
                        tile.actual_clip_w,
                        tile.actual_clip_h,
                        padding,
                        min_size,
                        max_size,
                    )
                    global_row = lr[:7] + [flt, edge_min_dist, int(tile.clip_size)]
                    global_row[0] += tile.clip_x1
                    global_row[1] += tile.clip_y1
                    global_row[2] += tile.clip_x1
                    global_row[3] += tile.clip_y1
                    all_box.append(global_row)
                if debug_clip:
                    import cv2

                    clip_debug = tile.clip.copy()
                    dx, dy = tile.pad_off_x, tile.pad_off_y
                    for result in results:
                        lr, flt, _edge_min_dist = self._slice_local_box_prefilter(
                            result,
                            tile.actual_clip_w,
                            tile.actual_clip_h,
                            padding,
                            min_size,
                            max_size,
                        )
                        bx1, by1, bx2, by2 = lr[:4]
                        px1, py1, px2, py2 = bx1 + dx, by1 + dy, bx2 + dx, by2 + dy
                        color = (180, 180, 180) if flt else (0, 0, 255)
                        cv2.rectangle(clip_debug, (px1, py1), (px2, py2), color, 2)
                    ph, pw = clip_debug.shape[:2]
                    window_name = (
                        f"clip x:{tile.clip_x1}-{tile.clip_x2} "
                        f"y:{tile.clip_y1}-{tile.clip_y2} padded:{pw}x{ph}"
                    )
                    cv2.imshow(window_name, clip_debug)
                    cv2.waitKey(0)
                    cv2.destroyWindow(window_name)

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

        if self.ior_threshold > 0:
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
            if len(pick) > 9 and pick[9] is not None:
                merged.append(pick[9])
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
                **(
                    {"clip_tile_size": int(box[9])}
                    if len(box) > 9 and box[9] is not None
                    else {}
                ),
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
        parsed = parse_clip_detect_id(detect_id)
        if parsed is None:
            return None
        return parsed[1], parsed[2]

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
        tile_cs = det.get("clip_tile_size")
        if tile_cs is not None:
            try:
                tile_cs = int(tile_cs)
            except (TypeError, ValueError):
                tile_cs = None
        effective_cs = tile_cs if tile_cs else clip_size
        if origin is None or not effective_cs:
            return None
        clip_x1, clip_y1 = origin
        clip_x2 = min(w, clip_x1 + effective_cs)
        clip_y2 = min(h, clip_y1 + effective_cs)
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
                padding=True, pad_full_image_to_square=False,
                nms_iou: float | None = None, max_det: int | None = None,
                nms_agnostic: bool | None = None,
                clip_profiles: list[ClipProfile] | None = None,
                clip_start: int = 0,
                clip_batch_size: int = 1,
                debug=False, debug_clip=False,
                min_size=5, max_size=None, edge_reject_distance=0,
                edge_reject_conf_threshold=None,
                device=None,
                return_all_rows: bool = False,
                ):
        """
        推理
        :param device:
        :param image:
        :param clip_size: 按照正方形切片，如果大于或者等于全图，不切
        :param overlap_size: 多个切片重叠区域像素大小
        :param padding: 默认 True。切片模式下边缘切片任一边小于 clip_size 则补成与分片边长一致的
            clip_size×clip_size 正方形；原图居中、空缺白底；再将检测框坐标移回切片局部坐标。
        :param pad_full_image_to_square: 默认 False。整图检测模式下（不切片）也先将整图白底补成正方形再检测，
            用于对齐训练/推理对 imgsz 的假设；检测框坐标会再映射回原图坐标系。
        :param edge_reject_distance: merge 之后切片边缘距离阈值（像素），<=0 表示不按距离滤除
        :param edge_reject_conf_threshold: merge 之后与距离联合过滤的置信度上限（不含）；
            None 时使用 self.conf_thresh。若需「仅按距离、忽略置信度」可传入大于 1 的值。
        :param clip_profiles: 多套切片；多于 1 套时依次滑窗后统一 merge
        :param clip_start: 单套滑窗时网格起始像素（x、y 同值）；``0`` 为 ``(0,0)``
        :param debug: 调试用
        :return:
        """
        w = image.shape[1]
        h = image.shape[0]
        if clip_profiles is not None and len(clip_profiles) > 1:
            return self._predict_multi_clip_profiles(
                image,
                w,
                h,
                clip_profiles,
                padding=padding,
                pad_full_image_to_square=pad_full_image_to_square,
                nms_iou=nms_iou,
                max_det=max_det,
                nms_agnostic=nms_agnostic,
                debug=debug,
                debug_clip=debug_clip,
                min_size=min_size,
                max_size=max_size,
                edge_reject_distance=edge_reject_distance,
                edge_reject_conf_threshold=edge_reject_conf_threshold,
                device=device,
                return_all_rows=return_all_rows,
                clip_batch_size=clip_batch_size,
            )
        if clip_profiles is not None and len(clip_profiles) == 1:
            clip_size, overlap_size, clip_start = unpack_clip_profile(clip_profiles[0])
        else:
            clip_start = max(0, int(clip_start or 0))
        all_box = []
        kwargs = {}
        infer_conf = self._infer_conf(return_all_rows)
        if not clip_size or not overlap_size or (clip_size >= w and clip_size >= h or clip_size <= overlap_size <= 1):
            # kwargs["imgsz"] = clip_size
            # kwargs["overlap"] = overlap_size/clip_size
            pad_off_x = pad_off_y = 0
            img_infer = image
            if pad_full_image_to_square and w != h:
                side = int(max(w, h))
                img_infer, pad_off_x, pad_off_y, _pw, _ph = _pad_tile_to_clip_square(
                    image, w, h, side
                )

            results = self._predict(
                img_infer,
                infer_conf,
                device=device,
                nms_iou=nms_iou,
                max_det=max_det,
                nms_agnostic=nms_agnostic,
                **kwargs,
            )
            results = self._convert_result(results)
            if pad_off_x or pad_off_y:
                for r in results:
                    r["x1"] = int(r["x1"]) - int(pad_off_x)
                    r["y1"] = int(r["y1"]) - int(pad_off_y)
                    r["x2"] = int(r["x2"]) - int(pad_off_x)
                    r["y2"] = int(r["y2"]) - int(pad_off_y)
            for r in results:
                if float(r.get("conf", 0.0)) < self.conf_thresh:
                    r["filter"] = True
                    continue
                w_b = max(0, int(r["x2"]) - int(r["x1"]))
                h_b = max(0, int(r["y2"]) - int(r["y1"]))
                if min_size and (w_b < min_size or h_b < min_size):
                    r["filter"] = True
                elif max_size and (w_b > max_size or h_b > max_size):
                    r["filter"] = True
            if debug:
                self.draw(image, results)

            if return_all_rows:
                return results
            return [r for r in results if not r.get("filter")]

        # 切片
        tiles = [
            _prepare_detect_clip_tile(
                image, clip_x1, clip_y1, clip_x2, clip_y2, clip_size, padding
            )
            for clip_x1, clip_y1, clip_x2, clip_y2 in get_clip(
                w, h, clip_size, overlap_size, clip_start=clip_start
            )
        ]
        self._infer_detect_sliding_tiles(
            tiles,
            infer_conf=infer_conf,
            clip_batch_size=clip_batch_size,
            padding=padding,
            min_size=min_size,
            max_size=max_size,
            device=device,
            nms_iou=nms_iou,
            max_det=max_det,
            nms_agnostic=nms_agnostic,
            all_box=all_box,
            debug_clip=debug_clip,
        )

        results = self.merge_ior(all_box)
        results = self._apply_post_merge_edge_filter(
            results, w, h, clip_size,
            edge_reject_distance=edge_reject_distance,
            edge_reject_conf_threshold=edge_reject_conf_threshold,
        )

        if debug:
            self.draw(image, results)
        if return_all_rows:
            return results
        return [r for r in results if not r.get("filter")]

    def _predict_multi_clip_profiles(
        self,
        image,
        w: int,
        h: int,
        profiles: list[ClipProfile],
        *,
        padding=True,
        pad_full_image_to_square=False,
        nms_iou: float | None = None,
        max_det: int | None = None,
        nms_agnostic: bool | None = None,
        debug=False,
        debug_clip=False,
        min_size=5,
        max_size=None,
        edge_reject_distance=0,
        edge_reject_conf_threshold=None,
        device=None,
        return_all_rows: bool = False,
        clip_batch_size: int = 1,
    ):
        """多套切片依次推理，最后统一 merge_ior 与边缘过滤。"""
        all_box = []
        infer_conf = self._infer_conf(return_all_rows)
        edge_clip_size = 0

        for profile_idx, profile in enumerate(profiles):
            clip_size, overlap_size, clip_start = unpack_clip_profile(profile)
            if uses_clip_inference_path(w, h, clip_size, overlap_size):
                edge_clip_size = max(edge_clip_size, int(clip_size))
                tiles = [
                    _prepare_detect_clip_tile(
                        image,
                        clip_x1,
                        clip_y1,
                        clip_x2,
                        clip_y2,
                        clip_size,
                        padding,
                        profile_idx=profile_idx,
                    )
                    for clip_x1, clip_y1, clip_x2, clip_y2 in get_clip(
                        w, h, clip_size, overlap_size, clip_start=clip_start
                    )
                ]
                self._infer_detect_sliding_tiles(
                    tiles,
                    infer_conf=infer_conf,
                    clip_batch_size=clip_batch_size,
                    padding=padding,
                    min_size=min_size,
                    max_size=max_size,
                    device=device,
                    nms_iou=nms_iou,
                    max_det=max_det,
                    nms_agnostic=nms_agnostic,
                    all_box=all_box,
                    debug_clip=debug_clip,
                )
                continue

            pad_off_x = pad_off_y = 0
            img_infer = image
            if pad_full_image_to_square and w != h:
                side = int(max(w, h))
                img_infer, pad_off_x, pad_off_y, _pw, _ph = _pad_tile_to_clip_square(
                    image, w, h, side
                )
            detect_id = make_clip_detect_id(profile_idx, 0, 0)
            results = self._predict(
                img_infer,
                infer_conf,
                detect_id=detect_id,
                device=device,
                nms_iou=nms_iou,
                max_det=max_det,
                nms_agnostic=nms_agnostic,
            )
            results = self._convert_result(results)
            for r in results:
                if pad_off_x or pad_off_y:
                    r["x1"] = int(r["x1"]) - int(pad_off_x)
                    r["y1"] = int(r["y1"]) - int(pad_off_y)
                    r["x2"] = int(r["x2"]) - int(pad_off_x)
                    r["y2"] = int(r["y2"]) - int(pad_off_y)
                if float(r.get("conf", 0.0)) < self.conf_thresh:
                    r["filter"] = True
                    continue
                w_b = max(0, int(r["x2"]) - int(r["x1"]))
                h_b = max(0, int(r["y2"]) - int(r["y1"]))
                if min_size and (w_b < min_size or h_b < min_size):
                    r["filter"] = True
                elif max_size and (w_b > max_size or h_b > max_size):
                    r["filter"] = True
                r["clip_tile_size"] = int(clip_size) if clip_size else None
                row = [
                    r["x1"],
                    r["y1"],
                    r["x2"],
                    r["y2"],
                    r["conf"],
                    r["cls_id"],
                    r["detect_id"],
                    r.get("filter", False),
                    None,
                ]
                if r.get("clip_tile_size"):
                    row.append(r["clip_tile_size"])
                all_box.append(row)

        results = self.merge_ior(all_box)
        results = self._apply_post_merge_edge_filter(
            results,
            w,
            h,
            edge_clip_size or max((cs for cs, _os in profiles), default=0),
            edge_reject_distance=edge_reject_distance,
            edge_reject_conf_threshold=edge_reject_conf_threshold,
        )
        if debug:
            self.draw(image, results)
        if return_all_rows:
            return results
        return [r for r in results if not r.get("filter")]

