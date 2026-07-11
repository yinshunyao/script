#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Detail  : YOLO 实例分割推理封装（滑窗、坐标回映射、跨切片 IoR 合并）。
#           与 detect/cls 解耦，仅供分割验证与业务脚本调用。

from __future__ import annotations

import logging
from typing import Any, NamedTuple

import cv2
import numpy as np
import torch
from script.predict.model_infer_lock import model_infer_guard
from script.predict.model_yolo_cache import get_cached_yolo
from script.predict.model_channel import (
    detect_model_input_channels,
    mps_safe_device,
    preprocess_yolo_input,
    scale_polygon_points,
    scale_xyxy,
    yolo_input_coord_scale,
)
from script.predict.model_detect import (
    ClipProfile,
    _pad_tile_to_clip_square,
    format_clip_slice_label_suffix,
    ior,
    make_clip_detect_id,
    resolve_profile_imgsz,
    unpack_clip_profile,
    uses_clip_inference_path,
)
from script.tools.roi_heyifei import iter_clip_tiles_for_image


class _SegClipTile(NamedTuple):
    """单张 seg 滑窗 tile（batch 推理单元）。"""

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
    profile_idx: int
    clip_tile_seq: int
    profile_total: int
    profile_imgsz: int | None


def _prepare_seg_clip_tile(
    image: np.ndarray,
    clip_x1: int,
    clip_y1: int,
    clip_x2: int,
    clip_y2: int,
    clip_size: int,
    padding: bool,
    *,
    profile_idx: int,
    clip_tile_seq: int,
    profile_total: int,
    profile_imgsz: int | None,
) -> _SegClipTile:
    clip = image[clip_y1:clip_y2, clip_x1:clip_x2]
    actual_clip_w = clip_x2 - clip_x1
    actual_clip_h = clip_y2 - clip_y1
    pad_off_x = pad_off_y = 0
    if padding and (actual_clip_w < clip_size or actual_clip_h < clip_size):
        clip, pad_off_x, pad_off_y, _pw, _ph = _pad_tile_to_clip_square(
            clip, actual_clip_w, actual_clip_h, clip_size
        )
    detect_id = make_clip_detect_id(profile_idx, clip_x1, clip_y1)
    return _SegClipTile(
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
        profile_idx,
        clip_tile_seq,
        profile_total,
        profile_imgsz,
    )


def _polygon_from_mask_xy(mask_xy: Any) -> list[list[int]]:
    if mask_xy is None:
        return []
    try:
        arr = np.asarray(mask_xy, dtype=np.float64)
    except Exception:
        return []
    if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] < 2:
        return []
    out: list[list[int]] = []
    for row in arr:
        out.append([int(round(float(row[0]))), int(round(float(row[1])))])
    return out


def _bbox_from_polygon(poly: list[list[int]]) -> tuple[int, int, int, int]:
    if not poly:
        return 0, 0, 0, 0
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def _sync_row_bbox_from_polygon(row: dict[str, Any]) -> dict[str, Any]:
    """有有效 polygon 时，将 x1..y2 同步为多边形轴对齐外接框。"""
    poly = list(row.get("polygon") or [])
    if len(poly) < 3:
        return row
    x1, y1, x2, y2 = _bbox_from_polygon(poly)
    out = dict(row)
    out["x1"], out["y1"], out["x2"], out["y2"] = x1, y1, x2, y2
    return out


def _shift_polygon(poly: list[list[int]], dx: int, dy: int) -> list[list[int]]:
    if not poly:
        return []
    return [[int(p[0]) + int(dx), int(p[1]) + int(dy)] for p in poly]


def _clip_polygon_to_rect(
    poly: list[list[int]],
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> list[list[int]]:
    """将多边形顶点裁剪到矩形 [x1,y1,x2,y2) 内（用于 padding 后去掉填充区外的点）。"""
    if not poly:
        return []
    w = max(1, int(x2) - int(x1))
    h = max(1, int(y2) - int(y1))
    clipped: list[list[int]] = []
    for px, py in poly:
        cx = min(max(int(px), int(x1)), int(x2) - 1)
        cy = min(max(int(py), int(y1)), int(y2) - 1)
        clipped.append([cx, cy])
    if len(clipped) < 3:
        return [[x1, y1], [x2 - 1, y1], [x2 - 1, y2 - 1], [x1, y2 - 1]]
    # 去掉连续重复点
    dedup: list[list[int]] = [clipped[0]]
    for pt in clipped[1:]:
        if pt != dedup[-1]:
            dedup.append(pt)
    if len(dedup) < 3:
        return clipped
    return dedup


def _contour_for_test(poly: list[list[int]], max_points: int) -> np.ndarray | None:
    """将多边形转为 cv2 轮廓数组 (K,1,2) float32，并等间隔下采样到 max_points 以控性能。"""
    if not poly or len(poly) < 3:
        return None
    arr = np.asarray(poly, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] < 2:
        return None
    if max_points and arr.shape[0] > int(max_points):
        idx = np.linspace(0, arr.shape[0] - 1, int(max_points)).round().astype(int)
        arr = arr[idx]
    return arr.reshape(-1, 1, 2)


def _sample_points(poly: list[list[int]], max_points: int) -> np.ndarray | None:
    """多边形顶点等间隔下采样为点集 (M,2) float32。"""
    if not poly or len(poly) < 3:
        return None
    arr = np.asarray(poly, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] < 2:
        return None
    if max_points and arr.shape[0] > int(max_points):
        idx = np.linspace(0, arr.shape[0] - 1, int(max_points)).round().astype(int)
        arr = arr[idx]
    return arr


def _frac_against_contour(
    points: np.ndarray, contour: np.ndarray, edge_px: float
) -> tuple[float, float]:
    """返回 (edge_frac, contain_frac)：points 中到 contour 边界距离 ≤edge_px 的占比、落在 contour 内部(含 1px 容差)的占比。"""
    if points is None or contour is None or len(points) == 0:
        return 0.0, 0.0
    near = 0
    inside = 0
    tol = 1.0
    for px, py in points:
        d = cv2.pointPolygonTest(contour, (float(px), float(py)), True)
        if abs(d) <= edge_px:
            near += 1
        if d >= -tol:
            inside += 1
    n = len(points)
    return near / n, inside / n


def _polygon_merge_similarity(
    poly_a: list[list[int]],
    poly_b: list[list[int]],
    *,
    edge_px: float,
    max_points: int,
) -> tuple[float, float]:
    """计算两多边形的 (边缘贴合比例, 包含率)，双向取较大值。退化多边形回退外接框四角。"""

    def _fallback(poly: list[list[int]]) -> list[list[int]]:
        if poly and len(poly) >= 3:
            return poly
        x1, y1, x2, y2 = _bbox_from_polygon(poly)
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

    pa = _fallback(poly_a)
    pb = _fallback(poly_b)
    cnt_a = _contour_for_test(pa, max_points)
    cnt_b = _contour_for_test(pb, max_points)
    pts_a = _sample_points(pa, max_points)
    pts_b = _sample_points(pb, max_points)
    if cnt_a is None or cnt_b is None:
        return 0.0, 0.0
    edge_ab, contain_ab = _frac_against_contour(pts_a, cnt_b, edge_px)
    edge_ba, contain_ba = _frac_against_contour(pts_b, cnt_a, edge_px)
    return max(edge_ab, edge_ba), max(contain_ab, contain_ba)


def _polygon_outside_circle_fraction(
    poly: list[list[int]],
    cx: int,
    cy: int,
    radius: int,
    *,
    max_points: int,
) -> float:
    """多边形采样点落在圆外的占比（用于识别跨 ROI 边界的实例）。"""
    roi_poly = _circle_to_polygon(cx, cy, radius, num_points=max(32, max_points))
    contour = _contour_for_test(roi_poly, max_points)
    if not poly or len(poly) < 3:
        x1, y1, x2, y2 = _bbox_from_polygon(poly)
        poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    pts = _sample_points(poly, max_points)
    if contour is None or pts is None or len(pts) == 0:
        return 0.0
    outside = 0
    for px, py in pts:
        if cv2.pointPolygonTest(contour, (float(px), float(py)), True) < -1.0:
            outside += 1
    return outside / len(pts)


def _same_clip_profile(row_a: dict[str, Any], row_b: dict[str, Any]) -> bool:
    """两检测是否来自同一 ``clip_profiles`` 套（缺省视为同套）。"""
    pa = row_a.get("clip_profile_idx")
    pb = row_b.get("clip_profile_idx")
    if pa is None or pb is None:
        return True
    return int(pa) == int(pb)


def _circle_to_polygon(
    cx: int,
    cy: int,
    radius: int,
    *,
    num_points: int = 64,
) -> list[list[int]]:
    """ROI 圆盘转多边形顶点（供边缘贴合判据复用 ``_polygon_merge_similarity``）。"""
    n = max(16, int(num_points))
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return [
        [int(round(cx + radius * float(np.cos(a)))), int(round(cy + radius * float(np.sin(a))))]
        for a in angles
    ]


class ModelSegmenter:
    """YOLO segment 任务封装：滑窗推理 + 多边形回映射 + 跨切片合并。"""

    def __init__(
        self,
        model_path: str,
        conf_thresh: float = 0.25,
        conf_merge: float = 0.1,
        conf_merge_draw: float = 0.01,
        ior_threshold: float = 0.5,
        device: str | None = None,
        augment: bool = False,
        *,
        nms_iou: float | None = None,
        max_det: int | None = None,
        nms_agnostic: bool | None = None,
        imgsz: int = 0,
        retina_masks: bool = False,
        poly_merge: bool = False,
        poly_merge_edge_px: float = 5.0,
        poly_merge_edge_ratio: float = 0.5,
        poly_merge_contain_ratio: float = 0.7,
        poly_merge_cross_class: bool = True,
        poly_merge_max_points: int = 80,
        roi_disk_edge_hug_ratio: float = 0.55,
        roi_disk_edge_outside_min: float = 0.02,
        filter_rows_by_roi_boundary: bool = True,
        gray_contrast_enhance: bool = False,
        gray_clahe_clip: float = 2.0,
        gray_clahe_tile: int = 8,
        gray_contrast_debug_save: bool = False,
    ):
        if device is None:
            self.device = self._auto_detect_device()
        else:
            self.device = device
        self.augment = bool(augment)
        self.model_path = str(model_path)
        self._infer_task = "segment"
        self.model = get_cached_yolo(model_path, task="segment")
        self.model_ch = detect_model_input_channels(self.model)
        if self.model_ch == 1:
            logging.info("分割模型为单通道(ch=1)，推理将以单通道灰度输入: %s", model_path)
            mps_safe_device(self.device, self.model_ch)
        self.conf_thresh = float(conf_thresh)
        self.conf_merge = float(conf_merge)
        self.conf_merge_draw = float(conf_merge_draw)
        self.ior_threshold = float(ior_threshold)
        self.nms_iou = nms_iou
        self.max_det = max_det
        self.nms_agnostic = nms_agnostic
        self.imgsz = int(imgsz or 0)
        self.retina_masks = bool(retina_masks)
        self.poly_merge = bool(poly_merge)
        self.poly_merge_edge_px = float(poly_merge_edge_px)
        self.poly_merge_edge_ratio = float(poly_merge_edge_ratio)
        self.poly_merge_contain_ratio = float(poly_merge_contain_ratio)
        self.poly_merge_cross_class = bool(poly_merge_cross_class)
        self.poly_merge_max_points = int(poly_merge_max_points)
        self.roi_disk_edge_hug_ratio = float(roi_disk_edge_hug_ratio)
        self.roi_disk_edge_outside_min = float(roi_disk_edge_outside_min)
        self._enable_roi_boundary_filter = bool(filter_rows_by_roi_boundary)
        self.gray_contrast_enhance = bool(gray_contrast_enhance)
        self.gray_clahe_clip = float(gray_clahe_clip)
        self.gray_clahe_tile = int(gray_clahe_tile)
        self.gray_contrast_debug_save = bool(gray_contrast_debug_save)
        if self.gray_contrast_enhance:
            logging.info(
                "分割推理已开启灰度 CLAHE 对比度增强 (clip=%.2f, tile=%d): %s",
                self.gray_clahe_clip,
                self.gray_clahe_tile,
                model_path,
            )
        logging.info("分割模型使用设备: %s", self.device)

    def _effective_imgsz(
        self,
        profile: ClipProfile | None,
        imgsz: int | None,
    ) -> int | None:
        if profile is not None:
            return resolve_profile_imgsz(profile, imgsz, model_default=self.imgsz)
        use = imgsz if imgsz is not None else self.imgsz
        return int(use) if use and int(use) > 0 else None

    @staticmethod
    def _auto_detect_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _infer_conf(self, return_all_rows: bool) -> float:
        """``return_all_rows`` 时用更低 ``conf_merge_draw`` 供 DRAW_FILTER 绘制。"""
        return self.conf_merge_draw if return_all_rows else self.conf_merge

    def _predict(
        self,
        image: np.ndarray,
        conf: float,
        detect_id: str = "0-0",
        device: str | None = None,
        *,
        imgsz: int | None = None,
        nms_iou: float | None = None,
        max_det: int | None = None,
        nms_agnostic: bool | None = None,
        retina_masks: bool | None = None,
        debug_save_stem: str | None = None,
        debug_save_dir: str | None = None,
    ) -> list[dict[str, Any]]:
        kwargs: dict[str, Any] = {}
        use_imgsz = int(imgsz if imgsz is not None else self.imgsz)
        if use_imgsz > 0:
            kwargs["imgsz"] = use_imgsz
        if nms_iou is None:
            nms_iou = self.nms_iou
        if max_det is None:
            max_det = self.max_det
        if nms_agnostic is None:
            nms_agnostic = self.nms_agnostic
        if retina_masks is None:
            retina_masks = self.retina_masks
        if nms_iou is not None:
            kwargs["iou"] = float(nms_iou)
        if max_det is not None:
            kwargs["max_det"] = int(max_det)
        if nms_agnostic is not None:
            kwargs["agnostic_nms"] = bool(nms_agnostic)
        # 默认关闭；仅显式打开时才传给 YOLO，保持历史行为
        if retina_masks:
            kwargs["retina_masks"] = True

        save_stem = None
        save_dir = None
        if self.gray_contrast_enhance and self.gray_contrast_debug_save and debug_save_dir:
            save_stem = debug_save_stem or detect_id.replace("-", "_")
            save_dir = debug_save_dir

        source_shape = image.shape[:2]
        yolo_in = preprocess_yolo_input(
            image,
            self.model_ch,
            gray_contrast_enhance=self.gray_contrast_enhance,
            clahe_clip=self.gray_clahe_clip,
            clahe_tile=self.gray_clahe_tile,
            target_imgsz=use_imgsz,
            debug_save_dir=save_dir,
            debug_save_stem=save_stem,
        )
        sx, sy = yolo_input_coord_scale(source_shape, yolo_in)

        with model_infer_guard(self.model_path, task=self._infer_task):
            pred = self.model.predict(
                yolo_in,
                verbose=False,
                device=mps_safe_device(device or self.device, self.model_ch),
                augment=self.augment,
                conf=float(conf),
                **kwargs,
            )
        out: list[dict[str, Any]] = []
        if not pred or pred[0].boxes is None or len(pred[0].boxes) == 0:
            return out

        r0 = pred[0]
        has_masks = r0.masks is not None
        n = len(r0.boxes)
        for i in range(n):
            try:
                box_conf = float(r0.boxes.conf[i].item())
            except Exception:
                continue
            if box_conf < float(conf):
                continue
            cls_id = int(r0.boxes.cls[i].item())
            x1, y1, x2, y2 = [float(v) for v in r0.boxes.xyxy[i].tolist()]
            if sx != 1.0 or sy != 1.0:
                x1, y1, x2, y2 = scale_xyxy(x1, y1, x2, y2, sx, sy)
            else:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            row: dict[str, Any] = {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "conf": box_conf,
                "cls_id": cls_id,
                "class_name": self.model.names[cls_id],
                "detect_id": detect_id,
            }
            poly: list[list[float | int]] = []
            if has_masks and r0.masks.xy is not None and i < len(r0.masks.xy):
                poly = _polygon_from_mask_xy(r0.masks.xy[i])
            if not poly:
                poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            elif sx != 1.0 or sy != 1.0:
                poly = scale_polygon_points(poly, sx, sy)
            row["polygon"] = poly
            out.append(row)
        return out

    def _predict_batch(
        self,
        images: list[np.ndarray],
        detect_ids: list[str],
        conf: float,
        device: str | None = None,
        *,
        imgsz: int | None = None,
        nms_iou: float | None = None,
        max_det: int | None = None,
        nms_agnostic: bool | None = None,
        retina_masks: bool | None = None,
    ) -> list[list[dict[str, Any]]]:
        """多 tile YOLO seg batch；单张时退化为 ``_predict``。"""
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
                    nms_iou=nms_iou,
                    max_det=max_det,
                    nms_agnostic=nms_agnostic,
                    retina_masks=retina_masks,
                )
            ]
        kwargs: dict[str, Any] = {}
        use_imgsz = int(imgsz if imgsz is not None else self.imgsz)
        if use_imgsz > 0:
            kwargs["imgsz"] = use_imgsz
        if nms_iou is None:
            nms_iou = self.nms_iou
        if max_det is None:
            max_det = self.max_det
        if nms_agnostic is None:
            nms_agnostic = self.nms_agnostic
        if retina_masks is None:
            retina_masks = self.retina_masks
        if nms_iou is not None:
            kwargs["iou"] = float(nms_iou)
        if max_det is not None:
            kwargs["max_det"] = int(max_det)
        if nms_agnostic is not None:
            kwargs["agnostic_nms"] = bool(nms_agnostic)
        if retina_masks:
            kwargs["retina_masks"] = True

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
                conf=float(conf),
                **kwargs,
            )
        if not isinstance(pred, list):
            pred = [pred]
        batch_out: list[list[dict[str, Any]]] = []
        for idx, r in enumerate(pred):
            detect_id = detect_ids[idx] if idx < len(detect_ids) else detect_ids[-1]
            sx, sy = scales[idx] if idx < len(scales) else scales[-1]
            rows: list[dict[str, Any]] = []
            if r is not None and r.boxes is not None and len(r.boxes) > 0:
                has_masks = r.masks is not None
                n = len(r.boxes)
                for i in range(n):
                    try:
                        box_conf = float(r.boxes.conf[i].item())
                    except Exception:
                        continue
                    if box_conf < float(conf):
                        continue
                    cls_id = int(r.boxes.cls[i].item())
                    x1, y1, x2, y2 = [float(v) for v in r.boxes.xyxy[i].tolist()]
                    if sx != 1.0 or sy != 1.0:
                        x1, y1, x2, y2 = scale_xyxy(x1, y1, x2, y2, sx, sy)
                    else:
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    row: dict[str, Any] = {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "conf": box_conf,
                        "cls_id": cls_id,
                        "class_name": self.model.names[cls_id],
                        "detect_id": detect_id,
                    }
                    poly: list[list[float | int]] = []
                    if has_masks and r.masks.xy is not None and i < len(r.masks.xy):
                        poly = _polygon_from_mask_xy(r.masks.xy[i])
                    if not poly:
                        poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                    elif sx != 1.0 or sy != 1.0:
                        poly = scale_polygon_points(poly, sx, sy)
                    row["polygon"] = poly
                    rows.append(row)
            batch_out.append(rows)
        if len(batch_out) < len(images):
            batch_out.extend([[]] * (len(images) - len(batch_out)))
        return batch_out[: len(images)]

    def _append_seg_tile_rows(
        self,
        tile: "_SegClipTile",
        local_rows: list[dict[str, Any]],
        *,
        padding: bool,
        min_size: int,
        max_size: int | None,
        all_rows: list[dict[str, Any]],
    ) -> None:
        for r in local_rows:
            if tile.pad_off_x or tile.pad_off_y:
                r = dict(r)
                r["polygon"] = _shift_polygon(
                    r.get("polygon") or [], -tile.pad_off_x, -tile.pad_off_y
                )
                r["x1"] = int(r["x1"]) - tile.pad_off_x
                r["y1"] = int(r["y1"]) - tile.pad_off_y
                r["x2"] = int(r["x2"]) - tile.pad_off_x
                r["y2"] = int(r["y2"]) - tile.pad_off_y
            r, flt = self._slice_local_prefilter(
                r,
                tile.actual_clip_w,
                tile.actual_clip_h,
                padding,
                min_size,
                max_size,
            )
            r = dict(r)
            poly = _shift_polygon(list(r.get("polygon") or []), tile.clip_x1, tile.clip_y1)
            r["polygon"] = poly
            r["x1"] = int(r["x1"]) + tile.clip_x1
            r["y1"] = int(r["y1"]) + tile.clip_y1
            r["x2"] = int(r["x2"]) + tile.clip_x1
            r["y2"] = int(r["y2"]) + tile.clip_y1
            r["filter"] = bool(flt)
            r["clip_tile_size"] = int(tile.clip_size)
            r["clip_profile_idx"] = tile.profile_idx
            r["clip_tile_seq"] = tile.clip_tile_seq
            r["clip_profile_total"] = tile.profile_total
            all_rows.append(r)

    def _infer_seg_sliding_tiles(
        self,
        tiles: list["_SegClipTile"],
        *,
        infer_conf: float,
        clip_batch_size: int,
        padding: bool,
        min_size: int,
        max_size: int | None,
        device: str | None,
        nms_iou: float | None,
        max_det: int | None,
        nms_agnostic: bool | None,
        retina_masks: bool | None,
        all_rows: list[dict[str, Any]],
        debug_clip: bool = False,
    ) -> None:
        bs = max(1, int(clip_batch_size or 1))
        for start in range(0, len(tiles), bs):
            chunk = tiles[start : start + bs]
            if bs <= 1 or len(chunk) == 1:
                batch_rows = [
                    self._predict(
                        t.clip,
                        infer_conf,
                        detect_id=t.detect_id,
                        device=device,
                        imgsz=t.profile_imgsz,
                        nms_iou=nms_iou,
                        max_det=max_det,
                        nms_agnostic=nms_agnostic,
                        retina_masks=retina_masks,
                    )
                    for t in chunk
                ]
            else:
                imgsz0 = chunk[0].profile_imgsz
                if any(t.profile_imgsz != imgsz0 for t in chunk):
                    batch_rows = [
                        self._predict(
                            t.clip,
                            infer_conf,
                            detect_id=t.detect_id,
                            device=device,
                            imgsz=t.profile_imgsz,
                            nms_iou=nms_iou,
                            max_det=max_det,
                            nms_agnostic=nms_agnostic,
                            retina_masks=retina_masks,
                        )
                        for t in chunk
                    ]
                else:
                    batch_rows = self._predict_batch(
                        [t.clip for t in chunk],
                        [t.detect_id for t in chunk],
                        infer_conf,
                        device=device,
                        imgsz=imgsz0,
                        nms_iou=nms_iou,
                        max_det=max_det,
                        nms_agnostic=nms_agnostic,
                        retina_masks=retina_masks,
                    )
            for tile, local_rows in zip(chunk, batch_rows):
                self._append_seg_tile_rows(
                    tile,
                    local_rows,
                    padding=padding,
                    min_size=min_size,
                    max_size=max_size,
                    all_rows=all_rows,
                )
                if debug_clip:
                    self._debug_draw_clip(
                        tile.clip,
                        local_rows,
                        tile.pad_off_x,
                        tile.pad_off_y,
                        tile.clip_x1,
                        tile.clip_y1,
                        tile.clip_x2,
                        tile.clip_y2,
                        clip_tile_seq=tile.clip_tile_seq,
                        clip_profile_idx=tile.profile_idx,
                        clip_profile_total=tile.profile_total,
                    )

    @staticmethod
    def _box_row_filtered(row: dict[str, Any]) -> bool:
        return bool(row.get("filter"))

    def _slice_local_prefilter(
        self,
        row: dict[str, Any],
        actual_clip_w: int,
        actual_clip_h: int,
        padding: bool,
        min_size: int | None,
        max_size: int | None,
    ) -> tuple[dict[str, Any], bool]:
        r = dict(row)
        if float(r.get("conf", 0.0)) < self.conf_thresh:
            return r, True

        poly = list(r.get("polygon") or [])
        if padding and poly:
            poly = _clip_polygon_to_rect(poly, 0, 0, actual_clip_w, actual_clip_h)
            r["polygon"] = poly

        if poly:
            x1, y1, x2, y2 = _bbox_from_polygon(poly)
        else:
            try:
                x1, y1, x2, y2 = (
                    int(r["x1"]),
                    int(r["y1"]),
                    int(r["x2"]),
                    int(r["y2"]),
                )
            except (KeyError, TypeError, ValueError):
                return r, True
        r["x1"], r["y1"], r["x2"], r["y2"] = x1, y1, x2, y2

        w_b = max(0, x2 - x1)
        h_b = max(0, y2 - y1)
        if min_size and (w_b < min_size or h_b < min_size):
            return r, True
        if max_size and (w_b > max_size or h_b > max_size):
            return r, True
        return r, False

    def merge_ior(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """同类、不同切片、IoR 超阈值时合并；保留置信度最高实例的多边形与外接框。"""
        n = len(rows)
        if n == 0:
            return []

        parent = list(range(n))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        if self.ior_threshold > 0:
            for i in range(n):
                for j in range(i + 1, n):
                    if self._box_row_filtered(rows[i]) or self._box_row_filtered(rows[j]):
                        continue
                    if rows[i].get("detect_id") == rows[j].get("detect_id"):
                        continue
                    if not _same_clip_profile(rows[i], rows[j]):
                        continue
                    if int(rows[i].get("cls_id", -1)) != int(rows[j].get("cls_id", -1)):
                        continue
                    bi = [
                        rows[i]["x1"],
                        rows[i]["y1"],
                        rows[i]["x2"],
                        rows[i]["y2"],
                    ]
                    bj = [
                        rows[j]["x1"],
                        rows[j]["y1"],
                        rows[j]["x2"],
                        rows[j]["y2"],
                    ]
                    if ior(bi, bj) > self.ior_threshold:
                        union(i, j)

        clusters: dict[int, list[int]] = {}
        for i in range(n):
            clusters.setdefault(find(i), []).append(i)

        merged: list[dict[str, Any]] = []
        for idxs in clusters.values():
            group = [rows[i] for i in idxs]
            pick = max(group, key=lambda x: float(x.get("conf", 0.0)))
            max_conf = max(float(x.get("conf", 0.0)) for x in group)
            flt = any(self._box_row_filtered(x) for x in group)
            out = dict(pick)
            out.update(
                {
                    "conf": float(max_conf),
                    "filter": bool(flt),
                }
            )
            out = _sync_row_bbox_from_polygon(out)
            merged.append(out)
        return merged

    def merge_polygon_similar(
        self, rows: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """多边形相似合并：边缘贴合比例或包含率达标即判为同一目标并合并。

        与 ``merge_ior`` 区别：不跳过 ``detect_id`` 相同（解决整图同帧重复检出），
        判据基于多边形轮廓而非 bbox IoR。``poly_merge=False`` 时直接返回原列表。
        """
        if not self.poly_merge:
            return rows
        n = len(rows)
        if n < 2:
            return rows

        edge_px = self.poly_merge_edge_px
        edge_ratio = self.poly_merge_edge_ratio
        contain_ratio = self.poly_merge_contain_ratio
        max_points = self.poly_merge_max_points

        parent = list(range(n))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        def _bbox_near(bi: list[int], bj: list[int]) -> bool:
            # 各按 edge_px 外扩后判断相交，避免对远离实例算多边形距离
            return not (
                bi[2] + edge_px < bj[0]
                or bj[2] + edge_px < bi[0]
                or bi[3] + edge_px < bj[1]
                or bj[3] + edge_px < bi[1]
            )

        for i in range(n):
            for j in range(i + 1, n):
                if self._box_row_filtered(rows[i]) or self._box_row_filtered(rows[j]):
                    continue
                if not self.poly_merge_cross_class and int(
                    rows[i].get("cls_id", -1)
                ) != int(rows[j].get("cls_id", -1)):
                    continue
                bi = [
                    int(rows[i]["x1"]),
                    int(rows[i]["y1"]),
                    int(rows[i]["x2"]),
                    int(rows[i]["y2"]),
                ]
                bj = [
                    int(rows[j]["x1"]),
                    int(rows[j]["y1"]),
                    int(rows[j]["x2"]),
                    int(rows[j]["y2"]),
                ]
                if not _bbox_near(bi, bj):
                    continue
                edge_frac, contain_frac = _polygon_merge_similarity(
                    list(rows[i].get("polygon") or []),
                    list(rows[j].get("polygon") or []),
                    edge_px=edge_px,
                    max_points=max_points,
                )
                if _same_clip_profile(rows[i], rows[j]):
                    should_merge = (
                        edge_frac >= edge_ratio or contain_frac >= contain_ratio
                    )
                else:
                    # 跨 profile 仅用包含率去重，避免 edge 贴合误并相邻虫体
                    should_merge = contain_frac >= contain_ratio
                if should_merge:
                    union(i, j)

        clusters: dict[int, list[int]] = {}
        for i in range(n):
            clusters.setdefault(find(i), []).append(i)

        merged: list[dict[str, Any]] = []
        for idxs in clusters.values():
            group = [rows[i] for i in idxs]
            if len(group) == 1:
                merged.append(group[0])
                continue
            pick = max(group, key=lambda x: float(x.get("conf", 0.0)))
            max_conf = max(float(x.get("conf", 0.0)) for x in group)
            flt = any(self._box_row_filtered(x) for x in group)
            out = dict(pick)
            out.update(
                {
                    "conf": float(max_conf),
                    "filter": bool(flt),
                }
            )
            out = _sync_row_bbox_from_polygon(out)
            merged.append(out)
        return merged

    def filter_rows_by_roi_boundary(
        self,
        rows: list[dict[str, Any]],
        roi_circle: tuple[int, int, int] | None,
    ) -> list[dict[str, Any]]:
        """
        ROI 圆盘边缘误报过滤（``roi_circle`` 有效时生效，不依赖 ``poly_merge``）。

        - **跨边界**：分割多边形贴圆盘且采样点有显著比例落在圆外；
        - **贴边**：分割曲线与圆盘边缘高度重合（典型圆盘弧边误分割），即使主要在圆内也过滤。
        """
        if (
            not self._enable_roi_boundary_filter
            or self.roi_disk_edge_hug_ratio <= 0
        ):
            return rows
        if roi_circle is None:
            return rows
        cx, cy, radius = roi_circle
        roi_poly = _circle_to_polygon(
            cx, cy, radius, num_points=max(32, self.poly_merge_max_points)
        )
        edge_px = self.poly_merge_edge_px
        edge_ratio = self.poly_merge_edge_ratio
        hug_ratio = self.roi_disk_edge_hug_ratio
        outside_min = self.roi_disk_edge_outside_min
        max_points = self.poly_merge_max_points

        out: list[dict[str, Any]] = []
        filtered_n = 0
        for row in rows:
            if row.get("filter"):
                out.append(row)
                continue
            poly = list(row.get("polygon") or [])
            edge_frac, _ = _polygon_merge_similarity(
                poly,
                roi_poly,
                edge_px=edge_px,
                max_points=max_points,
            )
            outside_frac = _polygon_outside_circle_fraction(
                poly, cx, cy, radius, max_points=max_points
            )
            cross_boundary = edge_frac >= edge_ratio and outside_frac > outside_min
            edge_hug = edge_frac >= hug_ratio
            if cross_boundary or edge_hug:
                nr = dict(row)
                nr["filter"] = True
                nr["filter_reason"] = "roi_edge"
                nr["roi_edge_frac"] = float(edge_frac)
                nr["roi_outside_frac"] = float(outside_frac)
                nr["roi_edge_mode"] = (
                    "cross_boundary" if cross_boundary else "edge_hug"
                )
                out.append(nr)
                filtered_n += 1
            else:
                out.append(row)
        if filtered_n:
            logging.info(
                "ROI 圆盘边缘误报过滤 %d 条 (cross: edge>=%.2f & outside>%.2f; "
                "hug: edge>=%.2f)",
                filtered_n,
                edge_ratio,
                outside_min,
                hug_ratio,
            )
        return out

    def _finish_segment_predict_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        image: np.ndarray,
        roi_circle: tuple[int, int, int] | None,
        return_all_rows: bool,
        debug: bool,
    ) -> list[dict[str, Any]]:
        rows = self.filter_rows_by_roi_boundary(rows, roi_circle)
        if debug:
            self._debug_draw(image, rows)
        if return_all_rows:
            return rows
        return [r for r in rows if not r.get("filter")]

    def predict(
        self,
        image: np.ndarray,
        clip_size: int = 960,
        overlap_size: int = 200,
        *,
        clip_profiles: list[ClipProfile] | None = None,
        clip_start: int = 0,
        padding: bool = True,
        pad_full_image_to_square: bool = False,
        min_size: int = 3,
        max_size: int | None = None,
        device: str | None = None,
        imgsz: int | None = None,
        nms_iou: float | None = None,
        max_det: int | None = None,
        nms_agnostic: bool | None = None,
        retina_masks: bool | None = None,
        debug: bool = False,
        debug_clip: bool = False,
        return_all_rows: bool = False,
        debug_image_stem: str | None = None,
        gray_contrast_debug_dir: str | None = None,
        roi_circle: tuple[int, int, int] | None = None,
        clip_batch_size: int = 1,
    ) -> list[dict[str, Any]]:
        """
        整图或滑窗分割推理，返回全图坐标下的实例列表。

        每条结果字段：
        - polygon: [[x,y], ...] 像素坐标
        - x1,y1,x2,y2: 外接框
        - conf, cls_id, cls_name, detect_id, filter

        ``return_all_rows=True`` 时保留 ``filter=True`` 的实例（供 ``DRAW_FILTER`` 绘制）；
        整图与切片路径均先用 ``conf_merge`` 低阈值出框，再按 ``conf_thresh`` 等标记 filter。
        """
        h, w = image.shape[:2]
        all_rows: list[dict[str, Any]] = []
        infer_conf = self._infer_conf(return_all_rows)

        if clip_profiles is not None and len(clip_profiles) > 1:
            return self._predict_multi_clip_profiles(
                image,
                w,
                h,
                clip_profiles,
                infer_conf=infer_conf,
                padding=padding,
                pad_full_image_to_square=pad_full_image_to_square,
                min_size=min_size,
                max_size=max_size,
                device=device,
                imgsz=imgsz,
                nms_iou=nms_iou,
                max_det=max_det,
                nms_agnostic=nms_agnostic,
                retina_masks=retina_masks,
                debug=debug,
                debug_clip=debug_clip,
                return_all_rows=return_all_rows,
                debug_image_stem=debug_image_stem,
                gray_contrast_debug_dir=gray_contrast_debug_dir,
                roi_circle=roi_circle,
                clip_batch_size=clip_batch_size,
            )
        if clip_profiles is not None and len(clip_profiles) == 1:
            clip_size, overlap_size, clip_start = unpack_clip_profile(clip_profiles[0])
            imgsz_eff = self._effective_imgsz(clip_profiles[0], imgsz)
        else:
            imgsz_eff = self._effective_imgsz(None, imgsz)
            clip_start = max(0, int(clip_start or 0))

        use_clip = bool(
            clip_size
            and overlap_size
            and not (clip_size >= w and clip_size >= h)
            and not (clip_size <= overlap_size <= 1)
        )

        if not use_clip:
            pad_off_x = pad_off_y = 0
            img_infer = image
            if pad_full_image_to_square and w != h:
                side = int(max(w, h))
                img_infer, pad_off_x, pad_off_y, _pw, _ph = _pad_tile_to_clip_square(
                    image, w, h, side
                )
            local_rows = self._predict(
                img_infer,
                infer_conf,
                device=device,
                imgsz=imgsz_eff,
                nms_iou=nms_iou,
                max_det=max_det,
                nms_agnostic=nms_agnostic,
                retina_masks=retina_masks,
                debug_save_stem=debug_image_stem,
                debug_save_dir=gray_contrast_debug_dir,
            )
            for r in local_rows:
                if pad_off_x or pad_off_y:
                    r["polygon"] = _shift_polygon(
                        r.get("polygon") or [], -pad_off_x, -pad_off_y
                    )
                    r["x1"] = int(r["x1"]) - pad_off_x
                    r["y1"] = int(r["y1"]) - pad_off_y
                    r["x2"] = int(r["x2"]) - pad_off_x
                    r["y2"] = int(r["y2"]) - pad_off_y
                r, flt = self._slice_local_prefilter(
                    r, w, h, padding=False, min_size=min_size, max_size=max_size
                )
                r["filter"] = bool(flt)
                all_rows.append(r)
            all_rows = self.merge_polygon_similar(all_rows)
            return self._finish_segment_predict_rows(
                all_rows,
                image=image,
                roi_circle=roi_circle,
                return_all_rows=return_all_rows,
                debug=debug,
            )

        tiles = [
            _prepare_seg_clip_tile(
                image,
                clip_x1,
                clip_y1,
                clip_x2,
                clip_y2,
                clip_size,
                padding,
                profile_idx=0,
                clip_tile_seq=clip_tile_seq,
                profile_total=1,
                profile_imgsz=imgsz_eff,
            )
            for clip_tile_seq, (clip_x1, clip_y1, clip_x2, clip_y2) in enumerate(
                iter_clip_tiles_for_image(
                    w,
                    h,
                    clip_size,
                    overlap_size,
                    clip_start=clip_start,
                    roi_circle=roi_circle,
                )
            )
        ]
        self._infer_seg_sliding_tiles(
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
            retina_masks=retina_masks,
            all_rows=all_rows,
            debug_clip=debug_clip,
        )

        results = self.merge_ior(all_rows)
        results = self.merge_polygon_similar(results)
        return self._finish_segment_predict_rows(
            results,
            image=image,
            roi_circle=roi_circle,
            return_all_rows=return_all_rows,
            debug=debug,
        )

    def _predict_multi_clip_profiles(
        self,
        image: np.ndarray,
        w: int,
        h: int,
        profiles: list[ClipProfile],
        *,
        infer_conf: float,
        padding: bool,
        pad_full_image_to_square: bool,
        min_size: int,
        max_size: int | None,
        device: str | None,
        imgsz: int | None,
        nms_iou: float | None,
        max_det: int | None,
        nms_agnostic: bool | None,
        retina_masks: bool | None,
        debug: bool,
        debug_clip: bool,
        return_all_rows: bool,
        debug_image_stem: str | None,
        gray_contrast_debug_dir: str | None,
        roi_circle: tuple[int, int, int] | None = None,
        clip_batch_size: int = 1,
    ) -> list[dict[str, Any]]:
        """多套切片依次推理，最后统一 merge_ior 与 merge_polygon_similar。"""
        all_rows: list[dict[str, Any]] = []
        profile_total = len(profiles)

        for profile_idx, profile in enumerate(profiles):
            clip_size, overlap_size, clip_start = unpack_clip_profile(profile)
            profile_imgsz = self._effective_imgsz(profile, imgsz)
            if uses_clip_inference_path(w, h, clip_size, overlap_size):
                tiles = [
                    _prepare_seg_clip_tile(
                        image,
                        clip_x1,
                        clip_y1,
                        clip_x2,
                        clip_y2,
                        clip_size,
                        padding,
                        profile_idx=profile_idx,
                        clip_tile_seq=clip_tile_seq,
                        profile_total=profile_total,
                        profile_imgsz=profile_imgsz,
                    )
                    for clip_tile_seq, (clip_x1, clip_y1, clip_x2, clip_y2) in enumerate(
                        iter_clip_tiles_for_image(
                            w,
                            h,
                            clip_size,
                            overlap_size,
                            clip_start=clip_start,
                            roi_circle=roi_circle,
                        )
                    )
                ]
                self._infer_seg_sliding_tiles(
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
                    retina_masks=retina_masks,
                    all_rows=all_rows,
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
            local_rows = self._predict(
                img_infer,
                infer_conf,
                detect_id=detect_id,
                device=device,
                imgsz=profile_imgsz,
                nms_iou=nms_iou,
                max_det=max_det,
                nms_agnostic=nms_agnostic,
                retina_masks=retina_masks,
                debug_save_stem=debug_image_stem,
                debug_save_dir=gray_contrast_debug_dir,
            )
            for r in local_rows:
                if pad_off_x or pad_off_y:
                    r["polygon"] = _shift_polygon(
                        r.get("polygon") or [], -pad_off_x, -pad_off_y
                    )
                    r["x1"] = int(r["x1"]) - pad_off_x
                    r["y1"] = int(r["y1"]) - pad_off_y
                    r["x2"] = int(r["x2"]) - pad_off_x
                    r["y2"] = int(r["y2"]) - pad_off_y
                r, flt = self._slice_local_prefilter(
                    r, w, h, padding=False, min_size=min_size, max_size=max_size
                )
                r["filter"] = bool(flt)
                if clip_size:
                    r["clip_tile_size"] = int(clip_size)
                r["clip_profile_idx"] = profile_idx
                r["clip_tile_seq"] = 0
                r["clip_profile_total"] = profile_total
                all_rows.append(r)

        results = self.merge_ior(all_rows)
        results = self.merge_polygon_similar(results)
        return self._finish_segment_predict_rows(
            results,
            image=image,
            roi_circle=roi_circle,
            return_all_rows=return_all_rows,
            debug=debug,
        )

    @staticmethod
    def _debug_draw(image: np.ndarray, rows: list[dict[str, Any]]) -> None:
        img = image.copy()
        for r in rows:
            poly = np.array(r.get("polygon") or [], dtype=np.int32)
            color = (180, 180, 180) if r.get("filter") else (0, 0, 255)
            if poly.shape[0] >= 3:
                cv2.polylines(img, [poly], True, color, 2)
            cv2.putText(
                img,
                f"{r.get('class_name', '')}-{float(r.get('conf', 0)):.2f}"
                f"{format_clip_slice_label_suffix(r)}",
                (int(r["x1"]), max(0, int(r["y1"]) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                1,
            )
        cv2.imshow("seg", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _debug_draw_clip(
        self,
        clip: np.ndarray,
        local_rows: list[dict[str, Any]],
        pad_off_x: int,
        pad_off_y: int,
        clip_x1: int,
        clip_y1: int,
        clip_x2: int,
        clip_y2: int,
        *,
        clip_tile_seq: int | None = None,
        clip_profile_idx: int = 0,
        clip_profile_total: int = 1,
    ) -> None:
        img = clip.copy()
        for r in local_rows:
            poly = _shift_polygon(r.get("polygon") or [], pad_off_x, pad_off_y)
            arr = np.array(poly, dtype=np.int32)
            if arr.shape[0] >= 3:
                cv2.polylines(img, [arr], True, (0, 0, 255), 2)
        seq_label = ""
        if clip_tile_seq is not None:
            seq = int(clip_tile_seq) + 1
            if int(clip_profile_total) > 1:
                seq_label = f" #{int(clip_profile_idx) + 1}-{seq}"
            else:
                seq_label = f" #{seq}"
        title = f"clip{seq_label} {clip_x1}-{clip_x2} {clip_y1}-{clip_y2}"
        cv2.imshow(title, img)
        cv2.waitKey(0)
        cv2.destroyWindow(title)
