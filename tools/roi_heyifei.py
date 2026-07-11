#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Detail  : 推理前 ROI 预处理（插件式）。当前支持「黑框圆盘」：二值化定位圆盘，圆外置白。

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import cv2
import numpy as np

RoiPluginFn = Callable[[np.ndarray, dict[str, Any]], "RoiDetectResult | None"]

_ROI_PLUGINS: dict[str, RoiPluginFn] = {}


@dataclass(frozen=True)
class RoiDetectResult:
    """单张 ROI 插件识别结果。"""

    roi_type: str
    center: tuple[int, int]
    radius: int
    score: float = 1.0


@dataclass(frozen=True)
class RoiApplyResult:
    """整图 ROI 预处理输出。"""

    image: np.ndarray
    applied: bool
    roi_type: str | None = None
    center: tuple[int, int] | None = None
    radius: int | None = None
    score: float = 0.0


def register_roi_plugin(name: str, fn: RoiPluginFn) -> None:
    key = str(name or "").strip()
    if not key:
        raise ValueError("ROI 插件名不能为空")
    _ROI_PLUGINS[key] = fn


def list_roi_plugins() -> tuple[str, ...]:
    return tuple(sorted(_ROI_PLUGINS.keys()))


def resolve_roi_options(cfg: dict[str, Any] | None) -> dict[str, Any]:
    """从根模型 JSON 解析 ROI 开关与参数。"""
    c = cfg or {}
    plugins = c.get("roi_plugins")
    if plugins is None:
        plugin_names = ("disk_circle",)
    elif isinstance(plugins, str):
        plugin_names = (plugins.strip(),) if plugins.strip() else ("disk_circle",)
    elif isinstance(plugins, (list, tuple)):
        plugin_names = tuple(
            str(x).strip() for x in plugins if str(x or "").strip()
        ) or ("disk_circle",)
    else:
        plugin_names = ("disk_circle",)

    fill = c.get("roi_fill_color")
    if fill is None:
        fill_bgr = (255, 255, 255)
    elif isinstance(fill, (list, tuple)) and len(fill) >= 3:
        fill_bgr = (int(fill[0]), int(fill[1]), int(fill[2]))
    else:
        fill_bgr = (255, 255, 255)

    return {
        "enabled": bool(c.get("roi_switch", False)),
        "plugins": plugin_names,
        "fill_bgr": fill_bgr,
        "disk_circle": {
            "blur_ksize": int(c.get("roi_disk_blur_ksize", 9)),
            "threshold": c.get("roi_disk_threshold"),  # None -> Otsu（轮廓回退路径）
            "min_radius_ratio": float(c.get("roi_disk_min_radius_ratio", 0.28)),
            "max_radius_ratio": float(c.get("roi_disk_max_radius_ratio", 0.52)),
            "center_tol_ratio": float(c.get("roi_disk_center_tol_ratio", 0.12)),
            "border_margin_ratio": float(c.get("roi_disk_border_margin_ratio", 0.05)),
            "max_border_mean": float(c.get("roi_disk_max_border_mean", 120.0)),
            "min_contrast": float(c.get("roi_disk_min_contrast", 35.0)),
            "min_circularity": float(c.get("roi_disk_min_circularity", 0.58)),
            "min_score": float(c.get("roi_disk_min_score", 0.50)),
            "hough_dp": float(c.get("roi_disk_hough_dp", 1.2)),
            "hough_param1": float(c.get("roi_disk_hough_param1", 80.0)),
            "hough_param2": float(c.get("roi_disk_hough_param2", 40.0)),
            # Hough 在全分辨率上极慢；检测阶段缩放到该边长（0=不缩放）
            "detect_max_side": int(c.get("roi_disk_detect_max_side", 1024)),
        },
    }


def roi_circle_from_apply(result: RoiApplyResult) -> tuple[int, int, int] | None:
    """从 ROI 预处理结果提取圆盘 (cx, cy, radius)；未应用时返回 None。"""
    if not result.applied or result.center is None or result.radius is None:
        return None
    return (int(result.center[0]), int(result.center[1]), int(result.radius))


def clip_tile_intersects_roi(
    clip_x1: int,
    clip_y1: int,
    clip_x2: int,
    clip_y2: int,
    roi_circle: tuple[int, int, int] | None,
) -> bool:
    """切片矩形是否与 ROI 圆相交；``roi_circle`` 为 None 时视为全图有效。"""
    if roi_circle is None:
        return True
    cx, cy, radius = roi_circle
    closest_x = max(clip_x1, min(cx, clip_x2))
    closest_y = max(clip_y1, min(cy, clip_y2))
    dx = cx - closest_x
    dy = cy - closest_y
    return (dx * dx + dy * dy) <= radius * radius


def iter_clip_tiles_for_image(
    w: int,
    h: int,
    clip_size: int,
    overlap_size: int,
    clip_start: int = 0,
    *,
    roi_circle: tuple[int, int, int] | None = None,
):
    """滑窗坐标生成器；``roi_circle`` 存在时仅产出与 ROI 相交的窗。"""
    from script.predict.model_detect import get_clip

    for clip_x1, clip_y1, clip_x2, clip_y2 in get_clip(
        w, h, clip_size, overlap_size, clip_start=clip_start
    ):
        if clip_tile_intersects_roi(clip_x1, clip_y1, clip_x2, clip_y2, roi_circle):
            yield clip_x1, clip_y1, clip_x2, clip_y2


def _mask_outside_circle(
    image_bgr: np.ndarray,
    center: tuple[int, int],
    radius: int,
    *,
    fill_bgr: tuple[int, int, int],
) -> np.ndarray:
    out = image_bgr.copy()
    h, w = out.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, thickness=-1)
    out[mask == 0] = fill_bgr
    return out


def _resize_gray_for_detect(gray: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    h, w = gray.shape[:2]
    if max_side <= 0 or max(h, w) <= max_side:
        return gray, 1.0
    scale = max_side / float(max(h, w))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    small = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
    return small, scale


def _scale_roi_result(result: RoiDetectResult, inv_scale: float) -> RoiDetectResult:
    if inv_scale == 1.0:
        return result
    cx, cy = result.center
    return RoiDetectResult(
        roi_type=result.roi_type,
        center=(int(round(cx * inv_scale)), int(round(cy * inv_scale))),
        radius=int(round(result.radius * inv_scale)),
        score=result.score,
    )


def _border_mean_gray(gray: np.ndarray, margin_ratio: float) -> float:
    h, w = gray.shape[:2]
    m = max(2, int(min(h, w) * margin_ratio))
    strips = (gray[:m, :], gray[-m:, :], gray[:, :m], gray[:, -m:])
    return float(np.mean([float(s.mean()) for s in strips]))


def _score_disk_candidate(
    gray: np.ndarray,
    cx: int,
    cy: int,
    radius: int,
    *,
    min_r: float,
    max_r: float,
    center_tol: float,
    img_cx: float,
    img_cy: float,
    min_contrast: float,
    circularity: float = 1.0,
) -> float | None:
    h, w = gray.shape[:2]
    if radius < min_r or radius > max_r:
        return None
    if abs(cx - img_cx) > center_tol or abs(cy - img_cy) > center_tol:
        return None

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius, 255, thickness=-1)
    inside_mean = float(cv2.mean(gray, mask=mask)[0])
    outside_mask = cv2.bitwise_not(mask)
    outside_mean = float(cv2.mean(gray, mask=outside_mask)[0])
    contrast = inside_mean - outside_mean
    if contrast < min_contrast:
        return None

    radius_score = 1.0 - abs(radius - (min_r + max_r) * 0.5) / max(1.0, max_r - min_r)
    center_score = 1.0 - (abs(cx - img_cx) + abs(cy - img_cy)) / max(1.0, 2.0 * center_tol)
    contrast_score = min(1.0, contrast / 120.0)
    return (
        0.30 * circularity
        + 0.25 * radius_score
        + 0.20 * center_score
        + 0.25 * contrast_score
    )


def _detect_disk_circle_hough(
    gray: np.ndarray,
    blurred: np.ndarray,
    opts: dict[str, Any],
    *,
    min_r: float,
    max_r: float,
    center_tol: float,
    img_cx: float,
    img_cy: float,
) -> RoiDetectResult | None:
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=float(opts.get("hough_dp", 1.2)),
        minDist=max(min_r, 1.0),
        param1=float(opts.get("hough_param1", 80.0)),
        param2=float(opts.get("hough_param2", 40.0)),
        minRadius=int(max(1, min_r)),
        maxRadius=int(max(min_r + 1, max_r)),
    )
    if circles is None:
        return None

    best: RoiDetectResult | None = None
    best_score = -1.0
    min_contrast = float(opts.get("min_contrast", 35.0))
    for cx_f, cy_f, r_f in circles[0]:
        cx, cy, radius = int(round(float(cx_f))), int(round(float(cy_f))), int(round(float(r_f)))
        score = _score_disk_candidate(
            gray,
            cx,
            cy,
            radius,
            min_r=min_r,
            max_r=max_r,
            center_tol=center_tol,
            img_cx=img_cx,
            img_cy=img_cy,
            min_contrast=min_contrast,
            circularity=1.0,
        )
        if score is None or score <= best_score:
            continue
        best_score = score
        best = RoiDetectResult(
            roi_type="disk_circle",
            center=(cx, cy),
            radius=radius,
            score=float(score),
        )
    return best


def _detect_disk_circle_contour(
    gray: np.ndarray,
    blurred: np.ndarray,
    opts: dict[str, Any],
    *,
    min_r: float,
    max_r: float,
    center_tol: float,
    img_cx: float,
    img_cy: float,
) -> RoiDetectResult | None:
    thr_cfg = opts.get("threshold")
    if thr_cfg is not None:
        _, binary = cv2.threshold(blurred, int(thr_cfg), 255, cv2.THRESH_BINARY)
    else:
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best: RoiDetectResult | None = None
    best_score = -1.0
    min_circularity = float(opts.get("min_circularity", 0.58))
    min_contrast = float(opts.get("min_contrast", 35.0))
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < np.pi * min_r * min_r * 0.5:
            continue
        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue
        circularity = 4.0 * np.pi * area / (peri * peri)
        if circularity < min_circularity:
            continue
        (cx_f, cy_f), radius_f = cv2.minEnclosingCircle(cnt)
        cx, cy, radius = int(round(cx_f)), int(round(cy_f)), int(round(radius_f))
        score = _score_disk_candidate(
            gray,
            cx,
            cy,
            radius,
            min_r=min_r,
            max_r=max_r,
            center_tol=center_tol,
            img_cx=img_cx,
            img_cy=img_cy,
            min_contrast=min_contrast,
            circularity=circularity,
        )
        if score is None or score <= best_score:
            continue
        best_score = score
        best = RoiDetectResult(
            roi_type="disk_circle",
            center=(cx, cy),
            radius=radius,
            score=float(score),
        )
    return best


def detect_disk_circle(
    image_bgr: np.ndarray,
    opts: dict[str, Any],
) -> RoiDetectResult | None:
    """
    识别「深色外框 + 中心亮圆盘」场景；正常全幅白底图应返回 None。
    """
    if image_bgr is None or image_bgr.size == 0:
        return None
    h, w = image_bgr.shape[:2]
    if h < 32 or w < 32:
        return None

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    border_mean = _border_mean_gray(
        gray, float(opts.get("border_margin_ratio", 0.05))
    )
    if border_mean > float(opts.get("max_border_mean", 120.0)):
        return None

    detect_max_side = int(opts.get("detect_max_side", 1024))
    detect_gray, scale = _resize_gray_for_detect(gray, detect_max_side)
    inv_scale = 1.0 / scale

    k = max(3, int(opts.get("blur_ksize", 9)) | 1)
    blurred = cv2.GaussianBlur(detect_gray, (k, k), 2)

    dh, dw = detect_gray.shape[:2]
    min_side = float(min(dh, dw))
    min_r = min_side * float(opts.get("min_radius_ratio", 0.28))
    max_r = min_side * float(opts.get("max_radius_ratio", 0.52))
    center_tol = min_side * float(opts.get("center_tol_ratio", 0.12))
    img_cx, img_cy = dw * 0.5, dh * 0.5
    min_score = float(opts.get("min_score", 0.50))

    # 对比度评分仍在原图尺度上校验
    full_min_side = float(min(h, w))
    full_min_r = full_min_side * float(opts.get("min_radius_ratio", 0.28))
    full_max_r = full_min_side * float(opts.get("max_radius_ratio", 0.52))
    full_center_tol = full_min_side * float(opts.get("center_tol_ratio", 0.12))
    full_img_cx, full_img_cy = w * 0.5, h * 0.5

    def _finalize_candidate(candidate: RoiDetectResult | None) -> RoiDetectResult | None:
        if candidate is None:
            return None
        mapped = _scale_roi_result(candidate, inv_scale)
        score = _score_disk_candidate(
            gray,
            mapped.center[0],
            mapped.center[1],
            mapped.radius,
            min_r=full_min_r,
            max_r=full_max_r,
            center_tol=full_center_tol,
            img_cx=full_img_cx,
            img_cy=full_img_cy,
            min_contrast=float(opts.get("min_contrast", 35.0)),
            circularity=1.0,
        )
        if score is None:
            return None
        return RoiDetectResult(
            roi_type=mapped.roi_type,
            center=mapped.center,
            radius=mapped.radius,
            score=float(score),
        )

    raw_best = _detect_disk_circle_hough(
        detect_gray,
        blurred,
        opts,
        min_r=min_r,
        max_r=max_r,
        center_tol=center_tol,
        img_cx=img_cx,
        img_cy=img_cy,
    )
    best = _finalize_candidate(raw_best)
    if best is None or best.score < min_score:
        raw_contour = _detect_disk_circle_contour(
            detect_gray,
            blurred,
            opts,
            min_r=min_r,
            max_r=max_r,
            center_tol=center_tol,
            img_cx=img_cx,
            img_cy=img_cy,
        )
        contour_best = _finalize_candidate(raw_contour)
        if contour_best is not None and (
            best is None or contour_best.score > best.score
        ):
            best = contour_best

    if best is None or best.score < min_score:
        return None
    return best


def _apply_disk_circle(
    image_bgr: np.ndarray,
    cfg: dict[str, Any],
) -> RoiDetectResult | None:
    opts = (cfg or {}).get("disk_circle") or {}
    return detect_disk_circle(image_bgr, opts)


register_roi_plugin("disk_circle", _apply_disk_circle)


def apply_roi_preprocess(
    image_bgr: np.ndarray,
    cfg: dict[str, Any] | None = None,
) -> RoiApplyResult:
    """
    按配置尝试 ROI 插件；未识别到 ROI 时原样返回（全图检测）。
    """
    if image_bgr is None or getattr(image_bgr, "size", 0) == 0:
        return RoiApplyResult(image=image_bgr, applied=False)

    opts = resolve_roi_options(cfg)
    if not opts["enabled"]:
        return RoiApplyResult(image=image_bgr, applied=False)

    for name in opts["plugins"]:
        plugin = _ROI_PLUGINS.get(name)
        if plugin is None:
            logging.warning("未知 ROI 插件 %r，已跳过", name)
            continue
        detected = plugin(image_bgr, opts)
        if detected is None:
            continue
        masked = _mask_outside_circle(
            image_bgr,
            detected.center,
            detected.radius,
            fill_bgr=tuple(opts["fill_bgr"]),
        )
        logging.debug(
            "ROI 已应用: type=%s center=%s radius=%d score=%.3f",
            detected.roi_type,
            detected.center,
            detected.radius,
            detected.score,
        )
        return RoiApplyResult(
            image=masked,
            applied=True,
            roi_type=detected.roi_type,
            center=detected.center,
            radius=detected.radius,
            score=detected.score,
        )

    logging.debug("ROI 开关已开但未识别到有效区域，保持全图")
    return RoiApplyResult(image=image_bgr, applied=False)


if __name__ == "__main__":
    # /Users/shunyaoyin/miniconda310/miniconda3/envs/yolo11/bin/python /Users/shunyaoyin/Documents/code/ai-company/insect/script/tools/roi_heyifei.py
    from pathlib import Path

    DISK_IMAGE = (
        "/Volumes/shunyao-h1/训练数据/测试集/友商/2fe61adf-b56b-438a-b422-6bbc319cd238.jpg"
    )
    NORMAL_IMAGE = (
        "/Volumes/shunyao-h1/训练数据/测试集/友商/220986868.jpg"
    )
    OUTPUT_DIR = "/tmp/roi_heyifei_preview"

    cfg = {"roi_switch": True}
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    for label, path in (("disk", DISK_IMAGE), ("normal", NORMAL_IMAGE)):
        img = cv2.imread(path)
        if img is None:
            print(f"skip unreadable: {path}")
            continue
        res = apply_roi_preprocess(img, cfg)
        print(
            f"{label}: applied={res.applied} type={res.roi_type} "
            f"center={res.center} radius={res.radius} score={res.score:.3f}"
        )
        cv2.imwrite(str(out_dir / f"{label}_roi.jpg"), res.image)
