#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Detail  : 分割推理业务工具（与 predict_size_validate_lib 解耦）：多边形绘制、VOC 写出、
#           图像遍历、简单几何评估。供 predict_seg_validate.py 使用。

from __future__ import annotations

import json
import logging
import math
import re
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from xml.dom import minidom

import cv2
import numpy as np

PIC_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

_FILE = Path(__file__).resolve()
_SCRIPT_DIR = _FILE.parent
# 与本模块同目录（script/）下的算法阈值配置
DEFAULT_INSECT_ALG_SEG_JSON = _SCRIPT_DIR / "insect_alg_seg.json"
INSECT_ALG_SEG_JSON_REL = "insect_alg_seg.json"


def resolve_insect_alg_seg_path(path: str | Path | None = None) -> Path:
    """
    解析 insect_alg_seg.json 路径。
    相对路径一律相对于 ``script/``（本文件所在目录），不依赖进程 cwd。
    """
    if path is None:
        return DEFAULT_INSECT_ALG_SEG_JSON
    p = Path(path)
    if not p.is_absolute():
        p = _SCRIPT_DIR / p
    return p


def load_insect_alg(path: str | Path | None = None) -> dict[str, Any] | None:
    p = resolve_insect_alg_seg_path(path)
    if not p.is_file():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


_ROUTE_PATTERNS_KEY = "__route_patterns__"

# out 路由键：尺寸区间，如 [50, 300) 表示 50≤dia<300（方括号/圆括号为开闭区间）
_OUT_DIA_INTERVAL_RE = re.compile(
    r"^([\[\(])\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*([\]\)])$"
)


def _route_entry_is_enabled(entry: Any) -> bool:
    """``out`` 子项 ``enable`` 未配置时默认为 True。"""
    if entry is None or not isinstance(entry, dict):
        return True
    return bool(entry.get("enable", True))


def parse_out_dia_interval_key(pattern: str) -> tuple[float, bool, float, bool] | None:
    """
    解析 ``out`` 尺寸区间路由键。

    返回 ``(lo, lo_inclusive, hi, hi_inclusive)``；非区间键返回 ``None``。
    示例：``[50, 300)`` → 50≤dia<300。
    """
    m = _OUT_DIA_INTERVAL_RE.match(str(pattern or "").strip())
    if not m:
        return None
    lo = float(m.group(2))
    hi = float(m.group(3))
    return lo, m.group(1) == "[", hi, m.group(4) == "]"


def dia_matches_out_interval(
    dia_px: float,
    interval: tuple[float, bool, float, bool],
) -> bool:
    lo, lo_inc, hi, hi_inc = interval
    if lo_inc:
        ok_lo = dia_px >= lo
    else:
        ok_lo = dia_px > lo
    if hi_inc:
        ok_hi = dia_px <= hi
    else:
        ok_hi = dia_px < hi
    return ok_lo and ok_hi


def is_out_dia_interval_route_key(pattern: str) -> bool:
    return parse_out_dia_interval_key(pattern) is not None


def match_out_route_pattern(pattern: str, class_name: str) -> bool:
    """
    ``insect_alg_all.json`` 的 ``out`` 键与模型输出类名匹配。

    - ``*``：匹配任意类名（含空串）
    - 尺寸区间键（如 ``[50, 300)``）不按类名匹配，见 ``resolve_out_route_entry``
    - 其它：按 Python ``re.fullmatch`` 正则匹配（如 ``yee|ming`` 同时匹配 yee、ming）
    - 非法正则时退化为与键名完全相等
    """
    pat = str(pattern or "").strip()
    cn = str(class_name or "").strip()
    if not pat:
        return False
    if is_out_dia_interval_route_key(pat):
        return False
    if pat == "*":
        return True
    try:
        return re.fullmatch(pat, cn) is not None
    except re.error:
        logging.warning("out 路由键非法正则 %r，已按字面量匹配", pat)
        return pat == cn


def resolve_out_route_entry(
    route_table: dict[str, Any] | None,
    class_name: str,
    *,
    dia_px: float | None = None,
    entry_enabled: Any = None,
) -> tuple[str | None, Any]:
    """
    按 ``out`` 对象键的**声明顺序**取首个匹配项（先声明者优先）。

    - 尺寸区间键（``[50, 300)`` 等）在提供 ``dia_px`` 时按对角线像素匹配；
      优先于 ``*``，与同序的类名键一并按声明顺序竞争。
    - ``enable=false`` 的分支跳过（未配置 ``enable`` 视为开启）。
    """
    if not route_table:
        return None, None
    cn = str(class_name or "").strip()
    is_enabled = entry_enabled if entry_enabled is not None else _route_entry_is_enabled

    def _try_match(pattern_key: str, entry: Any) -> tuple[str, Any] | None:
        if not is_enabled(entry):
            return None
        pat = str(pattern_key)
        interval = parse_out_dia_interval_key(pat)
        if interval is not None:
            if dia_px is None:
                return None
            if dia_matches_out_interval(float(dia_px), interval):
                return pat, entry
            return None
        if match_out_route_pattern(pat, cn):
            return pat, entry
        return None

    wildcard_entry: Any = None
    for pattern_key, entry in route_table.items():
        pat = str(pattern_key)
        if pat == "*":
            if wildcard_entry is None:
                wildcard_entry = entry
            continue
        hit = _try_match(pat, entry)
        if hit is not None:
            return hit[0], hit[1]

    if wildcard_entry is not None and is_enabled(wildcard_entry):
        if match_out_route_pattern("*", cn):
            return "*", wildcard_entry
    return None, None


def _alg_sub_from_out_entry(entry: dict[str, Any]) -> dict[str, Any]:
    sub: dict[str, Any] = {}
    if "detect_conf" in entry and entry["detect_conf"] is not None:
        sub["detect_conf"] = float(entry["detect_conf"])
    if "cls_conf" in entry and entry["cls_conf"] is not None:
        sub["cls_conf"] = float(entry["cls_conf"])
    if "dia" in entry and entry["dia"] is not None:
        sub["dia"] = entry["dia"]
    return sub


def build_alg_table_from_out(
    out: dict[str, Any] | None,
    *,
    entry_enabled: Any = None,
) -> dict[str, Any]:
    """
    将 ``out`` 子树中的 per-class detect_conf / cls_conf / dia 转为
    PredictSize / PredictSeg 使用的 insect_alg 表（含有序 ``__route_patterns__``）。
    """
    if not out:
        return {}
    is_enabled = entry_enabled
    if is_enabled is None:

        def is_enabled(entry: Any) -> bool:  # noqa: ANN401
            if entry is None or not isinstance(entry, dict):
                return True
            return bool(entry.get("enable", True))

    patterns: list[tuple[str, dict[str, Any]]] = []
    for cls_key, entry in out.items():
        if entry is None or not isinstance(entry, dict):
            continue
        if not is_enabled(entry):
            continue
        if is_out_dia_interval_route_key(str(cls_key)):
            continue
        sub = _alg_sub_from_out_entry(entry)
        if sub:
            patterns.append((str(cls_key), sub))
    if not patterns:
        return {}
    return {_ROUTE_PATTERNS_KEY: patterns}


def _iter_alg_sub_entries(alg: dict[str, Any] | None):
    """按配置顺序遍历 insect_alg 项（含 ``__route_patterns__``）。"""
    if not alg:
        return
    ordered = alg.get(_ROUTE_PATTERNS_KEY)
    if isinstance(ordered, list):
        for item in ordered:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                pat, entry = item[0], item[1]
                if isinstance(entry, dict):
                    yield str(pat), entry
        return
    for k, v in alg.items():
        if str(k).startswith("__") or not isinstance(v, dict):
            continue
        yield str(k), v


def lookup_insect_alg_by_pattern_key(alg: dict[str, Any] | None, pattern_key: str) -> dict[str, Any]:
    """按 ``out`` 配置键（如 ``yee|ming``）取门限子表，非模型类名。"""
    pk = str(pattern_key or "").strip()
    for pat, entry in _iter_alg_sub_entries(alg):
        if pat == pk:
            return entry
    return {}


def lookup_insect_alg_entry(alg: dict[str, Any] | None, class_name: str) -> dict[str, Any]:
    """按模型输出类名在 insect_alg 中取首个匹配的门限子表。"""
    cn = str(class_name or "").strip()
    for pat, entry in _iter_alg_sub_entries(alg):
        if match_out_route_pattern(pat, cn):
            return entry
    return {}


def resolve_seg_detect_conf(
    alg: dict[str, Any] | None,
    conf_thresh: float,
    *,
    insect_alg_profile: str | None = None,
    cls_list: list[str] | None = None,
) -> float:
    """从 insect_alg_seg.json 解析分割置信度门限（detect_conf）。"""
    if not alg:
        return float(conf_thresh)
    keys_try: list[str] = []
    if insect_alg_profile:
        keys_try.append(str(insect_alg_profile))
    for c in cls_list or []:
        cs = str(c).strip() if isinstance(c, str) else ""
        if cs and cs not in keys_try:
            keys_try.append(cs)
    if "insect" not in keys_try:
        keys_try.append("insect")
    for k in keys_try:
        entry = lookup_insect_alg_by_pattern_key(alg, k)
        if not entry:
            entry = lookup_insect_alg_entry(alg, k)
        if "detect_conf" in entry and entry["detect_conf"] is not None:
            return float(entry["detect_conf"])
    return float(conf_thresh)


def resolve_seg_dia_range(
    alg: dict[str, Any] | None,
    *,
    insect_alg_profile: str | None = None,
    cls_list: list[str] | None = None,
) -> tuple[float, float] | None:
    """
    从 insect_alg_seg.json 解析矩形对角线像素范围 ``dia: [min, max]``。
    解析顺序与 ``resolve_seg_detect_conf`` 一致（profile → cls_list → insect）。
    """
    if not alg:
        return None
    keys_try: list[str] = []
    if insect_alg_profile:
        keys_try.append(str(insect_alg_profile))
    for c in cls_list or []:
        cs = str(c).strip() if isinstance(c, str) else ""
        if cs and cs not in keys_try:
            keys_try.append(cs)
    if "insect" not in keys_try:
        keys_try.append("insect")
    for k in keys_try:
        entry = lookup_insect_alg_by_pattern_key(alg, k)
        if not entry:
            entry = lookup_insect_alg_entry(alg, k)
        dia = entry.get("dia")
        if isinstance(dia, (list, tuple)) and len(dia) >= 2:
            lo = float(dia[0])
            hi = float(dia[1])
            if lo > hi:
                lo, hi = hi, lo
            return lo, hi
    return None


def bbox_diag_px(x1: int, y1: int, x2: int, y2: int) -> float:
    """轴对齐外接矩形对角线长度（像素）。"""
    return float(math.hypot(max(0, int(x2) - int(x1)), max(0, int(y2) - int(y1))))


def bbox_diag_px_from_row(r: dict[str, Any]) -> float:
    x1, y1, x2, y2 = bbox_from_row(r)
    return bbox_diag_px(x1, y1, x2, y2)


def filter_rows_by_bbox_diag_range(
    rows: list[dict[str, Any]],
    dia_min: float,
    dia_max: float,
    *,
    class_keys: set[str] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """
    按外接矩形对角线过滤误报：仅保留 ``dia_min <= diag <= dia_max`` 的实例。
    ``class_keys`` 非空时仅对 ``cls_name`` / ``class_name`` 命中集合的实例做过滤，其余保留。
    返回 ``(保留列表, 剔除数量)``。
    """
    kept: list[dict[str, Any]] = []
    dropped = 0
    lo = float(dia_min)
    hi = float(dia_max)
    for r in rows:
        cls = str(r.get("cls_name") or r.get("class_name") or "").strip()
        if class_keys is not None and cls not in class_keys:
            kept.append(r)
            continue
        d = bbox_diag_px_from_row(r)
        if lo <= d <= hi:
            kept.append(r)
        else:
            dropped += 1
    return kept, dropped


def resolve_cls_top1_threshold(
    alg: dict[str, Any] | None,
    predicted_cls_name: str,
    cls_top1_conf_threshold: float | None,
) -> float | None:
    """分类 top1 门限：优先 JSON 中该类 cls_conf，否则用全局 cls_top1_conf_threshold。"""
    entry = lookup_insect_alg_entry(alg, predicted_cls_name)
    if entry and "cls_conf" in entry and entry["cls_conf"] is not None:
        return float(entry["cls_conf"])
    if cls_top1_conf_threshold is not None:
        return float(cls_top1_conf_threshold)
    return None


def resolve_mask_rate_range(entry: dict[str, Any] | None) -> tuple[float, float] | None:
    """
    从 ``out.<类名>`` 路由项解析 ``mask_rate: [lo, hi]``（闭区间）。
    未配置或格式无效时返回 None。
    """
    if not entry or not isinstance(entry, dict):
        return None
    mr = entry.get("mask_rate")
    if mr is None:
        return None
    if not isinstance(mr, (list, tuple)) or len(mr) < 2:
        return None
    try:
        lo = float(mr[0])
        hi = float(mr[1])
    except (TypeError, ValueError):
        return None
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def ensure_row_has_bbox_fields(row: dict[str, Any]) -> dict[str, Any]:
    """确保 row 含 ``x1..y2``；缺失时从 ``location`` / ``polygon`` 外接框补全。"""
    if all(k in row for k in ("x1", "y1", "x2", "y2")):
        return row
    bbox = row_location_to_bbox(row)
    if bbox is None:
        return row
    out = dict(row)
    out["x1"], out["y1"], out["x2"], out["y2"] = bbox
    return out


def compute_mask_bbox_fill_ratio_from_row(row: dict[str, Any]) -> float | None:
    """
    分割 mask 填充率：多边形栅格面积 / 实例外接框 ``x1,y1,x2,y2`` 面积。

    与训练侧 ``11-stat_mask_bbox_fill_ratio`` 的 alpha 口径不同；推理仅有 polygon 时用本函数。
    无有效 polygon（点数 < 3）或 bbox 为空时返回 None。
    """
    poly = list(row.get("polygon") or [])
    if len(poly) < 3:
        return None
    bbox = row_location_to_bbox(row)
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0 or bh <= 0:
        return None
    local = np.asarray(
        [[int(p[0]) - x1, int(p[1]) - y1] for p in poly],
        dtype=np.int32,
    )
    mask = np.zeros((bh, bw), dtype=np.uint8)
    cv2.fillPoly(mask, [local], 255)
    mask_area = int(np.count_nonzero(mask))
    bbox_area = bw * bh
    if bbox_area <= 0:
        return None
    return mask_area / float(bbox_area)


def mask_rate_passes(row: dict[str, Any], entry: dict[str, Any] | None) -> bool:
    """
    是否通过 ``mask_rate`` 门限。未配置 ``mask_rate`` 时恒为 True；
    已配置但无法计算填充率时不做过滤（保持原行为）。
    """
    rng = resolve_mask_rate_range(entry)
    if rng is None:
        return True
    ratio = compute_mask_bbox_fill_ratio_from_row(row)
    if ratio is None:
        return True
    lo, hi = rng
    return lo <= ratio <= hi


def row_location_to_bbox(row: dict[str, Any]) -> tuple[int, int, int, int] | None:
    """统一结果 ``location`` / ``x1..y2`` / ``polygon`` 外接框。"""
    loc = row.get("location")
    if isinstance(loc, (list, tuple)) and len(loc) >= 4:
        return int(loc[0]), int(loc[1]), int(loc[2]), int(loc[3])
    if all(k in row for k in ("x1", "y1", "x2", "y2")):
        try:
            return (
                int(row["x1"]),
                int(row["y1"]),
                int(row["x2"]),
                int(row["y2"]),
            )
        except (TypeError, ValueError):
            return None
    poly = row.get("polygon")
    if isinstance(poly, list) and len(poly) >= 3:
        return bbox_from_row(row)
    return None


def _clamp_bbox_to_image(
    bbox: tuple[int, int, int, int],
    img_w: int,
    img_h: int,
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = bbox
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(int(img_w), int(x2))
    y2 = min(int(img_h), int(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _binarize_crop_black_mask(image_bgr: np.ndarray) -> np.ndarray | None:
    """检测框裁切 → 灰度+CLAHE+Otsu 二值化，返回黑色像素布尔掩码（局部坐标）。"""
    from script.predict.model_cls import ModelCls

    if image_bgr is None or image_bgr.size == 0:
        return None
    binary_bgr = ModelCls.bgr_gray_clahe_otsu_to_bgr(image_bgr)
    if binary_bgr is None or binary_bgr.size == 0:
        return None
    return binary_bgr[:, :, 0] == 0


def _odd_kernel_size(length: int, *, min_size: int = 3, max_size: int = 31) -> int:
    size = max(min_size, int(length))
    if size % 2 == 0:
        size += 1
    return min(max_size, size)


def _component_border_touch_sides(comp: np.ndarray) -> int:
    if comp is None or not np.any(comp):
        return 0
    h, w = comp.shape[:2]
    sides = 0
    if comp[0, :].any():
        sides += 1
    if comp[-1, :].any():
        sides += 1
    if comp[:, 0].any():
        sides += 1
    if comp[:, -1].any():
        sides += 1
    return sides


def _primary_insect_component_mask(mask_bool: np.ndarray) -> np.ndarray:
    """
    取最大连通域；若存在贴边 ≤2 侧的较大连通域则优先（虫体常靠 bbox 一角）。
    """
    if mask_bool is None or mask_bool.size == 0:
        return mask_bool
    fg_u8 = mask_bool.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg_u8, connectivity=8)
    if n <= 1:
        return mask_bool
    best_insect: np.ndarray | None = None
    best_insect_area = 0
    best_any_area = 0
    best_any_label = 1
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area > best_any_area:
            best_any_area = area
            best_any_label = i
        comp = labels == i
        if _component_border_touch_sides(comp) <= 2 and area > best_insect_area:
            best_insect_area = area
            best_insect = comp
    if best_insect is not None and best_insect_area >= best_any_area * 0.18:
        return best_insect
    return labels == best_any_label


def _primary_foreground_component_mask(mask_bool: np.ndarray) -> np.ndarray:
    """保留面积最大的前景连通域。"""
    return _primary_insect_component_mask(mask_bool)


def _largest_foreground_component_mask(
    mask_bool: np.ndarray,
    *,
    center_xy: tuple[int, int],
) -> np.ndarray:
    """保留包含中心点的连通域，否则取最大连通域。"""
    if mask_bool is None or mask_bool.size == 0:
        return mask_bool
    fg_u8 = mask_bool.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg_u8, connectivity=8)
    if n <= 1:
        return mask_bool
    cx, cy = center_xy
    h, w = mask_bool.shape[:2]
    if 0 <= cy < h and 0 <= cx < w:
        center_label = int(labels[cy, cx])
        if center_label > 0:
            return labels == center_label
    return _primary_foreground_component_mask(mask_bool)


def _in_big_fg_mask_area_ratio(mask_bool: np.ndarray) -> float:
    if mask_bool is None or not np.any(mask_bool):
        return 0.0
    h, w = mask_bool.shape[:2]
    if h <= 0 or w <= 0:
        return 0.0
    return float(np.count_nonzero(mask_bool)) / float(h * w)


def _crop_border_background_mask(
    image_bgr: np.ndarray,
    *,
    bg_percentile: float = 8.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    边框 Lab 采样托盘背景，返回 ``(color_dist, is_tray_strict, bg_ref_dist)``。

    ``bg_ref_dist`` 为边框颜色距离中值，用于虫体/托盘相对阈值。
    """
    h, w = image_bgr.shape[:2]
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    strip = max(2, min(h, w) // 18)
    border = np.zeros((h, w), dtype=bool)
    border[:strip, :] = True
    border[h - strip :, :] = True
    border[:, :strip] = True
    border[:, w - strip :] = True
    bg = np.median(lab[border], axis=0)
    dist = np.linalg.norm(lab - bg.reshape(1, 1, 3), axis=2)
    border_dist = dist[border]
    bg_ref = float(np.median(border_dist)) if border_dist.size else 0.0
    inner_dist = dist[~border]
    if inner_dist.size == 0:
        return dist, np.ones((h, w), dtype=bool), bg_ref
    p_tray = float(np.percentile(inner_dist, 12.0))
    if bg_ref > 1e-6:
        p_tray = min(p_tray, bg_ref * 1.05)
    return dist, dist <= p_tray, bg_ref


def _crop_insect_hint_mask(
    dist: np.ndarray,
    dark: np.ndarray,
    is_tray: np.ndarray,
    *,
    bg_ref: float,
    border: np.ndarray | None = None,
) -> np.ndarray:
    """虫体种子：深色 Otsu 或明显高于托盘/浅背景的有色区域（控制种子面积）。"""
    if border is None:
        border = is_tray
    inner = dist[~border] if np.any(~border) else dist.ravel()
    if inner.size == 0:
        return dark & ~is_tray
    p_color = float(np.percentile(inner, 40.0))
    if bg_ref > 1e-6:
        p_color = max(p_color, bg_ref * 1.16)
    colored = dist >= p_color
    return (dark | colored) & ~is_tray


def _hint_centroid_y(mask_bool: np.ndarray, h: int) -> float:
    if mask_bool is None or not np.any(mask_bool) or h <= 0:
        return float(h) * 0.5
    ys = np.where(mask_bool)[0]
    if ys.size == 0:
        return float(h) * 0.5
    return float(ys.mean())


def _lower_insect_seed_from_dark(dark_only: np.ndarray, *, h: int, w: int) -> np.ndarray:
    """取贴底或偏下深色连通域，用于纠正靠上误判的对角阴影种子。"""
    if dark_only is None or not np.any(dark_only):
        return dark_only
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        dark_only.astype(np.uint8), connectivity=8
    )
    if n <= 1:
        return dark_only
    lower = np.zeros((h, w), dtype=bool)
    for i in range(1, n):
        comp = labels == i
        if int(stats[i, cv2.CC_STAT_AREA]) < 24:
            continue
        ys = np.where(comp)[0]
        if comp[-1, :].any():
            lower |= comp
        elif not comp[0, :].any() and ys.size > 0 and float(ys.mean()) > h * 0.58:
            lower |= comp
    return lower


def _build_lower_insect_hint(
    lower_seed: np.ndarray,
    dist: np.ndarray,
    dark: np.ndarray,
    is_tray: np.ndarray,
    *,
    bg_ref: float,
    border: np.ndarray,
    h: int,
    w: int,
) -> np.ndarray:
    if not np.any(lower_seed):
        return lower_seed
    colored = _crop_insect_hint_mask(
        dist, dark, is_tray, bg_ref=bg_ref, border=border
    )
    env = _seed_bbox_envelope(lower_seed, h=h, w=w, pad_ratio=0.30)
    return (lower_seed | (colored & env)) & ~is_tray


def _enrich_insect_hint_with_color(
    insect_hint: np.ndarray,
    dist: np.ndarray,
    dark: np.ndarray,
    is_tray: np.ndarray,
    *,
    bg_ref: float,
    border: np.ndarray,
    h: int,
    w: int,
) -> np.ndarray:
    """翅膜等浅色虫体：在种子包络内并入有色区域，补全脉间浅区。"""
    if not np.any(insect_hint):
        return insect_hint
    colored = _crop_insect_hint_mask(
        dist, dark, is_tray, bg_ref=bg_ref, border=border
    )
    env = _seed_bbox_envelope(insect_hint, h=h, w=w, pad_ratio=0.28)
    enrich = colored & env & ~is_tray
    merged = (insect_hint | enrich) & ~is_tray
    if float(np.count_nonzero(merged)) / float(h * w) > 0.42:
        return insect_hint
    return merged


def _clip_fg_by_color_support(
    fg: np.ndarray,
    dist: np.ndarray,
    is_tray: np.ndarray,
    *,
    bg_ref: float,
    border: np.ndarray,
) -> np.ndarray:
    """仅剔除伸入托盘空白的前景泄漏，保留浅色翅膜。"""
    if fg is None or not np.any(fg):
        return fg
    inner = dist[~border] if np.any(~border) else dist.ravel()
    if inner.size == 0:
        return fg
    p_color = float(np.percentile(inner, 28.0))
    if bg_ref > 1e-6:
        p_color = max(p_color, bg_ref * 1.07)
    leak = fg & is_tray & (dist < p_color)
    fg_pixels = int(np.count_nonzero(fg))
    if fg_pixels <= 0:
        return fg
    if float(np.count_nonzero(leak)) / float(fg_pixels) < 0.05:
        return fg
    clipped = fg & ~leak
    if not np.any(clipped):
        return fg
    return _primary_insect_component_mask(clipped)


def _otsu_refined_crop_foreground_candidates(
    crop_bgr: np.ndarray,
    is_tray: np.ndarray,
) -> list[np.ndarray]:
    """Otsu 主连通域/全深色前景 + 孔洞填充，用于补全偏小的种子生长结果。"""
    otsu = _binarize_crop_black_mask(crop_bgr)
    if otsu is None:
        return []
    scoped = otsu & ~is_tray
    if not np.any(scoped):
        return []
    candidates: list[np.ndarray] = []
    seen_pixels: set[int] = set()
    for base in (_primary_insect_component_mask(scoped), scoped):
        if not np.any(base):
            continue
        refined = _refine_in_big_fg_mask(base, envelope=None)
        refined = _fill_interior_holes_only(refined)
        if not np.any(refined):
            continue
        px = int(np.count_nonzero(refined))
        if px in seen_pixels:
            continue
        seen_pixels.add(px)
        candidates.append(refined)
    return candidates


def _recover_undersized_in_big_fg(
    crop_bgr: np.ndarray,
    fg: np.ndarray | None,
) -> np.ndarray | None:
    """种子生长 mask 明显小于 Otsu 精炼结果时，回退到更大且托盘泄漏可控的候选。"""
    if crop_bgr is None or crop_bgr.size == 0:
        return fg
    _, is_tray, _ = _crop_border_background_mask(crop_bgr, bg_percentile=8.0)
    h, w = crop_bgr.shape[:2]
    fg_pixels = int(np.count_nonzero(fg)) if fg is not None and np.any(fg) else 0
    best = fg
    best_pixels = fg_pixels
    min_gain = max(48, int(fg_pixels * 0.08)) if fg_pixels > 0 else 24
    for alt in _otsu_refined_crop_foreground_candidates(crop_bgr, is_tray):
        alt_pixels = int(np.count_nonzero(alt))
        if alt_pixels <= best_pixels + min_gain:
            continue
        alt_ratio = alt_pixels / float(h * w)
        if alt_ratio > 0.62:
            continue
        tray_leak = float(np.count_nonzero(alt & is_tray)) / float(alt_pixels)
        if tray_leak > 0.07:
            continue
        best = alt
        best_pixels = alt_pixels
    return best


def _crop_otsu_dark_mask(image_bgr: np.ndarray) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blur_k = _odd_kernel_size(min(h, w) // 12)
    blurred = cv2.GaussianBlur(enhanced, (blur_k, blur_k), 0)
    _, bin_inv = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return bin_inv > 0


def _crop_closed_edge_mask(image_bgr: np.ndarray) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blur_k = _odd_kernel_size(min(h, w) // 14)
    blurred = cv2.GaussianBlur(enhanced, (blur_k, blur_k), 0)
    med = float(np.median(blurred))
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * med))
    upper = int(min(255, (1.0 + sigma) * med))
    if upper <= lower:
        upper = min(255, lower + 1)
    edges = cv2.Canny(blurred, lower, upper)
    close_k = _odd_kernel_size(min(h, w) // 20, min_size=3, max_size=15)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    return closed > 0


def _mask_border_foreground_ratio(mask_bool: np.ndarray) -> float:
    if mask_bool is None or not np.any(mask_bool):
        return 0.0
    h, w = mask_bool.shape[:2]
    border = np.concatenate(
        [mask_bool[0, :], mask_bool[-1, :], mask_bool[:, 0], mask_bool[:, -1]]
    )
    return float(np.count_nonzero(border)) / float(max(1, border.size))


def _seed_bbox_envelope(
    insect_hint: np.ndarray,
    *,
    h: int,
    w: int,
    pad_ratio: float = 0.18,
) -> np.ndarray:
    """以虫体种子外接框（加边距）限制生长范围，避免铺满托盘空白。"""
    pts = np.column_stack(np.where(insect_hint))
    if pts.size == 0:
        return np.zeros((h, w), dtype=bool)
    ys = pts[:, 0]
    xs = pts[:, 1]
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    bw, bh = x2 - x1 + 1, y2 - y1 + 1
    px = max(2, int(bw * pad_ratio))
    py = max(2, int(bh * pad_ratio))
    env = np.zeros((h, w), dtype=bool)
    env[
        max(0, y1 - py) : min(h, y2 + py + 1),
        max(0, x1 - px) : min(w, x2 + px + 1),
    ] = True
    return env


def _fill_holes_within_limits(
    mask_bool: np.ndarray,
    *,
    envelope: np.ndarray | None = None,
    max_area_ratio: float = 0.66,
) -> np.ndarray:
    """孔洞填充；有包络时在包络内优先填实（翅脉间浅膜）。"""
    if mask_bool is None or not np.any(mask_bool):
        return mask_bool
    h, w = mask_bool.shape[:2]
    before = int(np.count_nonzero(mask_bool))
    scoped = mask_bool & envelope if envelope is not None else mask_bool
    if not np.any(scoped):
        scoped = mask_bool
    if envelope is not None and np.any(envelope):
        env_ratio = float(np.count_nonzero(envelope)) / float(h * w)
        env_scoped = scoped & envelope
        if np.any(env_scoped):
            filled_env = _fill_binary_mask_holes(env_scoped) & envelope
            if env_ratio <= 0.58 and float(np.count_nonzero(filled_env)) >= float(
                np.count_nonzero(env_scoped)
            ):
                if float(np.count_nonzero(filled_env)) / float(h * w) <= max(
                    max_area_ratio, env_ratio * 1.08
                ):
                    return filled_env
    filled = _fill_binary_mask_holes(scoped)
    if envelope is not None:
        filled = filled & envelope
    after = int(np.count_nonzero(filled))
    if after <= 0:
        return scoped
    if after / float(h * w) > max_area_ratio:
        return scoped
    if before > 0 and after > before * 1.85 and after / float(h * w) > 0.45:
        return scoped
    return filled


def _fill_interior_holes_only(
    mask_bool: np.ndarray,
    *,
    envelope: np.ndarray | None = None,
    max_hole_area_ratio: float = 0.09,
) -> np.ndarray:
    """仅填充不与边界连通的小孔洞（翅脉间浅膜），避免全图孔洞填充。"""
    if mask_bool is None or not np.any(mask_bool):
        return mask_bool
    h, w = mask_bool.shape[:2]
    scoped = mask_bool & envelope if envelope is not None else mask_bool
    if not np.any(scoped):
        return mask_bool
    inv = (~scoped).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    filled = scoped.copy()
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 12:
            continue
        comp = labels == i
        if (
            comp[0, :].any()
            or comp[-1, :].any()
            or comp[:, 0].any()
            or comp[:, -1].any()
        ):
            continue
        if area / float(h * w) > max_hole_area_ratio:
            continue
        filled |= comp
    if envelope is not None:
        filled &= envelope
    return filled


def _grow_seed_foreground_mask(
    insect_hint: np.ndarray,
    *,
    h: int,
    w: int,
    close_divisor: int,
    close_iters: int = 2,
    max_size: int = 21,
    envelope: np.ndarray | None = None,
) -> np.ndarray:
    """仅对虫体种子闭运算连通并孔洞填充，不依赖全图边缘轮廓。"""
    if not np.any(insect_hint):
        return insect_hint
    scoped_hint = insect_hint & envelope if envelope is not None else insect_hint
    if not np.any(scoped_hint):
        scoped_hint = insect_hint
    k = _odd_kernel_size(min(h, w) // close_divisor, min_size=3, max_size=max_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    seed_u8 = cv2.morphologyEx(
        scoped_hint.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=close_iters,
    )
    fg = _primary_foreground_component_mask(seed_u8 > 0)
    if envelope is not None:
        fg = fg & envelope
    return _fill_holes_within_limits(fg, envelope=envelope)


def _local_flood_from_hint(
    insect_hint: np.ndarray,
    dist: np.ndarray,
    is_tray: np.ndarray,
    *,
    bg_ref: float,
    h: int,
    w: int,
    envelope: np.ndarray | None = None,
) -> np.ndarray:
    """从虫体种子质心出发，仅在「非托盘」局部区域内泛洪，避免铺满整框。"""
    if not np.any(insect_hint):
        return insect_hint
    if bg_ref > 1e-6:
        local_valid = (dist > bg_ref * 1.03) & ~is_tray
    else:
        local_valid = ~is_tray
    valid_u8 = local_valid.astype(np.uint8) * 255
    k = _odd_kernel_size(min(h, w) // 28, min_size=3, max_size=7)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    valid_u8 = cv2.morphologyEx(valid_u8, cv2.MORPH_CLOSE, kernel, iterations=1)

    moments = cv2.moments(insect_hint.astype(np.uint8))
    if float(moments.get("m00", 0.0)) <= 0:
        return insect_hint
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    cx = min(max(cx, 0), w - 1)
    cy = min(max(cy, 0), h - 1)

    flooded = valid_u8.copy()
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    if flooded[cy, cx] == 0:
        pts = np.column_stack(np.where(insect_hint))
        if pts.size == 0:
            return insect_hint
        cy, cx = int(pts[0, 0]), int(pts[0, 1])
    cv2.floodFill(flooded, ff_mask, (cx, cy), 128)
    flood_region = flooded == 128
    band_k = _odd_kernel_size(min(h, w) // 18, min_size=3, max_size=11)
    band_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band_k, band_k))
    hint_band = cv2.dilate(
        insect_hint.astype(np.uint8) * 255, band_kernel, iterations=3
    )
    flood_fg = flood_region & (hint_band > 0)
    if envelope is not None:
        flood_fg = flood_fg & envelope
    if not np.any(flood_fg):
        flood_fg = _grow_seed_foreground_mask(
            insect_hint,
            h=h,
            w=w,
            close_divisor=16,
            close_iters=2,
            max_size=17,
            envelope=envelope,
        )
    return _fill_holes_within_limits(
        _primary_foreground_component_mask(flood_fg), envelope=envelope
    )


def _edge_constrained_grow_mask(
    image_bgr: np.ndarray,
    insect_hint: np.ndarray,
    is_tray: np.ndarray,
    *,
    envelope: np.ndarray | None = None,
) -> np.ndarray:
    """边缘仅作窄带约束：在虫体种子膨胀带内保留边缘，避免跨背景的大轮廓。"""
    h, w = image_bgr.shape[:2]
    if not np.any(insect_hint):
        return insect_hint
    edges = _crop_closed_edge_mask(image_bgr) & ~is_tray
    band_k = _odd_kernel_size(min(h, w) // 22, min_size=3, max_size=9)
    band_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band_k, band_k))
    hint_band = cv2.dilate(
        insect_hint.astype(np.uint8) * 255, band_kernel, iterations=2
    )
    edge_in_band = edges & (hint_band > 0)
    merged = insect_hint | edge_in_band
    if envelope is not None:
        merged = merged & envelope
    return _grow_seed_foreground_mask(
        merged, h=h, w=w, close_divisor=18, close_iters=2, max_size=17, envelope=envelope
    )


def _clip_fg_exclude_background(
    fg: np.ndarray,
    is_tray: np.ndarray,
) -> np.ndarray:
    if fg is None or not np.any(fg):
        return fg
    clipped = fg & ~is_tray
    if not np.any(clipped):
        return clipped
    return _primary_foreground_component_mask(clipped)


def _refine_in_big_fg_mask(
    mask_bool: np.ndarray,
    *,
    envelope: np.ndarray | None = None,
) -> np.ndarray:
    """闭运算连通 + 受限孔洞填充（不膨胀，避免铺满背景）。"""
    if mask_bool is None or not np.any(mask_bool):
        return mask_bool
    h, w = mask_bool.shape[:2]
    fg_u8 = mask_bool.astype(np.uint8) * 255
    k = _odd_kernel_size(min(h, w) // 26, min_size=3, max_size=9)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    fg_u8 = cv2.morphologyEx(fg_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
    filled = _fill_holes_within_limits(fg_u8 > 0, envelope=envelope)
    return _fill_interior_holes_only(filled, envelope=envelope)


def _score_edge_binary_fg_mask(
    mask_bool: np.ndarray,
    *,
    insect_hint: np.ndarray,
    is_tray: np.ndarray,
) -> float:
    """覆盖虫体种子、惩罚托盘泄漏、过大面积与贴边前景。"""
    if mask_bool is None or not np.any(mask_bool):
        return -1.0
    h, w = mask_bool.shape[:2]
    ratio = _in_big_fg_mask_area_ratio(mask_bool)
    if ratio < 0.03 or ratio > 0.68:
        return -1.0
    fg_pixels = int(np.count_nonzero(mask_bool))
    tray_leak = float(np.count_nonzero(mask_bool & is_tray)) / float(fg_pixels)
    if tray_leak > 0.06:
        return -1.0
    border_fg = _mask_border_foreground_ratio(mask_bool)
    tall_crop = h > w * 1.15
    border_limit = 0.30 if tall_crop else 0.24
    ratio_border_limit = 0.58 if tall_crop else 0.50
    if border_fg > border_limit and ratio > ratio_border_limit:
        return -1.0
    hint_pixels = int(np.count_nonzero(insect_hint))
    if hint_pixels > 0:
        hint_recall = float(np.count_nonzero(mask_bool & insect_hint)) / float(
            hint_pixels
        )
        if hint_recall < 0.35:
            return -1.0
    else:
        hint_recall = 0.4
    size_penalty = max(0.0, ratio - 0.55) * 1.2
    undersize_penalty = max(0.0, 0.30 - ratio) * 1.0
    return (
        hint_recall * 0.58
        + min(ratio, 0.55) * 0.24
        - tray_leak * 2.5
        - border_fg * 0.45
        - size_penalty
        - undersize_penalty
    )


def _edge_binary_crop_foreground_mask(image_bgr: np.ndarray) -> np.ndarray | None:
    """
    虫体种子生长 + 可选边缘窄带约束；禁止全图轮廓填充。

    托盘由边框色差界定，浅色翅膜在种子闭运算+孔洞填充后并入虫体。
    """
    h, w = image_bgr.shape[:2]
    if h < 8 or w < 8:
        return None

    dist, is_tray, bg_ref = _crop_border_background_mask(
        image_bgr, bg_percentile=8.0
    )
    h, w = image_bgr.shape[:2]
    strip = max(2, min(h, w) // 18)
    border = np.zeros((h, w), dtype=bool)
    border[:strip, :] = True
    border[h - strip :, :] = True
    border[:, :strip] = True
    border[:, w - strip :] = True
    dark = _crop_otsu_dark_mask(image_bgr)
    dark_only = dark & ~is_tray
    dark_ratio = float(np.count_nonzero(dark_only)) / float(h * w)
    comp_hint = _primary_insect_component_mask(dark_only) & ~is_tray
    comp_ratio = float(np.count_nonzero(comp_hint)) / float(h * w)
    dark_pixels = int(np.count_nonzero(dark_only))
    dark_cover = (
        float(np.count_nonzero(comp_hint & dark_only)) / float(dark_pixels)
        if dark_pixels > 0
        else 0.0
    )
    if dark_ratio >= 0.10 and dark_cover >= 0.55 and comp_ratio >= 0.20:
        insect_hint = comp_hint if np.any(comp_hint) else dark_only
    else:
        insect_hint = _crop_insect_hint_mask(
            dist, dark, is_tray, bg_ref=bg_ref, border=border
        )
        if np.any(comp_hint):
            insect_hint = insect_hint | comp_hint
        insect_hint = _primary_insect_component_mask(insect_hint & ~is_tray)
        if not np.any(insect_hint):
            insect_hint = _crop_insect_hint_mask(
                dist, dark, is_tray, bg_ref=bg_ref, border=border
            )
    insect_hint = _enrich_insect_hint_with_color(
        insect_hint,
        dist,
        dark,
        is_tray,
        bg_ref=bg_ref,
        border=border,
        h=h,
        w=w,
    )
    envelope = _seed_bbox_envelope(insect_hint, h=h, w=w, pad_ratio=0.22) & ~is_tray
    hint_jobs: list[tuple[np.ndarray, np.ndarray]] = [(insect_hint, envelope)]

    lower_seed = _lower_insect_seed_from_dark(dark_only, h=h, w=w)
    if np.any(lower_seed):
        primary_seed = comp_hint if np.any(comp_hint) else insect_hint
        primary_cy = _hint_centroid_y(primary_seed, h)
        lower_cy = _hint_centroid_y(lower_seed, h)
        lower_area = int(np.count_nonzero(lower_seed))
        if (
            primary_cy < h * 0.44
            and lower_cy > h * 0.55
            and lower_area >= 80
            and lower_area <= int(np.count_nonzero(primary_seed)) * 0.45
        ):
            lower_hint = _build_lower_insect_hint(
                lower_seed,
                dist,
                dark,
                is_tray,
                bg_ref=bg_ref,
                border=border,
                h=h,
                w=w,
            )
            lower_env = (
                _seed_bbox_envelope(lower_hint, h=h, w=w, pad_ratio=0.28) & ~is_tray
            )
            if np.any(lower_hint):
                hint_jobs.append((lower_hint, lower_env))

    colored_hint = _crop_insect_hint_mask(
        dist, dark, is_tray, bg_ref=bg_ref, border=border
    ) & ~is_tray
    colored_env = _seed_bbox_envelope(colored_hint, h=h, w=w, pad_ratio=0.18) & ~is_tray

    raw_jobs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for job_idx, (job_hint, job_env) in enumerate(hint_jobs):
        hint_ratio = float(np.count_nonzero(job_hint)) / float(h * w)
        close_profiles = (
            (20, 2, 15),
            (14, 3, 21),
            (12, 3, 23),
        )
        if hint_ratio > 0.42:
            close_profiles = ((26, 2, 11), (20, 2, 15), (16, 2, 17))
        elif h > w * 1.15:
            close_profiles = ((18, 3, 19), (14, 3, 21), (12, 3, 23))
        elif len(hint_jobs) > 1 and job_idx > 0:
            close_profiles = ((12, 3, 25), (14, 3, 21), (16, 3, 19))

        for close_div, close_iter, max_k in close_profiles:
            grown = _grow_seed_foreground_mask(
                job_hint,
                h=h,
                w=w,
                close_divisor=close_div,
                close_iters=close_iter,
                max_size=max_k,
                envelope=job_env,
            )
            if np.any(grown):
                raw_jobs.append((grown, job_hint, job_env))

        flooded = _local_flood_from_hint(
            job_hint, dist, is_tray, bg_ref=bg_ref, h=h, w=w, envelope=job_env
        )
        if np.any(flooded):
            raw_jobs.append((flooded, job_hint, job_env))

        edge_grown = _edge_constrained_grow_mask(
            image_bgr, job_hint, is_tray, envelope=job_env
        )
        if np.any(edge_grown):
            raw_jobs.append((edge_grown, job_hint, job_env))

    dark_seed = _grow_seed_foreground_mask(
        dark_only,
        h=h,
        w=w,
        close_divisor=16,
        close_iters=2,
        max_size=19,
        envelope=envelope,
    )
    if np.any(dark_seed):
        raw_jobs.append((dark_seed, insect_hint, envelope))

    colored_grow = _grow_seed_foreground_mask(
        colored_hint,
        h=h,
        w=w,
        close_divisor=16,
        close_iters=3,
        max_size=21,
        envelope=colored_env,
    )
    if np.any(colored_grow):
        raw_jobs.append((colored_grow, insect_hint, envelope))

    for otsu_fg in _otsu_refined_crop_foreground_candidates(image_bgr, is_tray):
        raw_jobs.append((otsu_fg, insect_hint, envelope))

    best_mask: np.ndarray | None = None
    best_score = -1.0
    for raw, job_hint, job_env in raw_jobs:
        fg = _clip_fg_exclude_background(raw, is_tray)
        fg = _refine_in_big_fg_mask(fg, envelope=job_env)
        fg = _clip_fg_exclude_background(fg, is_tray)
        fg = _clip_fg_by_color_support(
            fg, dist, is_tray, bg_ref=bg_ref, border=border
        )
        if not np.any(fg):
            continue
        score = _score_edge_binary_fg_mask(
            fg, insect_hint=job_hint, is_tray=is_tray
        )
        if score > best_score:
            best_score = score
            best_mask = fg

    return _recover_undersized_in_big_fg(image_bgr, best_mask)


def _fill_crop_foreground_mask(image_bgr: np.ndarray) -> np.ndarray | None:
    """
    bbox 裁切虫体前景：种子生长 + 边缘窄带约束，边框色差剔除托盘。

    纯 OpenCV；用于 ``in_big_fill`` 分子。
    """
    if image_bgr is None or image_bgr.size == 0:
        return None
    h, w = image_bgr.shape[:2]
    if h < 4 or w < 4:
        return None

    fg = _edge_binary_crop_foreground_mask(image_bgr)
    if fg is not None and np.any(fg):
        return fg

    dist, is_tray, bg_ref = _crop_border_background_mask(
        image_bgr, bg_percentile=8.0
    )
    h, w = image_bgr.shape[:2]
    strip = max(2, min(h, w) // 18)
    border = np.zeros((h, w), dtype=bool)
    border[:strip, :] = True
    border[h - strip :, :] = True
    border[:, :strip] = True
    border[:, w - strip :] = True
    dark = _crop_otsu_dark_mask(image_bgr)
    dark_only = dark & ~is_tray
    dark_ratio = float(np.count_nonzero(dark_only)) / float(h * w)
    comp_hint = _primary_insect_component_mask(dark_only) & ~is_tray
    comp_ratio = float(np.count_nonzero(comp_hint)) / float(h * w)
    dark_pixels = int(np.count_nonzero(dark_only))
    dark_cover = (
        float(np.count_nonzero(comp_hint & dark_only)) / float(dark_pixels)
        if dark_pixels > 0
        else 0.0
    )
    if dark_ratio >= 0.10 and dark_cover >= 0.55 and comp_ratio >= 0.20:
        insect_hint = comp_hint if np.any(comp_hint) else dark_only
    else:
        insect_hint = _crop_insect_hint_mask(
            dist, dark, is_tray, bg_ref=bg_ref, border=border
        )
        if np.any(comp_hint):
            insect_hint = insect_hint | comp_hint
        insect_hint = _primary_insect_component_mask(insect_hint & ~is_tray)
        if not np.any(insect_hint):
            insect_hint = _crop_insect_hint_mask(
                dist, dark, is_tray, bg_ref=bg_ref, border=border
            )
    insect_hint = _enrich_insect_hint_with_color(
        insect_hint,
        dist,
        dark,
        is_tray,
        bg_ref=bg_ref,
        border=border,
        h=h,
        w=w,
    )
    envelope = _seed_bbox_envelope(insect_hint, h=h, w=w, pad_ratio=0.22) & ~is_tray
    fallback = _grow_seed_foreground_mask(
        insect_hint, h=h, w=w, close_divisor=14, close_iters=2, max_size=19,
        envelope=envelope,
    )
    fallback = _clip_fg_exclude_background(fallback, is_tray)
    fallback = _refine_in_big_fg_mask(fallback, envelope=envelope)
    fallback = _clip_fg_by_color_support(
        fallback, dist, is_tray, bg_ref=bg_ref, border=border
    )
    if np.any(fallback):
        return fallback

    black = _binarize_crop_black_mask(image_bgr)
    if black is None:
        return None
    scoped = _primary_insect_component_mask(black & ~is_tray)
    return _refine_in_big_fg_mask(scoped, envelope=envelope)


def _edge_fill_crop_foreground_mask(image_bgr: np.ndarray) -> np.ndarray | None:
    """兼容旧名；实际走 ``_fill_crop_foreground_mask``。"""
    return _fill_crop_foreground_mask(image_bgr)


def _fill_binary_mask_holes(mask_bool: np.ndarray) -> np.ndarray:
    """
    对二值前景掩码做孔洞填充：内部不与边界连通的空白区域并入前景。

    ``mask_bool`` 中 True 为前景（黑色虫体像素）。
    """
    if mask_bool is None or mask_bool.size == 0:
        return mask_bool
    if not np.any(mask_bool):
        return mask_bool
    fg = (mask_bool.astype(np.uint8) * 255)
    h, w = fg.shape
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    flooded = fg.copy()
    cv2.floodFill(flooded, ff_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flooded)
    filled = cv2.bitwise_or(fg, holes)
    return filled > 0


def _extract_in_big_crop_foreground_mask(
    crop_bgr: np.ndarray,
    *,
    fill_holes: bool,
) -> np.ndarray | None:
    """``in_big_fill`` 为真时用 GrabCut/背景差分提取虫体并填充，否则 Otsu 二值化。"""
    if fill_holes:
        return _fill_crop_foreground_mask(crop_bgr)
    return _binarize_crop_black_mask(crop_bgr)


def _binarize_black_in_big_ratio_parts(
    image_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    polygon: list[list[int]],
    *,
    fill_holes: bool = False,
) -> tuple[
    float | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    """
    返回 ``(ratio, crop_bgr, black_local, big_mask_bool, union_bool)``。

    分子为 ``black_local ∩ big_mask`` 像素数；相交为 0 时 ratio 为 None（不判误报）。
    分母为大虫分割 mask 像素数（并集去除小虫独有区域，等价于 intersect/big）。
    """
    if image_bgr is None or image_bgr.size == 0:
        return None, None, None, None, None
    if len(polygon) < 3:
        return None, None, None, None, None
    img_h, img_w = image_bgr.shape[:2]
    clamped = _clamp_bbox_to_image(bbox, img_w, img_h)
    if clamped is None:
        return None, None, None, None, None
    x1, y1, x2, y2 = clamped
    crop = image_bgr[y1:y2, x1:x2].copy()
    black_local = _extract_in_big_crop_foreground_mask(crop, fill_holes=fill_holes)
    if black_local is None:
        return None, crop, None, None, None

    bh, bw = y2 - y1, x2 - x1
    poly_local = np.asarray(
        [[int(p[0]) - x1, int(p[1]) - y1] for p in polygon],
        dtype=np.int32,
    )
    poly_mask = np.zeros((bh, bw), dtype=np.uint8)
    cv2.fillPoly(poly_mask, [poly_local], 255)
    big_mask_bool = poly_mask > 0
    intersect_bool = black_local & big_mask_bool
    union_bool = black_local | big_mask_bool

    intersect_pixels = int(np.count_nonzero(intersect_bool))
    if intersect_pixels <= 0:
        return None, crop, black_local, big_mask_bool, union_bool
    big_pixels = int(np.count_nonzero(big_mask_bool))
    if big_pixels <= 0:
        return None, crop, black_local, big_mask_bool, union_bool
    return (
        intersect_pixels / float(big_pixels),
        crop,
        black_local,
        big_mask_bool,
        union_bool,
    )


def compute_binarize_black_in_big_ratio(
    image_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    polygon: list[list[int]],
    *,
    fill_holes: bool = False,
) -> float | None:
    """
    稻飞虱框 ``in_big_conf`` 占比：

    - 分子：检测框裁切前景 mask 与大虫 polygon mask 的**相交**像素数
    - 分母：大虫 polygon mask 像素数（并集去掉小虫独有区域）
    - 相交为 0 时不计算（返回 None，视为正确检出不过滤）
    """
    ratio, _, _, _, _ = _binarize_black_in_big_ratio_parts(
        image_bgr, bbox, polygon, fill_holes=fill_holes
    )
    return ratio


def max_binarize_black_in_big_ratio_parts(
    small_row: dict[str, Any],
    big_rows: list[dict[str, Any]],
    image_bgr: np.ndarray,
    *,
    fill_holes: bool = False,
) -> tuple[
    float | None,
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str] | None,
]:
    """
    取多张大虫 polygon 中占比最高的一组调试数据。

    返回 ``(ratio, (crop_bgr, black_local, big_mask_bool, union_bool, big_class_name))``。
    """
    bbox = row_location_to_bbox(small_row)
    if bbox is None:
        return None, None
    best_ratio: float | None = None
    best_parts: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str] | None = None
    fallback_parts: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str] | None = None
    for big_row in big_rows:
        poly = list(big_row.get("polygon") or [])
        ratio, crop, black_local, big_mask_bool, union_bool = _binarize_black_in_big_ratio_parts(
            image_bgr, bbox, poly, fill_holes=fill_holes
        )
        if crop is None or black_local is None or big_mask_bool is None or union_bool is None:
            continue
        big_name = str(
            big_row.get("name")
            or big_row.get("cls_name")
            or big_row.get("class_name")
            or ""
        ).strip()
        parts = (crop, black_local, big_mask_bool, union_bool, big_name)
        if fallback_parts is None:
            fallback_parts = parts
        if ratio is None:
            continue
        if best_ratio is None or ratio > best_ratio:
            best_ratio = ratio
            best_parts = parts
    if best_parts is None:
        best_parts = fallback_parts
    return best_ratio, best_parts


def build_in_big_debug_panel(
    crop_bgr: np.ndarray,
    black_local: np.ndarray,
    big_mask_bool: np.ndarray,
    *,
    denom_bool: np.ndarray | None = None,
    ratio: float | None = None,
    threshold: float | None = None,
    big_class_name: str = "",
    fill_holes_applied: bool = False,
) -> np.ndarray | None:
    """左：小虫前景 mask；右：大虫分割 mask；右下：黄=相交(分子)、红=仅小虫、浅绿=仅大虫。"""
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    if black_local is None or big_mask_bool is None:
        return None
    h, w = crop_bgr.shape[:2]
    if black_local.shape != (h, w) or big_mask_bool.shape != (h, w):
        return None
    union_bool = denom_bool if denom_bool is not None else (
        black_local | big_mask_bool
    )
    if union_bool.shape != (h, w):
        return None

    bin_vis = np.full((h, w, 3), 255, dtype=np.uint8)
    bin_vis[black_local] = (0, 0, 0)

    big_vis = np.full((h, w, 3), 255, dtype=np.uint8)
    big_vis[big_mask_bool] = (0, 200, 0)

    combo_vis = np.full((h, w, 3), 255, dtype=np.uint8)
    combo_vis[union_bool] = (200, 255, 200)
    combo_vis[black_local] = (0, 0, 255)
    combo_vis[np.logical_and(black_local, big_mask_bool)] = (0, 255, 255)

    top = np.hstack([bin_vis, big_vis])
    bottom = np.hstack([crop_bgr, combo_vis])
    panel = np.vstack([top, bottom])

    header_h = 28
    header = np.full((header_h, panel.shape[1], 3), 32, dtype=np.uint8)
    parts = ["L:edge_bin" if fill_holes_applied else "L:bin_black", "R:big_mask"]
    if big_class_name:
        parts.append(f"big={big_class_name}")
    if ratio is not None:
        parts.append(f"cap/big={float(ratio):.2f}")
    if threshold is not None:
        parts.append(f"thr={float(threshold):.2f}")
    text = " | ".join(parts)
    cv2.putText(
        header,
        text,
        (6, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([header, panel])


def save_in_big_debug_panel(
    panel: np.ndarray,
    output_dir: str | Path,
    *,
    image_stem: str,
    instance_tag: str,
    ratio: float | None = None,
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ratio_part = "na"
    if ratio is not None:
        ratio_part = f"{float(ratio):.2f}".replace(".", "p")
    safe_tag = re.sub(r"[^\w.\-]+", "_", str(instance_tag).strip()) or "inst"
    safe_stem = re.sub(r"[^\w.\-]+", "_", str(image_stem).strip()) or "image"
    out_path = out_dir / f"{safe_stem}_{safe_tag}_i{ratio_part}.jpg"
    cv2.imwrite(str(out_path), panel)
    return out_path


def collect_in_big_debug_parts_for_row(
    small_row: dict[str, Any],
    big_rows: list[dict[str, Any]],
    image_bgr: np.ndarray,
    *,
    fill_holes: bool = False,
) -> tuple[float | None, tuple[np.ndarray, np.ndarray, np.ndarray, str] | None]:
    """返回 in_big 占比与调试图部件；无大虫 mask 时仍返回二值化裁切（overlap 全 False）。"""
    ratio, parts = max_binarize_black_in_big_ratio_parts(
        small_row, big_rows, image_bgr, fill_holes=fill_holes
    )
    if parts is not None:
        return ratio, parts
    bbox = row_location_to_bbox(small_row)
    if bbox is None or image_bgr is None or getattr(image_bgr, "size", 0) == 0:
        return ratio, None
    img_h, img_w = image_bgr.shape[:2]
    clamped = _clamp_bbox_to_image(bbox, img_w, img_h)
    if clamped is None:
        return ratio, None
    x1, y1, x2, y2 = clamped
    crop = image_bgr[y1:y2, x1:x2].copy()
    black_local = _extract_in_big_crop_foreground_mask(crop, fill_holes=fill_holes)
    if black_local is None:
        return ratio, None
    overlap_bool = np.zeros(black_local.shape, dtype=bool)
    return ratio, (crop, black_local, overlap_bool, overlap_bool, "")


def max_binarize_black_in_big_ratio(
    small_row: dict[str, Any],
    big_rows: list[dict[str, Any]],
    image_bgr: np.ndarray,
    *,
    fill_holes: bool = False,
) -> float | None:
    """小虫行相对多张大虫 polygon 的最大 in_big 黑色占比。"""
    best_row, ratio = find_best_in_big_big_row(
        small_row, big_rows, image_bgr, fill_holes=fill_holes
    )
    del best_row
    return ratio


def find_best_in_big_big_row(
    small_row: dict[str, Any],
    big_rows: list[dict[str, Any]],
    image_bgr: np.ndarray,
    *,
    fill_holes: bool = False,
) -> tuple[dict[str, Any] | None, float | None]:
    """
    小虫行相对多张大虫 polygon 中 in_big 占比最高的大虫实例。

    返回 ``(big_row, ratio)``；无有效相交时 ``(None, None)``。
    """
    bbox = row_location_to_bbox(small_row)
    if bbox is None:
        return None, None
    best_ratio: float | None = None
    best_row: dict[str, Any] | None = None
    for big_row in big_rows:
        poly = list(big_row.get("polygon") or [])
        ratio = compute_binarize_black_in_big_ratio(
            image_bgr, bbox, poly, fill_holes=fill_holes
        )
        if ratio is None:
            continue
        if best_ratio is None or ratio > best_ratio:
            best_ratio = ratio
            best_row = big_row
    return best_row, best_ratio


def should_classify_seg_instance(seg_class_name: str, cls_list: list[str] | None) -> bool:
    """
    cls_list 为空/None 时对所有分割实例做分类；
    否则当分割类名命中 ``out`` 键（支持 ``*`` 与正则，如 ``yee|ming``）之一时分类。
    """
    if not cls_list:
        return True
    cn = str(seg_class_name or "").strip()
    for pat in cls_list:
        ps = str(pat).strip()
        if is_out_dia_interval_route_key(ps):
            continue
        if match_out_route_pattern(ps, cn):
            return True
    return False


_CLS_CROP_BG_WHITE_BGR: tuple[int, int, int] = (255, 255, 255)
_CLS_CROP_BG_BLACK_BGR: tuple[int, int, int] = (0, 0, 0)


def resolve_cls_crop_background(value: Any) -> tuple[int, int, int] | None:
    """
    解析分类 polygon 裁切时「掩码外区域」的背景填充。

    - 未配置 / 空 / ``none`` / ``false``：不补背景，外接框内保留原图像素。
    - ``white`` / ``black``：掩码外填白或黑（BGR）。
    - ``[B,G,R]`` / ``"b,g,r"``：按 BGR 填充。

    供 ``insect_alg_all.json`` 中 ``models.cls.cls_crop_background`` 使用。
    仅 ``from_bbox: false`` 且存在 ``polygon`` 时生效（按 mask 填底）。
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return _CLS_CROP_BG_WHITE_BGR if value else None
    if isinstance(value, str):
        s = value.strip().lower()
        if not s or s in ("none", "off", "false", "0", "no", "null"):
            return None
        if s == "white":
            return _CLS_CROP_BG_WHITE_BGR
        if s == "black":
            return _CLS_CROP_BG_BLACK_BGR
        parts = s.replace(",", " ").split()
        if len(parts) == 3:
            return (int(parts[0]), int(parts[1]), int(parts[2]))
        raise ValueError(
            f"cls_crop_background 字符串须为空、white、black 或 'b,g,r' 三通道，当前: {value!r}"
        )
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return (int(value[0]), int(value[1]), int(value[2]))
    raise ValueError(
        f"cls_crop_background 须为 None、字符串或长度为 3 的 BGR 列表，当前: {value!r}"
    )


def resolve_cls_pad_color(value: Any) -> tuple[int, int, int]:
    """
    解析 ``ModelCls`` 白边补方时的填充色（BGR）。

    - 未配置 / 空：默认 **白边** ``(255,255,255)``（与历史行为一致）。
    - ``white`` / ``black`` 或 ``[B,G,R]``：与 :func:`resolve_cls_crop_background` 相同，但不支持「不填」语义。

    供 ``insect_alg_all.json`` 中 ``models.cls.cls_pad_color`` 使用。
    """
    if value is None:
        return _CLS_CROP_BG_WHITE_BGR
    if isinstance(value, str) and not str(value).strip():
        return _CLS_CROP_BG_WHITE_BGR
    bg = resolve_cls_crop_background(value)
    if bg is None:
        return _CLS_CROP_BG_WHITE_BGR
    return bg


def crop_instance_bgr_from_polygon(
    image_bgr: np.ndarray,
    polygon: list[list[int]],
    *,
    pad_ratio: float = 0.05,
    pad_px: int = 0,
    background_bgr: tuple[int, int, int] | None = None,
) -> np.ndarray | None:
    """
    按多边形掩码裁剪实例：外接框（可外扩）内保留虫体。

    ``background_bgr`` 为 ``None`` 时不改写掩码外像素；否则将掩码外填为给定 BGR（如白底、黑底）。
    """
    if image_bgr is None or image_bgr.size == 0:
        return None
    poly = list(polygon or [])
    if len(poly) < 3:
        return None

    h_img, w_img = image_bgr.shape[:2]
    x1, y1, x2, y2 = bbox_from_row({"polygon": poly})
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    pad = int(max(pad_px, round(max(bw, bh) * float(pad_ratio))))
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w_img, x2 + pad)
    y2 = min(h_img, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None

    crop = image_bgr[y1:y2, x1:x2].copy()
    local = np.asarray(
        [[int(p[0]) - x1, int(p[1]) - y1] for p in poly],
        dtype=np.int32,
    )
    if background_bgr is not None:
        mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [local], 255)
        bg = np.array(background_bgr, dtype=crop.dtype)
        crop[mask == 0] = bg
    return crop


def bgr_to_gray_bgr3(image_bgr: np.ndarray) -> np.ndarray:
    """BGR → 灰度 → 三通道 BGR（R=G=B），供分类 crop 源图使用。"""
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr
    if image_bgr.ndim == 2:
        return cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        return image_bgr
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def expand_bbox_to_square_in_image(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    img_w: int,
    img_h: int,
) -> tuple[int, int, int, int]:
    """
    以检测框中心扩成正方形窗口，优先在原图范围内滑动；仅当图幅不足时才缩边长。

    用于 bbox 扩方时在原图内尽量取上下文，减少后续白边 padding。
    """
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    side = max(bw, bh)
    side = min(side, int(img_w), int(img_h))
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    nx1 = int(round(cx - side * 0.5))
    ny1 = int(round(cy - side * 0.5))
    nx2 = nx1 + side
    ny2 = ny1 + side
    if nx1 < 0:
        nx2 -= nx1
        nx1 = 0
    if ny1 < 0:
        ny2 -= ny1
        ny1 = 0
    if nx2 > img_w:
        shift = nx2 - img_w
        nx1 = max(0, nx1 - shift)
        nx2 = img_w
    if ny2 > img_h:
        shift = ny2 - img_h
        ny1 = max(0, ny1 - shift)
        ny2 = img_h
    nx1 = max(0, min(nx1, max(0, img_w - 1)))
    ny1 = max(0, min(ny1, max(0, img_h - 1)))
    nx2 = max(nx1 + 1, min(nx2, img_w))
    ny2 = max(ny1 + 1, min(ny2, img_h))
    return nx1, ny1, nx2, ny2


def crop_instance_bgr_from_bbox(
    image_bgr: np.ndarray,
    row: dict[str, Any],
) -> np.ndarray | None:
    """按实例外接框从大图矩形截取（不做 pad_square，由 ModelCls.predict 处理）。"""
    if image_bgr is None or image_bgr.size == 0:
        return None
    x1, y1, x2, y2 = bbox_from_row(row)
    h_img, w_img = image_bgr.shape[:2]
    xi1 = max(0, int(x1))
    yi1 = max(0, int(y1))
    xi2 = min(w_img, int(x2))
    yi2 = min(h_img, int(y2))
    if xi2 <= xi1 or yi2 <= yi1:
        return None
    return image_bgr[yi1:yi2, xi1:xi2].copy()


def iter_row_cls_topk(r: dict) -> list[tuple[str, float]]:
    """读取分类 top 列表；无列表时退回 cls_name / cls_conf。"""
    raw = r.get("cls_topk")
    if raw is None:
        raw = r.get("cls_top3")
    out: list[tuple[str, float]] = []
    if isinstance(raw, list):
        for it in raw:
            if not isinstance(it, dict):
                continue
            nm = str(it.get("class_name", "") or "").strip()
            if not nm:
                continue
            try:
                cf = float(it.get("conf", 0) or 0.0)
            except Exception:
                cf = 0.0
            out.append((nm, cf))
    if not out:
        nm = str(r.get("cls_name", "") or "").strip()
        if nm:
            try:
                cf = float(r.get("cls_conf", r.get("conf", 0)) or 0.0)
            except Exception:
                cf = 0.0
            out.append((nm, cf))
    return out


def collect_images(input_path: str) -> tuple[Path, list[Path]]:
    input_p = Path(input_path)
    if input_p.is_file():
        image_files = [input_p] if input_p.suffix.lower() in PIC_EXT else []
    elif input_p.is_dir():
        image_files = sorted(
            p for p in input_p.rglob("*") if p.is_file() and p.suffix.lower() in PIC_EXT
        )
    else:
        image_files = []
    return input_p, image_files


def polygon_centroid(poly: list[list[int]]) -> tuple[int, int]:
    if not poly:
        return 0, 0
    arr = np.asarray(poly, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 1:
        return 0, 0
    m = cv2.moments(arr.astype(np.float32))
    if abs(m.get("m00", 0.0)) > 1e-6:
        cx = int(round(m["m10"] / m["m00"]))
        cy = int(round(m["m01"] / m["m00"]))
        return cx, cy
    return int(round(float(arr[:, 0].mean()))), int(round(float(arr[:, 1].mean())))


def bbox_from_row(r: dict[str, Any]) -> tuple[int, int, int, int]:
    poly = r.get("polygon") or []
    if poly:
        xs = [int(p[0]) for p in poly]
        ys = [int(p[1]) for p in poly]
        return min(xs), min(ys), max(xs), max(ys)
    return (
        int(r.get("x1", 0)),
        int(r.get("y1", 0)),
        int(r.get("x2", 0)),
        int(r.get("y2", 0)),
    )


def polygon_raster_iou(poly_a: list[list[int]], poly_b: list[list[int]], w: int, h: int) -> float:
    """两多边形 mask IoU（在图像尺寸画布上栅格化）。"""
    if w <= 0 or h <= 0:
        return 0.0
    ma = np.zeros((h, w), dtype=np.uint8)
    mb = np.zeros((h, w), dtype=np.uint8)
    if len(poly_a) >= 3:
        cv2.fillPoly(ma, [np.asarray(poly_a, dtype=np.int32)], 1)
    if len(poly_b) >= 3:
        cv2.fillPoly(mb, [np.asarray(poly_b, dtype=np.int32)], 1)
    inter = int(np.logical_and(ma, mb).sum())
    if inter <= 0:
        return 0.0
    union = int(np.logical_or(ma, mb).sum())
    return float(inter) / float(union) if union > 0 else 0.0


def box_iou(box1, box2) -> float:
    ax1, ay1, ax2, ay2 = [int(v) for v in box1[:4]]
    bx1, by1, bx2, by2 = [int(v) for v in box2[:4]]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = a_area + b_area - inter
    return float(inter) / float(denom) if denom > 0 else 0.0


def parse_voc_objects(xml_path: str) -> list[dict[str, Any]]:
    """
    读取 VOC xml：优先 polygon（本脚本写出格式），否则回退 bndbox。
    返回 name, polygon, x1,y1,x2,y2。
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    out: list[dict[str, Any]] = []
    for obj in root.findall("object"):
        name_el = obj.find("name")
        if name_el is None or not name_el.text:
            continue
        name = name_el.text.strip()

        poly: list[list[int]] = []
        seg = obj.find("segmentation")
        if seg is not None:
            pts_el = seg.find("points")
            if pts_el is not None and pts_el.text:
                raw = pts_el.text.strip()
                nums = [float(x) for x in raw.replace(";", ",").split(",") if x.strip()]
                if len(nums) >= 6 and len(nums) % 2 == 0:
                    for i in range(0, len(nums), 2):
                        poly.append([int(round(nums[i])), int(round(nums[i + 1]))])

        if not poly:
            bnd = obj.find("bndbox")
            if bnd is None:
                continue

            def _int(tag: str) -> int:
                el = bnd.find(tag)
                if el is None or el.text is None:
                    raise ValueError(f"missing {tag}")
                return int(float(el.text.strip()))

            x1, y1, x2, y2 = _int("xmin"), _int("ymin"), _int("xmax"), _int("ymax")
            poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        else:
            x1, y1, x2, y2 = bbox_from_row({"polygon": poly})

        out.append(
            {
                "name": name,
                "polygon": poly,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )
    return out


def write_voc_seg_xml(
    xml_path: str,
    folder_name: str,
    image_filename: str,
    width: int,
    height: int,
    depth: int,
    results: list[dict[str, Any]],
) -> None:
    """写出带 segmentation/points 与 bndbox 的 VOC xml。"""
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = folder_name or ""
    ET.SubElement(annotation, "filename").text = image_filename
    src = ET.SubElement(annotation, "source")
    ET.SubElement(src, "database").text = "Unknown"
    size_el = ET.SubElement(annotation, "size")
    ET.SubElement(size_el, "width").text = str(int(width))
    ET.SubElement(size_el, "height").text = str(int(height))
    ET.SubElement(size_el, "depth").text = str(int(depth))
    ET.SubElement(annotation, "segmented").text = "1"

    for r in results:
        poly = list(r.get("polygon") or [])
        if not poly:
            x1, y1, x2, y2 = bbox_from_row(r)
            poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        else:
            x1, y1, x2, y2 = bbox_from_row({"polygon": poly})

        x1 = int(max(0, min(x1, width - 1)))
        y1 = int(max(0, min(y1, height - 1)))
        x2 = int(max(0, min(x2, width)))
        y2 = int(max(0, min(y2, height)))
        if x2 <= x1:
            x2 = min(width, x1 + 1)
        if y2 <= y1:
            y2 = min(height, y1 + 1)

        clipped_poly: list[list[int]] = []
        for px, py in poly:
            clipped_poly.append(
                [
                    int(max(0, min(int(px), width - 1))),
                    int(max(0, min(int(py), height - 1))),
                ]
            )
        if len(clipped_poly) < 3:
            clipped_poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = str(r.get("cls_name", r.get("class_name", "unknown")))
        seg_sc = r.get("seg_conf", r.get("conf"))
        if seg_sc is not None:
            ET.SubElement(obj, "seg_score").text = f"{float(seg_sc):.6f}"
        cls_sc = r.get("cls_conf")
        if cls_sc is not None:
            ET.SubElement(obj, "score").text = f"{float(cls_sc):.6f}"
        elif seg_sc is not None:
            ET.SubElement(obj, "score").text = f"{float(seg_sc):.6f}"
        seg_nm = r.get("seg_cls_name")
        if seg_nm:
            ET.SubElement(obj, "seg_class").text = str(seg_nm)
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"

        seg_el = ET.SubElement(obj, "segmentation")
        pts_str = ",".join(f"{int(p[0])},{int(p[1])}" for p in clipped_poly)
        ET.SubElement(seg_el, "points").text = pts_str

        bnd = ET.SubElement(obj, "bndbox")
        ET.SubElement(bnd, "xmin").text = str(x1)
        ET.SubElement(bnd, "ymin").text = str(y1)
        ET.SubElement(bnd, "xmax").text = str(x2)
        ET.SubElement(bnd, "ymax").text = str(y2)

    rough = ET.tostring(annotation, encoding="utf-8")
    parsed = minidom.parseString(rough)
    pretty = parsed.toprettyxml(indent="\t", encoding="utf-8")
    os.makedirs(os.path.dirname(xml_path) or ".", exist_ok=True)
    with open(xml_path, "wb") as f:
        f.write(pretty)


def _auto_center_label_draw_params(img_w: int, img_h: int) -> dict[str, int | float]:
    """中心点与 top1 标签的自适应参数（第 2、3 行仍用固定小字样式）。"""
    base = float(max(int(img_w), int(img_h), 1))
    k = base / 1200.0
    k = max(0.75, min(2.5, k))

    top1_font_scale = float(max(0.9, min(3.0, 1.15 * k)))
    top1_text_thk = max(2, min(8, int(round(2.5 * k))))
    top1_pad_x = max(6, int(round(10 * k)))
    top1_pad_y = max(4, int(round(8 * k)))
    top1_gap = max(4, int(round(6 * k)))

    circle_r = max(3, min(14, int(round(5 * k))))

    return {
        "k": k,
        "top1_font_scale": top1_font_scale,
        "top1_text_thk": top1_text_thk,
        "top1_pad_x": top1_pad_x,
        "top1_pad_y": top1_pad_y,
        "top1_gap": top1_gap,
        "circle_r": circle_r,
    }


def _draw_seg_sub_label(img: np.ndarray, text: str, tx: int, ty: int) -> None:
    """第 2、3 行等小字标签（保持原固定样式）。"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    cv2.putText(
        img, text, (tx, ty), font, font_scale,
        (0, 0, 0), thickness=3, lineType=cv2.LINE_AA,
    )
    cv2.putText(
        img, text, (tx, ty), font, font_scale,
        (255, 255, 255), thickness=1, lineType=cv2.LINE_AA,
    )


def _draw_seg_top1_label(
    img: np.ndarray,
    text: str,
    *,
    cx: int,
    cy: int,
    w_img: int,
    h_img: int,
    params: dict[str, int | float],
) -> int:
    """
    绘制醒目的 top1 标签：黑底 + 亮黄实心字 + 黄框。
    返回下一行小字的 baseline y（OpenCV putText 的 org.y）。
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = float(params["top1_font_scale"])
    thickness = int(params["top1_text_thk"])
    pad_x = int(params["top1_pad_x"])
    pad_y = int(params["top1_pad_y"])
    gap = int(params["top1_gap"])
    circle_r = int(params["circle_r"])
    border_thk = max(1, int(round(float(params["k"]))))

    (tw, th), bl = cv2.getTextSize(text, font, font_scale, thickness)
    tx = int(max(0, min(cx - tw // 2, w_img - tw - 1)))
    ty = int(max(th + bl + pad_y, min(cy - circle_r - gap, h_img - 1)))

    bx1 = max(0, tx - pad_x)
    by1 = max(0, ty - th - bl - pad_y)
    bx2 = min(w_img - 1, tx + tw + pad_x)
    by2 = min(h_img - 1, ty + pad_y)
    cv2.rectangle(img, (bx1, by1), (bx2, by2), (0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
    cv2.rectangle(img, (bx1, by1), (bx2, by2), (0, 255, 255), border_thk, lineType=cv2.LINE_AA)
    cv2.putText(
        img, text, (tx, ty), font, font_scale,
        (0, 255, 255), thickness, lineType=cv2.LINE_AA,
    )
    return int(by2 + 2)


def draw_seg_output_image(
    image_bgr: np.ndarray,
    rows: list[dict[str, Any]],
    *,
    draw_polygons: bool = True,
    draw_bbox: bool = False,
    draw_center_point_and_label: bool = False,
    cls_output_top_n: int = 1,
    polygon_alpha: float = 0.25,
    line_thickness: int = 2,
) -> np.ndarray:
    """绘制多边形轮廓；可选外接框、中心点与类名标签。"""
    img = image_bgr.copy()
    h_img, w_img = img.shape[:2]
    label_params = (
        _auto_center_label_draw_params(w_img, h_img)
        if draw_center_point_and_label
        else None
    )
    overlay = img.copy()
    alpha = float(polygon_alpha)
    if alpha < 0:
        alpha = 0.0
    if alpha > 1:
        alpha = 1.0

    label_jobs: list[tuple[list[str], list[tuple[int, int]]]] = []

    for r in rows:
        poly = list(r.get("polygon") or [])
        if len(poly) < 3:
            continue
        arr = np.asarray(poly, dtype=np.int32)
        color_line = (0, 255, 0)
        color_fill = (0, 200, 0)
        if draw_polygons:
            cv2.fillPoly(overlay, [arr], color_fill)
            cv2.polylines(img, [arr], True, color_line, int(line_thickness), lineType=cv2.LINE_AA)

        if draw_bbox:
            x1, y1, x2, y2 = bbox_from_row(r)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 128, 0), 1, lineType=cv2.LINE_AA)

        if draw_center_point_and_label and label_params is not None:
            cx, cy = polygon_centroid(poly)
            if 0 <= cx < w_img and 0 <= cy < h_img:
                tn = max(1, int(cls_output_top_n))
                ents = iter_row_cls_topk(r)
                label_lines = [
                    f"{ents[i][0]} {ents[i][1]:.2f}" for i in range(min(tn, len(ents)))
                ]
                if not label_lines:
                    name = str(r.get("cls_name", r.get("class_name", "")) or "").strip()
                    try:
                        conf_f = float(r.get("cls_conf", r.get("conf", 0.0)) or 0.0)
                    except (TypeError, ValueError):
                        conf_f = 0.0
                    label_lines = [f"{name} {conf_f:.2f}".strip()] if name else []
                try:
                    seg_f = float(r.get("seg_conf", r.get("conf", 0.0)) or 0.0)
                except (TypeError, ValueError):
                    seg_f = 0.0
                if r.get("seg_cls_name") and label_lines:
                    label_lines.append(f"seg:{r.get('seg_cls_name')} {seg_f:.2f}")
                if label_lines:
                    label_jobs.append((label_lines, [(cx, cy)]))

    if draw_polygons and alpha > 0:
        cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)

    if draw_center_point_and_label and label_params is not None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        circle_r = int(label_params["circle_r"])
        sub_gap = 2
        for label_lines, centroids in label_jobs:
            cx, cy = centroids[0]
            cv2.circle(
                img,
                (cx, cy),
                circle_r,
                (0, 255, 255),
                thickness=-1,
                lineType=cv2.LINE_AA,
            )
            next_y = _draw_seg_top1_label(
                img,
                label_lines[0],
                cx=cx,
                cy=cy,
                w_img=w_img,
                h_img=h_img,
                params=label_params,
            )
            for sub_text in label_lines[1:]:
                font_scale = 0.45
                thickness = 1
                (tw, th), bl = cv2.getTextSize(sub_text, font, font_scale, thickness)
                ty = int(next_y + th + bl)
                if ty >= h_img:
                    break
                tx = int(max(0, min(cx - tw // 2, w_img - tw - 1)))
                _draw_seg_sub_label(img, sub_text, tx, ty)
                next_y = ty + sub_gap

    return img


def outputs_exist_for_skip(out_dir: str, rel_file: Path) -> bool:
    img_path = os.path.join(out_dir, rel_file.name)
    xml_path = os.path.join(out_dir, rel_file.stem + ".xml")
    return os.path.isfile(img_path) and os.path.isfile(xml_path)


def match_pred_gt_polygon(
    preds: list[dict[str, Any]],
    gts: list[dict[str, Any]],
    *,
    img_w: int,
    img_h: int,
    iou_threshold: float = 0.5,
    use_mask_iou: bool = True,
) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    """
    贪心一对一匹配：默认 mask IoU，无多边形时回退 bbox IoU。
    返回 (matches, matched_pred_indices, matched_gt_indices)。
    """
    thr = float(iou_threshold)
    if thr <= 0 or not preds or not gts:
        return [], set(), set()

    pairs: list[tuple[float, int, int]] = []
    for pi, p in enumerate(preds):
        pb = [p["x1"], p["y1"], p["x2"], p["y2"]]
        ppoly = p.get("polygon") or []
        for gj, g in enumerate(gts):
            gb = [g["x1"], g["y1"], g["x2"], g["y2"]]
            gpoly = g.get("polygon") or []
            if use_mask_iou and len(ppoly) >= 3 and len(gpoly) >= 3:
                sc = polygon_raster_iou(ppoly, gpoly, img_w, img_h)
            else:
                sc = box_iou(pb, gb)
            if sc >= thr:
                pairs.append((sc, pi, gj))
    pairs.sort(key=lambda x: (-x[0], x[1], x[2]))

    matched_p: set[int] = set()
    matched_g: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for sc, pi, gj in pairs:
        if pi in matched_p or gj in matched_g:
            continue
        matched_p.add(pi)
        matched_g.add(gj)
        matches.append((pi, gj, sc))
    return matches, matched_p, matched_g


def normalize_class_name(raw: str, merge: dict[str, list[str]] | None) -> str:
    if not raw or not merge:
        return str(raw or "").strip()
    alias: dict[str, str] = {}
    for key, vals in merge.items():
        alias[key] = key
        for v in vals or []:
            if str(v).strip() == "*":
                continue
            alias[str(v)] = key
    hit = alias.get(str(raw).strip())
    return hit if hit is not None else str(raw).strip()


def is_class_match(pred_cls: str, gt_name: str, merge: dict[str, list[str]] | None) -> bool:
    p = normalize_class_name(pred_cls, merge)
    g = normalize_class_name(gt_name, merge)
    if not p or not g:
        return False
    if p == g:
        return True
    if not merge:
        return False
    for key, vals in merge.items():
        group = {key, *vals}
        if "*" in group:
            continue
        if p in group and g in group:
            return True
    for key, vals in merge.items():
        if "*" in (vals or []) and (p == key or g == key):
            return True
    return False
