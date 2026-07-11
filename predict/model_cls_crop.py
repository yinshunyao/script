#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""嵌套分类 crop/pad：彩色源图、polygon/bbox 裁切与补方参数。"""

from __future__ import annotations

from typing import Any

import numpy as np

from script.predict_seg_lib import (
    bbox_from_row,
    crop_instance_bgr_from_bbox,
    crop_instance_bgr_from_polygon,
    resolve_cls_crop_background,
)


def resolve_from_bbox(cfg: dict[str, Any] | None, *, default: bool = True) -> bool:
    """嵌套 cls ``from_bbox``：false 时走 polygon 掩码 crop（需 row 含 polygon）。"""
    if not isinstance(cfg, dict):
        return default
    v = cfg.get("from_bbox")
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("false", "0", "no", "off"):
            return False
        if s in ("true", "1", "yes", "on"):
            return True
    return bool(v)


def get_cls_crop_source_image(image_bgr: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """分类 crop 源图：与 0705 一致使用彩色原图；灰度仅由 ``to_gray`` 在推理阶段控制。"""
    _ = cfg
    return image_bgr


def cls_infer_pad_square(cfg: dict[str, Any], crop: np.ndarray | None = None) -> bool:
    """送入 ``ModelCls.predict(_batch)`` 时是否做 ``pad_square``（由 ``to_square`` 控制）。"""
    _ = crop
    return bool(cfg.get("to_square", True))


def cls_infer_to_gray(cfg: dict[str, Any]) -> bool:
    """推理前是否再转灰（``to_gray``）。"""
    return bool(cfg.get("to_gray", False))


def crop_cls_instance_bgr(
    image_bgr: np.ndarray,
    row: dict[str, Any],
    cfg: dict[str, Any],
) -> np.ndarray | None:
    """
    嵌套分类 crop：彩色源图；``from_bbox=false`` 且有 polygon 时掩码抠图，否则外接框紧裁；
    ``to_square`` 补方交给 ``ModelCls.predict``。
    """
    source = get_cls_crop_source_image(image_bgr, cfg)
    pad_ratio = float(cfg.get("crop_pad_ratio", 0.05))
    use_polygon = not resolve_from_bbox(cfg)
    mask_bg = resolve_cls_crop_background(cfg.get("cls_crop_background"))

    if use_polygon and row.get("polygon"):
        return crop_instance_bgr_from_polygon(
            source,
            list(row["polygon"]),
            pad_ratio=pad_ratio,
            background_bgr=mask_bg,
        )

    return crop_instance_bgr_from_bbox(source, row)


def cls_crop_rect_for_row(
    row: dict[str, Any],
    cfg: dict[str, Any],
    *,
    img_w: int,
    img_h: int,
) -> tuple[int, int, int, int] | None:
    """GPU bbox batch crop 用的外接框矩形（不含 polygon 掩码）。"""
    _ = cfg
    x1, y1, x2, y2 = bbox_from_row(row)
    xi1 = max(0, int(x1))
    yi1 = max(0, int(y1))
    xi2 = min(img_w, int(x2))
    yi2 = min(img_h, int(y2))
    if xi2 <= xi1 or yi2 <= yi1:
        return None
    return xi1, yi1, xi2, yi2
