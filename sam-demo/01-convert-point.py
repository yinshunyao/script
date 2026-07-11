#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Detail  : 将目录内标注（LabelMe json / VOC xml）转为 SAM 点提示 JSON
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from sam_io import (
    POINT_MODES,
    assign_background_object_id,
    build_points_record,
    clip_point,
    is_labelme_polyline_shape,
    load_sam_points_config,
    parse_annotation_objects,
    polygon_to_prompt_points,
    save_points_json,
    scan_annotation_pairs,
    split_voc_objects_by_role,
)

logger = logging.getLogger(__name__)


def _effective_point_mode(obj: dict[str, Any], point_mode: str) -> str:
    """折线类 shape 固定按全部顶点取点（等同 vertices）。"""
    if is_labelme_polyline_shape(str(obj.get("shape_type") or "")):
        return "vertices"
    return point_mode


def _build_sam_objects(
    voc_objects: list[dict[str, Any]],
    *,
    width: int,
    height: int,
    point_mode: str,
    background_classes: list[str],
    match_mode: str,
    background_assign: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """
    前景多边形 → 前景点（label=1）；背景多边形 → 背景点（label=0）并归属到最近前景目标。
    """
    meta = {
        "foreground_total": 0,
        "background_total": 0,
        "background_assigned": 0,
        "background_orphan": 0,
    }
    foreground_voc, background_voc = split_voc_objects_by_role(
        voc_objects,
        background_classes,
        match_mode=match_mode,
    )
    meta["foreground_total"] = len(foreground_voc)
    meta["background_total"] = len(background_voc)

    if not foreground_voc:
        return [], meta

    sam_by_id: dict[int, dict[str, Any]] = {}
    for idx, obj in enumerate(foreground_voc, start=1):
        obj_point_mode = _effective_point_mode(obj, point_mode)
        prompt_pts = polygon_to_prompt_points(
            obj["polygon"],
            mode=obj_point_mode,
            source_points=obj.get("source_points"),
        )
        prompt_pts = [clip_point(p[0], p[1], width, height) for p in prompt_pts]
        sam_by_id[idx] = {
            "object_id": idx,
            "name": obj["name"],
            "points": prompt_pts,
            "labels": [1] * len(prompt_pts),
            "foreground_points": list(prompt_pts),
            "background_points": [],
            "source_polygon": obj["polygon"],
            "source_role": "foreground",
        }

    fg_for_assign = [
        {
            "object_id": idx,
            "polygon": obj["polygon"],
            "x1": obj["x1"],
            "y1": obj["y1"],
            "x2": obj["x2"],
            "y2": obj["y2"],
        }
        for idx, obj in enumerate(foreground_voc, start=1)
    ]

    for bg in background_voc:
        bg_point_mode = _effective_point_mode(bg, point_mode)
        bg_pts = polygon_to_prompt_points(
            bg["polygon"],
            mode=bg_point_mode,
            source_points=bg.get("source_points"),
        )
        target_id = assign_background_object_id(
            bg,
            fg_for_assign,
            mode=background_assign,
        )
        if target_id is None or target_id not in sam_by_id:
            meta["background_orphan"] += 1
            logger.warning(
                "skip background polygon %s: no foreground target to assign",
                bg.get("name"),
            )
            continue

        entry = sam_by_id[target_id]
        for px, py in bg_pts:
            px, py = clip_point(px, py, width, height)
            entry["points"].append([px, py])
            entry["labels"].append(0)
            entry.setdefault("background_points", []).append([px, py])
        entry.setdefault("background_sources", []).append(
            {
                "name": bg["name"],
                "polygon": bg["polygon"],
            }
        )
        meta["background_assigned"] += 1

    return [sam_by_id[k] for k in sorted(sam_by_id)], meta


def _log_scan_summary(scan: dict[str, Any]) -> None:
    fmt = scan.get("format", "unknown")
    logger.info(
        "scan[%s]: images=%d annotations=%d paired=%d image_no_ann=%d ann_no_image=%d",
        fmt,
        scan["image_total"],
        scan["annotation_total"],
        len(scan["pairs"]),
        len(scan["images_without_annotation"]),
        len(scan["annotation_without_image"]),
    )
    if scan["images_without_annotation"]:
        logger.info(
            "skip %d images without annotation file (not exported)",
            len(scan["images_without_annotation"]),
        )
    for ann_path in scan["annotation_without_image"]:
        logger.warning("skip orphan annotation (no image): %s", ann_path.name)


def convert_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    point_mode: str | None = "centroid",
    config_path: str | Path | None = None,
    background_classes: list[str] | None = None,
    copy_images: bool = True,
    annotation_format: str = "auto",
) -> dict[str, int]:
    """
    先扫描 ``input_dir``：仅处理 **有 LabelMe json 或 VOC xml 标注** 的样本；
    无标注文件的图片不处理、不输出。
    """
    cfg = load_sam_points_config(config_path)
    bg_classes = (
        list(background_classes)
        if background_classes is not None
        else list(cfg["background_classes"])
    )
    match_mode = str(cfg["match_mode"])
    background_assign = str(cfg["background_assign"])
    effective_point_mode = str(point_mode or cfg.get("point_mode") or "centroid").strip().lower()
    if effective_point_mode not in POINT_MODES:
        raise ValueError(
            f"unsupported point_mode: {effective_point_mode!r} "
            f"(expected one of {sorted(POINT_MODES)})"
        )

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input dir not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    scan = scan_annotation_pairs(input_dir, annotation_format=annotation_format)
    ann_format = str(scan.get("format") or annotation_format)
    _log_scan_summary(scan)

    stats: dict[str, Any] = {
        "annotation_format": ann_format,
        "image_total": int(scan["image_total"]),
        "annotation_total": int(scan["annotation_total"]),
        "paired_total": len(scan["pairs"]),
        "image_without_annotation": len(scan["images_without_annotation"]),
        "annotation_without_image": len(scan["annotation_without_image"]),
        "json_written": 0,
        "images_copied": 0,
        "object_total": 0,
        "foreground_total": 0,
        "background_total": 0,
        "background_assigned": 0,
        "background_orphan": 0,
        "skipped_no_objects": 0,
        "skipped_no_foreground": 0,
    }

    for ann_path, image_path in scan["pairs"]:
        width, height, _, ann_objects, image_name = parse_annotation_objects(
            ann_path,
            input_dir,
            annotation_format=ann_format,
        )
        if not ann_objects:
            logger.warning("skip %s: annotation has no shapes/objects", ann_path.name)
            stats["skipped_no_objects"] += 1
            continue

        sam_objects, meta = _build_sam_objects(
            ann_objects,
            width=width,
            height=height,
            point_mode=effective_point_mode,
            background_classes=bg_classes,
            match_mode=match_mode,
            background_assign=background_assign,
        )
        if not sam_objects:
            logger.warning(
                "skip %s: only background shapes (%d), no foreground to export",
                ann_path.name,
                meta["background_total"],
            )
            stats["skipped_no_foreground"] += 1
            continue

        stem = ann_path.stem
        out_image_name = image_path.name or image_name
        record = build_points_record(
            image_name=out_image_name,
            width=width,
            height=height,
            objects=sam_objects,
            source_annotation=ann_path.name,
            background_classes=bg_classes,
        )
        save_points_json(record, output_dir / f"{stem}.json")
        stats["json_written"] += 1

        if copy_images:
            shutil.copy2(image_path, output_dir / out_image_name)
            stats["images_copied"] += 1

        stats["object_total"] += len(sam_objects)
        stats["foreground_total"] += meta["foreground_total"]
        stats["background_total"] += meta["background_total"]
        stats["background_assigned"] += meta["background_assigned"]
        stats["background_orphan"] += meta["background_orphan"]
        logger.info(
            "exported %s + %s (%d targets, bg assigned %d) from %s",
            f"{stem}.json",
            out_image_name if copy_images else "(no image copy)",
            len(sam_objects),
            meta["background_assigned"],
            ann_path.name,
        )

    return stats


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )

    # 按需修改
    INPUT_DIR = "/Volumes/shunyao-h1/训练数据/训练集/大虫生产分割标注"
    OUTPUT_DIR = "/Volumes/shunyao-h1/训练数据/训练集/大虫生产分割标注-sam-points"
    # centroid | bbox_center | vertices（多边形全部顶点；LabelMe 优先用原始 points）
    # LabelMe 折线（line / linestrip / polyline）始终按全部顶点取点
    POINT_MODE = "vertices"
    # auto | labelme | voc
    ANNOTATION_FORMAT = "labelme"
    # None → 使用 sam-demo/config/sam_points.json
    CONFIG_PATH: str | Path | None = None
    # 非空时覆盖配置文件中的 background_classes
    BACKGROUND_CLASSES: list[str] | None = None
    # 仅输出有有效标注的样本：json + 图片复制到 OUTPUT_DIR
    COPY_IMAGES = True

    stats = convert_directory(
        INPUT_DIR,
        OUTPUT_DIR,
        point_mode=POINT_MODE,
        config_path=CONFIG_PATH,
        background_classes=BACKGROUND_CLASSES,
        copy_images=COPY_IMAGES,
        annotation_format=ANNOTATION_FORMAT,
    )
    logger.info("done: %s", stats)
