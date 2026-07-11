#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Detail  : 读取 SAM 点提示 JSON，推理并写出 LabelMe 分割 JSON（可直接 labelme 打开查看）
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from sam2_engine import Sam2Engine
from sam_io import (
    find_image_for_stem,
    is_sam_points_export,
    load_points_json,
    mask_to_polygon,
    read_image_rgb,
    write_labelme_seg_json,
)

logger = logging.getLogger(__name__)


def _resolve_image_path(input_dir: Path, record: dict[str, Any], json_path: Path) -> Path:
    image_name = str(record.get("image") or "").strip()
    if image_name:
        direct = input_dir / image_name
        if direct.is_file():
            return direct
    image_path = find_image_for_stem(input_dir, json_path.stem)
    if image_path is None:
        raise FileNotFoundError(
            f"image not found for {json_path.name} "
            f"(image field={image_name!r}, stem={json_path.stem})"
        )
    return image_path


def infer_one_image(
    engine: Sam2Engine,
    input_dir: Path,
    json_path: Path,
    out_labelme_path: Path,
    *,
    contour_epsilon_ratio: float = 0.002,
    min_contour_points: int = 3,
    copy_image: bool = True,
) -> dict[str, Any]:
    record = load_points_json(json_path)
    image_path = _resolve_image_path(input_dir, record, json_path)
    image_rgb = read_image_rgb(image_path)
    h, w = image_rgb.shape[:2]

    width = int(record.get("width") or w)
    height = int(record.get("height") or h)
    image_filename = image_path.name

    engine.set_image(image_rgb)
    results: list[dict[str, Any]] = []

    for obj in record["objects"]:
        object_id = int(obj.get("object_id", len(results) + 1))
        name = str(obj.get("name", "unknown"))
        points = obj.get("points") or []
        labels = obj.get("labels") or [1] * len(points)
        if not points:
            logger.warning("skip object %s in %s: no points", object_id, json_path.name)
            continue
        if len(labels) != len(points):
            labels = [1] * len(points)

        mask, score = engine.predict(
            [[int(p[0]), int(p[1])] for p in points],
            [int(v) for v in labels],
        )
        polygon = mask_to_polygon(
            mask,
            epsilon_ratio=contour_epsilon_ratio,
            min_points=min_contour_points,
        )
        if len(polygon) < 3:
            logger.warning(
                "object %s in %s: empty contour, fallback to prompt bbox",
                object_id,
                json_path.name,
            )
            xs = [int(p[0]) for p in points]
            ys = [int(p[1]) for p in points]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

        results.append(
            {
                "object_id": object_id,
                "name": name,
                "polygon": polygon,
                "sam_score": score,
            }
        )

    write_labelme_seg_json(
        out_labelme_path,
        image_filename=image_filename,
        width=width,
        height=height,
        results=results,
    )

    if copy_image:
        dest = out_labelme_path.parent / image_filename
        if not dest.is_file() or dest.resolve() != image_path.resolve():
            shutil.copy2(image_path, dest)

    return {
        "points_json": json_path.name,
        "labelme_json": out_labelme_path.name,
        "objects": len(results),
    }


def infer_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    checkpoint_path: str | Path,
    model_key: str = "sam2.1_tiny",
    device: str | None = None,
    points_subdir: str = "",
    contour_epsilon_ratio: float = 0.002,
    min_contour_points: int = 3,
    copy_images: bool = True,
) -> dict[str, Any]:
    """
    读取点提示 json，SAM2 推理后写出 LabelMe 分割 json 到 ``output_dir``。

    输出目录内每张图包含：``<stem>.json``（LabelMe 多边形）+ 同名图片（便于 ``labelme <dir>`` 打开）。
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    points_dir = input_dir / points_subdir if points_subdir else input_dir
    if not points_dir.is_dir():
        raise FileNotFoundError(f"points dir not found: {points_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    engine = Sam2Engine(
        model_key=model_key,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    stats: dict[str, Any] = {
        "points_json_total": 0,
        "labelme_written": 0,
        "images_copied": 0,
        "object_total": 0,
        "failed": 0,
    }

    json_files = sorted(p for p in points_dir.glob("*.json") if not p.name.startswith("._"))
    points_json_files: list[Path] = []
    for json_path in json_files:
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            logger.warning("skip invalid json: %s", json_path.name)
            continue
        if is_sam_points_export(data):
            points_json_files.append(json_path)

    stats["points_json_total"] = len(points_json_files)

    for json_path in points_json_files:
        out_labelme = output_dir / f"{json_path.stem}.json"
        try:
            info = infer_one_image(
                engine,
                input_dir,
                json_path,
                out_labelme,
                contour_epsilon_ratio=contour_epsilon_ratio,
                min_contour_points=min_contour_points,
                copy_image=copy_images,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("failed %s: %s", json_path.name, exc)
            stats["failed"] += 1
            continue

        stats["labelme_written"] += 1
        stats["object_total"] += int(info["objects"])
        if copy_images:
            stats["images_copied"] += 1
        logger.info(
            "wrote %s (%d shapes) from %s",
            info["labelme_json"],
            info["objects"],
            info["points_json"],
        )

    return stats


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )

    # ------------------------------------------------------------------ #
    # 入口参数（按需修改；IDE 直接运行）
    # ------------------------------------------------------------------ #

    # 01 输出目录：含点提示 json + 图片
    INPUT_DIR = "/root/sam-demo/大虫生产分割标注-sam-points/"

    # SAM 分割结果输出目录（LabelMe json + 图片）
    OUTPUT_DIR = "/root/sam-demo/大虫生产分割标注-sam-seg"

    # SAM2 权重：目录或单个 .pt 文件路径
    CHECKPOINT_PATH = "/data/models/sam2.1_hiera_small.pt"

    # 模型规格：sam2.1_tiny | sam2.1_small | sam2.1_base_plus | sam2.1_large
    MODEL_KEY = "sam2.1_small"

    # 推理设备：None 自动；或 cpu / cuda / mps
    DEVICE: str | None = None

    # 点提示 json 相对 INPUT_DIR 的子目录；与图片同目录则留空
    POINTS_SUBDIR = ""

    # mask 轮廓简化：epsilon = 比例 × 轮廓周长（越大多边形顶点越少）
    CONTOUR_EPSILON_RATIO = 0.002

    # 轮廓最少顶点数；少于该数视为退化
    MIN_CONTOUR_POINTS = 3

    # 是否将图片复制到 OUTPUT_DIR（LabelMe 需 json 与 imagePath 同目录）
    COPY_IMAGES = True

    stats = infer_directory(
        INPUT_DIR,
        OUTPUT_DIR,
        checkpoint_path=CHECKPOINT_PATH,
        model_key=MODEL_KEY,
        device=DEVICE,
        points_subdir=POINTS_SUBDIR,
        contour_epsilon_ratio=CONTOUR_EPSILON_RATIO,
        min_contour_points=MIN_CONTOUR_POINTS,
        copy_images=COPY_IMAGES,
    )
    logger.info("done: %s", stats)
