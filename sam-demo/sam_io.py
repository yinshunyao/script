#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SAM demo 批处理：VOC 标注读写、点提示 JSON、mask → 多边形。"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any
from xml.dom import minidom
import xml.etree.ElementTree as ET

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_SAM_POINTS_CONFIG = Path(__file__).resolve().parent / "config" / "sam_points.json"
POINT_MODES = frozenset({"centroid", "bbox_center", "vertices"})
LABELME_POLYLINE_SHAPE_TYPES = frozenset({"line", "linestrip", "polyline"})
logger = logging.getLogger(__name__)


def is_labelme_polyline_shape(shape_type: str) -> bool:
    """LabelMe 折线类 shape（line / linestrip / polyline）。"""
    return str(shape_type or "").strip().lower() in LABELME_POLYLINE_SHAPE_TYPES


def load_sam_points_config(path: str | Path | None = None) -> dict[str, Any]:
    """加载点提示配置（背景类别等）。"""
    cfg_path = Path(path) if path else DEFAULT_SAM_POINTS_CONFIG
    if not cfg_path.is_file():
        return {
            "background_classes": [],
            "match_mode": "exact",
            "background_assign": "nearest_foreground",
            "point_mode": "centroid",
        }
    with open(cfg_path, encoding="utf-8") as f:
        data = json.load(f)
    bg = data.get("background_classes") or []
    point_mode = str(data.get("point_mode") or "centroid").strip().lower()
    if point_mode not in POINT_MODES:
        point_mode = "centroid"
    return {
        "background_classes": [str(x).strip() for x in bg if str(x).strip()],
        "match_mode": str(data.get("match_mode") or "exact").strip().lower(),
        "background_assign": str(
            data.get("background_assign") or "nearest_foreground"
        ).strip().lower(),
        "point_mode": point_mode,
    }


def normalize_class_name(name: str) -> str:
    return str(name or "").strip().casefold()


def is_background_class(name: str, background_classes: list[str], *, match_mode: str = "exact") -> bool:
    """判断 VOC ``name`` 是否属于背景类别配置。"""
    if not background_classes:
        return False
    raw = str(name or "").strip()
    norm = normalize_class_name(raw)
    for item in background_classes:
        item_raw = str(item).strip()
        item_norm = normalize_class_name(item_raw)
        if not item_norm:
            continue
        if match_mode == "contains":
            if item_norm in norm or item_raw in raw:
                return True
        elif norm == item_norm or raw == item_raw:
            return True
    return False


def polygon_centroid(polygon: list[list[int]]) -> tuple[float, float]:
    pts = np.asarray(polygon, dtype=np.float64)
    if pts.size == 0:
        raise ValueError("empty polygon")
    return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))


def bbox_iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax1, ay1, ax2, ay2 = int(a["x1"]), int(a["y1"]), int(a["x2"]), int(a["y2"])
    bx1, by1, bx2, by2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])
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


def assign_background_object_id(
    bg_obj: dict[str, Any],
    foreground_objects: list[dict[str, Any]],
    *,
    mode: str = "nearest_foreground",
) -> int | None:
    """
    将背景多边形归属到某个前景 ``object_id``。

    - ``nearest_foreground``：质心距离最近的前景（默认）
    - ``max_iou``：与背景框 IoU 最大的前景；IoU 全为 0 时回退最近质心
    """
    if not foreground_objects:
        return None

    bg_cx, bg_cy = polygon_centroid(bg_obj["polygon"])
    if mode == "max_iou":
        best_id = None
        best_iou = -1.0
        for fg in foreground_objects:
            iou = bbox_iou(bg_obj, fg)
            if iou > best_iou:
                best_iou = iou
                best_id = int(fg["object_id"])
        if best_iou > 0 and best_id is not None:
            return best_id

    best_id = None
    best_dist = math.inf
    for fg in foreground_objects:
        fg_cx, fg_cy = polygon_centroid(fg["polygon"])
        dist = (bg_cx - fg_cx) ** 2 + (bg_cy - fg_cy) ** 2
        if dist < best_dist:
            best_dist = dist
            best_id = int(fg["object_id"])
    return best_id


def split_voc_objects_by_role(
    voc_objects: list[dict[str, Any]],
    background_classes: list[str],
    *,
    match_mode: str = "exact",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    foreground: list[dict[str, Any]] = []
    background: list[dict[str, Any]] = []
    for obj in voc_objects:
        if is_background_class(obj["name"], background_classes, match_mode=match_mode):
            background.append(obj)
        else:
            foreground.append(obj)
    return foreground, background


def list_images(directory: Path) -> list[Path]:
    out: list[Path] = []
    for p in sorted(directory.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            out.append(p)
    return out


def find_image_for_stem(directory: Path, stem: str) -> Path | None:
    for suf in IMAGE_SUFFIXES:
        candidate = directory / f"{stem}{suf}"
        if candidate.is_file():
            return candidate
    return None


def find_xml_for_stem(directory: Path, stem: str) -> Path | None:
    candidate = directory / f"{stem}.xml"
    return candidate if candidate.is_file() else None


def is_sam_points_export(data: dict[str, Any]) -> bool:
    """本 pipeline 输出的点提示 json（非 LabelMe 原始标注）。"""
    return isinstance(data.get("objects"), list) and "shapes" not in data


def is_labelme_annotation(data: dict[str, Any]) -> bool:
    return isinstance(data.get("shapes"), list)


def load_labelme_json(json_path: str | Path) -> dict[str, Any]:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    if not is_labelme_annotation(data):
        raise ValueError(f"not a LabelMe annotation json: {json_path}")
    return data


def labelme_shape_points_to_ints(
    points: list[list[float]] | list[tuple[float, float]],
) -> list[list[int]]:
    """LabelMe ``shapes[].points`` 原始顶点 → 整数像素坐标（不做 bbox 展开）。"""
    out: list[list[int]] = []
    for p in points or []:
        if len(p) < 2:
            continue
        out.append([int(round(float(p[0]))), int(round(float(p[1])))])
    return out


def labelme_points_to_polygon(
    points: list[list[float]] | list[tuple[float, float]],
    shape_type: str,
) -> list[list[int]]:
    """LabelMe shape → 整数像素多边形（折线类保留原始顶点，不展开为 bbox）。"""
    pts = [[float(p[0]), float(p[1])] for p in (points or []) if len(p) >= 2]
    st = str(shape_type or "polygon").strip().lower()

    if is_labelme_polyline_shape(st) and len(pts) >= 2:
        return [[int(round(p[0])), int(round(p[1]))] for p in pts]

    if st == "rectangle" and len(pts) >= 2:
        x1, y1 = pts[0]
        x2, y2 = pts[1]
        xmin, xmax = min(x1, x2), max(x1, x2)
        ymin, ymax = min(y1, y2), max(y1, y2)
        return [
            [int(round(xmin)), int(round(ymin))],
            [int(round(xmax)), int(round(ymin))],
            [int(round(xmax)), int(round(ymax))],
            [int(round(xmin)), int(round(ymax))],
        ]

    if len(pts) >= 3:
        return [[int(round(p[0])), int(round(p[1]))] for p in pts]

    if len(pts) == 2:
        x1, y1 = pts[0]
        x2, y2 = pts[1]
        xmin, xmax = min(x1, x2), max(x1, x2)
        ymin, ymax = min(y1, y2), max(y1, y2)
        return [
            [int(round(xmin)), int(round(ymin))],
            [int(round(xmax)), int(round(ymin))],
            [int(round(xmax)), int(round(ymax))],
            [int(round(xmin)), int(round(ymax))],
        ]

    if len(pts) == 1:
        return [[int(round(pts[0][0])), int(round(pts[0][1]))]]

    return []


def resolve_labelme_image_path(
    input_dir: Path,
    json_path: Path,
    data: dict[str, Any],
) -> Path | None:
    """按 ``imagePath`` 或 json 同名 stem 查找图片文件。"""
    input_dir = Path(input_dir)
    image_path_field = str(data.get("imagePath") or "").strip()
    candidates: list[Path] = []
    if image_path_field:
        p = Path(image_path_field)
        candidates.extend([input_dir / image_path_field, input_dir / p.name])
    found = find_image_for_stem(input_dir, json_path.stem)
    if found is not None:
        candidates.append(found)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def materialize_labelme_image(
    input_dir: Path,
    json_path: Path,
    data: dict[str, Any],
    *,
    dest_path: Path | None = None,
) -> Path | None:
    """
    若磁盘无图片，尝试从 LabelMe ``imageData``（base64）写出 jpg/png。
    """
    import base64

    existing = resolve_labelme_image_path(input_dir, json_path, data)
    if existing is not None:
        return existing

    image_data = data.get("imageData")
    if not image_data:
        return None

    image_path_field = str(data.get("imagePath") or "").strip()
    if dest_path is not None:
        out_path = Path(dest_path)
    elif image_path_field:
        out_path = input_dir / Path(image_path_field).name
    else:
        out_path = input_dir / f"{json_path.stem}.jpg"

    raw = base64.b64decode(image_data)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(raw)
    return out_path if out_path.is_file() else None


def parse_labelme_objects(
    json_path: str | Path,
    input_dir: str | Path | None = None,
    *,
    data: dict[str, Any] | None = None,
) -> tuple[int, int, int, list[dict[str, Any]], str]:
    """
    读取 LabelMe json，返回 (width, height, depth, objects, image_filename)。

    objects 结构与 ``parse_voc_objects`` 一致：name, polygon, x1,y1,x2,y2。
    """
    json_path = Path(json_path)
    payload = data if data is not None else load_labelme_json(json_path)

    width = int(payload.get("imageWidth") or 0)
    height = int(payload.get("imageHeight") or 0)
    image_name = str(payload.get("imagePath") or f"{json_path.stem}.jpg").strip()
    if Path(image_name).name:
        image_name = Path(image_name).name

    objects: list[dict[str, Any]] = []
    for shape in payload.get("shapes") or []:
        if not isinstance(shape, dict):
            continue
        label = str(shape.get("label") or "").strip()
        if not label:
            continue
        shape_type = str(shape.get("shape_type") or "polygon")
        raw_points = labelme_shape_points_to_ints(shape.get("points") or [])
        poly = labelme_points_to_polygon(shape.get("points") or [], shape_type)
        if not poly:
            continue
        if is_labelme_polyline_shape(shape_type) and len(raw_points) < 2:
            continue
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        objects.append(
            {
                "name": label,
                "polygon": poly,
                "source_points": raw_points,
                "shape_type": shape_type,
                "x1": min(xs),
                "y1": min(ys),
                "x2": max(xs),
                "y2": max(ys),
            }
        )

    if (width <= 0 or height <= 0) and input_dir is not None:
        img_path = resolve_labelme_image_path(Path(input_dir), json_path, payload)
        if img_path is not None and img_path.is_file():
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is not None:
                height, width = img.shape[:2]

    return width, height, 3, objects, image_name


def _image_index(input_dir: Path) -> dict[str, Path]:
    image_by_stem: dict[str, Path] = {}
    for img in list_images(input_dir):
        prev = image_by_stem.get(img.stem)
        if prev is None or img.suffix.lower() == ".jpg":
            image_by_stem[img.stem] = img
    return image_by_stem


def scan_voc_pairs(input_dir: Path) -> dict[str, Any]:
    input_dir = Path(input_dir)
    images = list_images(input_dir)
    xml_files = sorted(
        p for p in input_dir.glob("*.xml") if p.is_file() and not p.name.startswith("._")
    )
    image_by_stem = _image_index(input_dir)
    xml_by_stem = {p.stem: p for p in xml_files}

    pairs: list[tuple[Path, Path]] = []
    images_without_ann: list[Path] = []
    for stem in sorted(image_by_stem):
        img = image_by_stem[stem]
        ann = xml_by_stem.get(stem)
        if ann is not None:
            pairs.append((ann, img))
        else:
            images_without_ann.append(img)

    ann_without_image = [xml_by_stem[s] for s in sorted(xml_by_stem) if s not in image_by_stem]
    return {
        "format": "voc",
        "pairs": pairs,
        "images_without_annotation": images_without_ann,
        "annotation_without_image": ann_without_image,
        "image_total": len(images),
        "annotation_total": len(xml_files),
    }


def scan_labelme_pairs(input_dir: Path) -> dict[str, Any]:
    input_dir = Path(input_dir)
    images = list_images(input_dir)
    image_by_stem = _image_index(input_dir)
    image_by_name = {p.name: p for p in images}

    json_files = sorted(
        p for p in input_dir.glob("*.json") if p.is_file() and not p.name.startswith("._")
    )

    labelme_files: list[Path] = []
    for json_path in json_files:
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            logger.warning("skip invalid json: %s", json_path.name)
            continue
        if is_sam_points_export(data):
            continue
        if is_labelme_annotation(data):
            labelme_files.append(json_path)

    paired_stems: set[str] = set()
    pairs: list[tuple[Path, Path]] = []
    ann_without_image: list[Path] = []

    for json_path in labelme_files:
        data = load_labelme_json(json_path)
        image_path = resolve_labelme_image_path(input_dir, json_path, data)
        if image_path is None:
            image_path = materialize_labelme_image(input_dir, json_path, data)
        if image_path is None or not image_path.is_file():
            ann_without_image.append(json_path)
            continue
        pairs.append((json_path, image_path))
        paired_stems.add(json_path.stem)
        if image_path.name in image_by_name:
            paired_stems.add(image_path.stem)

    images_without_ann = [
        img for stem, img in sorted(image_by_stem.items()) if stem not in paired_stems
    ]

    return {
        "format": "labelme",
        "pairs": pairs,
        "images_without_annotation": images_without_ann,
        "annotation_without_image": ann_without_image,
        "image_total": len(images),
        "annotation_total": len(labelme_files),
    }


def detect_annotation_format(input_dir: Path) -> str:
    input_dir = Path(input_dir)
    for json_path in input_dir.glob("*.json"):
        if json_path.name.startswith("._"):
            continue
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            continue
        if is_labelme_annotation(data) and not is_sam_points_export(data):
            return "labelme"
    if any(input_dir.glob("*.xml")):
        return "voc"
    return "labelme"


def scan_annotation_pairs(
    input_dir: Path,
    *,
    annotation_format: str = "auto",
) -> dict[str, Any]:
    """
    扫描目录，仅保留 **有标注文件** 的样本。

    ``annotation_format``：``auto`` | ``labelme`` | ``voc``
    """
    input_dir = Path(input_dir)
    fmt = str(annotation_format or "auto").strip().lower()
    if fmt == "auto":
        fmt = detect_annotation_format(input_dir)

    if fmt == "labelme":
        return scan_labelme_pairs(input_dir)
    if fmt == "voc":
        return scan_voc_pairs(input_dir)
    raise ValueError(f"unsupported annotation_format: {annotation_format!r}")


def parse_annotation_objects(
    ann_path: Path,
    input_dir: Path,
    *,
    annotation_format: str,
) -> tuple[int, int, int, list[dict[str, Any]], str]:
    """统一解析 VOC xml 或 LabelMe json。"""
    fmt = annotation_format.lower()
    if fmt == "labelme" or ann_path.suffix.lower() == ".json":
        return parse_labelme_objects(ann_path, input_dir)
    width, height, depth, objects = parse_voc_objects(ann_path)
    return width, height, depth, objects, f"{ann_path.stem}.jpg"


def parse_voc_objects(xml_path: str | Path) -> tuple[int, int, int, list[dict[str, Any]]]:
    """
    读取 VOC xml：优先 ``segmentation/points`` 多边形，否则 ``bndbox`` 四角。

    返回 (width, height, depth, objects)；每个 object 含 name, polygon, x1,y1,x2,y2。
    """
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    size = root.find("size")
    width = int(float(size.find("width").text)) if size is not None else 0
    height = int(float(size.find("height").text)) if size is not None else 0
    depth_el = size.find("depth") if size is not None else None
    depth = int(float(depth_el.text)) if depth_el is not None and depth_el.text else 3

    objects: list[dict[str, Any]] = []
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
                    raise ValueError(f"missing {tag} in {xml_path}")
                return int(float(el.text.strip()))

            x1, y1, x2, y2 = _int("xmin"), _int("ymin"), _int("xmax"), _int("ymax")
            poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        else:
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

        objects.append(
            {
                "name": name,
                "polygon": poly,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )
    return width, height, depth, objects


def polygon_to_prompt_point(
    polygon: list[list[int]],
    *,
    mode: str = "centroid",
) -> list[int]:
    """将多边形（或 bbox 四角）转为 SAM 前景点坐标 [x, y]。"""
    points = polygon_to_prompt_points(
        polygon,
        mode=mode,
    )
    return points[0]


def polygon_to_prompt_points(
    polygon: list[list[int]],
    *,
    mode: str = "centroid",
    source_points: list[list[int]] | None = None,
) -> list[list[int]]:
    """
    将多边形转为 SAM 点提示坐标列表。

    ``vertices``：优先使用 LabelMe 原始顶点（``source_points``），否则使用 ``polygon`` 全部顶点；
    其余模式各返回 1 个点（质心或 bbox 中心）。
    """
    mode = str(mode or "centroid").strip().lower()
    if mode not in POINT_MODES:
        raise ValueError(f"unsupported point mode: {mode!r} (expected one of {sorted(POINT_MODES)})")

    if mode == "vertices":
        pts = source_points if source_points else polygon
        if not pts:
            raise ValueError("empty polygon")
        return [[int(p[0]), int(p[1])] for p in pts]

    if not polygon:
        raise ValueError("empty polygon")
    arr = np.asarray(polygon, dtype=np.float64)
    if mode == "bbox_center":
        x = float((arr[:, 0].min() + arr[:, 0].max()) / 2.0)
        y = float((arr[:, 1].min() + arr[:, 1].max()) / 2.0)
    else:
        x = float(np.mean(arr[:, 0]))
        y = float(np.mean(arr[:, 1]))
    return [[int(round(x)), int(round(y))]]


def clip_point(x: int, y: int, width: int, height: int) -> list[int]:
    if width > 0:
        x = int(max(0, min(width - 1, x)))
    if height > 0:
        y = int(max(0, min(height - 1, y)))
    return [x, y]


def build_points_record(
    *,
    image_name: str,
    width: int,
    height: int,
    objects: list[dict[str, Any]],
    source_annotation: str | None = None,
    background_classes: list[str] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "image": image_name,
        "width": int(width),
        "height": int(height),
        "source_annotation": source_annotation,
        "objects": objects,
    }
    if source_annotation and source_annotation.lower().endswith(".xml"):
        record["source_xml"] = source_annotation
    if background_classes is not None:
        record["background_classes"] = list(background_classes)
    return record


def save_points_json(record: dict[str, Any], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)


def load_points_json(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if "objects" not in data:
        raise ValueError(f"invalid points json (missing objects): {path}")
    return data


LABELME_VERSION = "5.4.1"


def build_labelme_shape_from_polygon(
    label: str,
    polygon: list[list[int]] | list[list[float]],
    *,
    object_id: int | None = None,
    sam_score: float | None = None,
) -> dict[str, Any]:
    """SAM 多边形 → LabelMe ``polygon`` shape（可直接用 LabelMe 打开查看）。"""
    points = [[float(p[0]), float(p[1])] for p in polygon]
    description = f"sam_score={sam_score:.6f}" if sam_score is not None else ""
    return {
        "label": str(label),
        "points": points,
        "group_id": int(object_id) if object_id is not None else None,
        "description": description,
        "shape_type": "polygon",
        "flags": {},
        "mask": None,
    }


def write_labelme_seg_json(
    out_path: str | Path,
    *,
    image_filename: str,
    width: int,
    height: int,
    results: list[dict[str, Any]],
) -> None:
    """
    写出 LabelMe 分割 json。

    ``results`` 每项含 ``name``、``polygon``，可选 ``object_id``、``sam_score``。
    ``imageData=null``，需与 ``imagePath`` 同名图片同目录存放以便 LabelMe 打开。
    """
    shapes: list[dict[str, Any]] = []
    for row in results:
        polygon = list(row.get("polygon") or [])
        if len(polygon) < 3:
            continue
        shapes.append(
            build_labelme_shape_from_polygon(
                str(row.get("name", "unknown")),
                polygon,
                object_id=row.get("object_id"),
                sam_score=row.get("sam_score"),
            )
        )
    payload = {
        "version": LABELME_VERSION,
        "flags": {},
        "shapes": shapes,
        "imagePath": str(image_filename),
        "imageData": None,
        "imageHeight": int(height),
        "imageWidth": int(width),
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def mask_to_polygon(
    mask: np.ndarray,
    *,
    epsilon_ratio: float = 0.002,
    min_points: int = 3,
) -> list[list[int]]:
    """二值 mask → 外轮廓多边形（像素坐标）。"""
    mask_u8 = np.asarray(mask, dtype=np.uint8)
    if mask_u8.ndim != 2:
        raise ValueError("mask must be 2D")
    if mask_u8.max() <= 1:
        mask_u8 = mask_u8 * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) <= 0:
        return []
    peri = cv2.arcLength(cnt, True)
    eps = max(1.0, float(epsilon_ratio) * float(peri))
    approx = cv2.approxPolyDP(cnt, eps, True)
    if approx is None or len(approx) < min_points:
        return []
    return [[int(p[0][0]), int(p[0][1])] for p in approx]


def write_voc_seg_xml(
    xml_path: str | Path,
    *,
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
        if poly:
            xs = [int(p[0]) for p in poly]
            ys = [int(p[1]) for p in poly]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        else:
            x1 = int(r.get("x1", 0))
            y1 = int(r.get("y1", 0))
            x2 = int(r.get("x2", 0))
            y2 = int(r.get("y2", 0))
            poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

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
        ET.SubElement(obj, "name").text = str(r.get("name", "unknown"))
        if r.get("object_id") is not None:
            ET.SubElement(obj, "object_id").text = str(int(r["object_id"]))
        if r.get("sam_score") is not None:
            ET.SubElement(obj, "sam_score").text = f"{float(r['sam_score']):.6f}"
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
    xml_path = Path(xml_path)
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(xml_path, "wb") as f:
        f.write(pretty)


def read_image_rgb(image_path: str | Path) -> np.ndarray:
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"cannot read image: {image_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
