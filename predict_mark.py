#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Detail  : 对检测目录图片执行 predict_all 推理；若存在同名 VOC xml 则几何匹配后
#           保留原标注、仅追加未匹配的新检测框，写出到目标目录形成合并标注。

from __future__ import annotations

import logging
import platform
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Literal
from xml.dom import minidom

import cv2
import numpy as np

AppendShape = Literal["bndbox", "polygon"]

_FILE = Path(__file__).resolve()
_ROOT = _FILE.parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from script.config_paths import INSECT_ALG_ALL_JSON_REL, resolve_insect_alg_all_path
from script.predict_all import (  # noqa: E402
    _clear_run_output_dir,
    _result_row_to_pred_dict,
    create_pipeline,
    draw_results,
)
from script.predict_seg_lib import bbox_from_row, collect_images
from script.predict_size_validate_lib import (
    match_pred_gt,
    outputs_exist_for_incremental_skip,
    parse_pascal_voc_objects,
)


def normalize_append_shape(shape: str) -> AppendShape:
    """新追加检测框写出形态：``bndbox`` 仅外接矩形；``polygon`` 写 segmentation（无 mask 时退化为矩形四点）。"""
    s = str(shape or "bndbox").strip().lower()
    if s not in ("bndbox", "polygon"):
        raise ValueError(f"append_shape must be 'bndbox' or 'polygon', got {shape!r}")
    return s  # type: ignore[return-value]


def _clip_box(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> tuple[int, int, int, int]:
    x1 = int(max(0, min(x1, width - 1)))
    y1 = int(max(0, min(y1, height - 1)))
    x2 = int(max(0, min(x2, width)))
    y2 = int(max(0, min(y2, height)))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return x1, y1, x2, y2


def _write_xml_tree(tree: ET.ElementTree, xml_path: Path) -> None:
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    rough = ET.tostring(tree.getroot(), encoding="utf-8")
    parsed = minidom.parseString(rough)
    pretty = parsed.toprettyxml(indent="\t", encoding="utf-8")
    with open(xml_path, "wb") as f:
        f.write(pretty)


def _append_pred_object(
    annotation_root: ET.Element,
    result: dict[str, Any],
    *,
    width: int,
    height: int,
    append_shape: AppendShape = "bndbox",
) -> None:
    """将单条 predict 结果追加为 VOC ``<object>``。"""
    row = _result_row_to_pred_dict(result)
    poly = list(result.get("polygon") or [])
    if poly:
        x1, y1, x2, y2 = bbox_from_row({"polygon": poly})
    else:
        x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
    x1, y1, x2, y2 = _clip_box(x1, y1, x2, y2, width, height)

    obj = ET.SubElement(annotation_root, "object")
    ET.SubElement(obj, "name").text = str(row["cls_name"])
    det_conf = float(row.get("conf", 0.0) or 0.0)
    cls_conf = float(row.get("cls_conf", det_conf) or det_conf)
    ET.SubElement(obj, "score").text = f"{cls_conf:.6f}"
    ET.SubElement(obj, "det_score").text = f"{det_conf:.6f}"
    ET.SubElement(obj, "pose").text = "Unspecified"
    ET.SubElement(obj, "truncated").text = "0"
    ET.SubElement(obj, "difficult").text = "0"

    write_polygon = append_shape == "polygon"
    if write_polygon:
        if len(poly) >= 3:
            clipped_poly: list[list[int]] = []
            for px, py in poly:
                clipped_poly.append(
                    [
                        int(max(0, min(int(px), width - 1))),
                        int(max(0, min(int(py), height - 1))),
                    ]
                )
        else:
            clipped_poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        if len(clipped_poly) >= 3:
            seg_el = ET.SubElement(obj, "segmentation")
            pts_str = ",".join(f"{int(p[0])},{int(p[1])}" for p in clipped_poly)
            ET.SubElement(seg_el, "points").text = pts_str

    bnd = ET.SubElement(obj, "bndbox")
    ET.SubElement(bnd, "xmin").text = str(x1)
    ET.SubElement(bnd, "ymin").text = str(y1)
    ET.SubElement(bnd, "xmax").text = str(x2)
    ET.SubElement(bnd, "ymax").text = str(y2)


def _update_annotation_meta(
    annotation_root: ET.Element,
    *,
    folder_name: str,
    image_filename: str,
    width: int,
    height: int,
    depth: int,
) -> None:
    folder_el = annotation_root.find("folder")
    if folder_el is None:
        folder_el = ET.Element("folder")
        annotation_root.insert(0, folder_el)
    folder_el.text = folder_name or ""

    filename_el = annotation_root.find("filename")
    if filename_el is None:
        filename_el = ET.Element("filename")
        annotation_root.insert(1, filename_el)
    filename_el.text = image_filename

    size_el = annotation_root.find("size")
    if size_el is None:
        size_el = ET.SubElement(annotation_root, "size")
    for tag, val in (("width", width), ("height", height), ("depth", depth)):
        child = size_el.find(tag)
        if child is None:
            child = ET.SubElement(size_el, tag)
        child.text = str(int(val))


def _new_annotation_root(
    *,
    folder_name: str,
    image_filename: str,
    width: int,
    height: int,
    depth: int,
    segmented: bool,
) -> ET.Element:
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = folder_name or ""
    ET.SubElement(root, "filename").text = image_filename
    src = ET.SubElement(root, "source")
    ET.SubElement(src, "database").text = "Unknown"
    size_el = ET.SubElement(root, "size")
    ET.SubElement(size_el, "width").text = str(int(width))
    ET.SubElement(size_el, "height").text = str(int(height))
    ET.SubElement(size_el, "depth").text = str(int(depth))
    ET.SubElement(root, "segmented").text = "1" if segmented else "0"
    return root


def write_pred_xml_for_results(
    image_bgr: np.ndarray,
    results: list[dict[str, Any]],
    *,
    out_dir: Path,
    rel_path: Path,
    append_shape: AppendShape = "bndbox",
) -> None:
    """无源 xml 时，按 ``append_shape`` 写出纯推理结果 VOC xml。"""
    h_img, w_img = image_bgr.shape[:2]
    depth = 3 if image_bgr.ndim >= 3 else 1
    out_dir.mkdir(parents=True, exist_ok=True)
    root = _new_annotation_root(
        folder_name=out_dir.name or "",
        image_filename=rel_path.name,
        width=w_img,
        height=h_img,
        depth=depth,
        segmented=append_shape == "polygon",
    )
    for result in results:
        _append_pred_object(
            root,
            result,
            width=w_img,
            height=h_img,
            append_shape=append_shape,
        )
    _write_xml_tree(ET.ElementTree(root), out_dir / f"{rel_path.stem}.xml")


def merge_xml_keep_gt_append_new_preds(
    src_xml_path: Path,
    results: list[dict[str, Any]],
    *,
    out_xml_path: Path,
    image_filename: str,
    folder_name: str,
    width: int,
    height: int,
    depth: int,
    geom_metric: str = "iou",
    geom_threshold: float = 0.4,
    append_shape: AppendShape = "bndbox",
) -> tuple[int, int]:
    """
    保留源 xml 全部 ``<object>``，将与 GT 几何未匹配的新检测追加到 xml。

    Returns:
        (kept_gt_count, appended_pred_count)
    """
    tree = ET.parse(str(src_xml_path))
    root = tree.getroot()
    gt_objects = root.findall("object")
    kept_gt = len(gt_objects)

    gts = parse_pascal_voc_objects(str(src_xml_path))
    pred_rows = [_result_row_to_pred_dict(r) for r in results]
    _matches, matched_p, _matched_g = match_pred_gt(
        pred_rows, gts, geom_threshold, metric=geom_metric
    )

    appended = 0
    for pi, result in enumerate(results):
        if pi in matched_p:
            continue
        _append_pred_object(
            root,
            result,
            width=width,
            height=height,
            append_shape=append_shape,
        )
        appended += 1

    _update_annotation_meta(
        root,
        folder_name=folder_name,
        image_filename=image_filename,
        width=width,
        height=height,
        depth=depth,
    )
    _write_xml_tree(tree, out_xml_path)
    return kept_gt, appended


def save_merged_annotation_for_image(
    image_bgr: np.ndarray,
    results: list[dict[str, Any]],
    *,
    src_xml_path: Path | None,
    out_dir: Path,
    rel_path: Path,
    geom_metric: str = "iou",
    geom_threshold: float = 0.4,
    append_shape: AppendShape = "bndbox",
) -> tuple[int, int]:
    """
    写出单张图的合并标注 xml。

    有源 xml → 保留原框并追加未匹配检测；无源 xml → 仅写推理结果。
    """
    h_img, w_img = image_bgr.shape[:2]
    depth = 3 if image_bgr.ndim >= 3 else 1
    out_dir.mkdir(parents=True, exist_ok=True)
    out_xml = out_dir / f"{rel_path.stem}.xml"

    if src_xml_path is not None and src_xml_path.is_file():
        return merge_xml_keep_gt_append_new_preds(
            src_xml_path,
            results,
            out_xml_path=out_xml,
            image_filename=rel_path.name,
            folder_name=out_dir.name or "",
            width=w_img,
            height=h_img,
            depth=depth,
            geom_metric=geom_metric,
            geom_threshold=geom_threshold,
            append_shape=append_shape,
        )

    write_pred_xml_for_results(
        image_bgr,
        results,
        out_dir=out_dir,
        rel_path=rel_path,
        append_shape=append_shape,
    )
    return 0, len(results)


def run_predict_mark(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    config_path: str | Path | None = None,
    root_ids: list[str] | None = None,
    geom_metric: str = "iou",
    geom_threshold: float = 0.4,
    append_shape: str | AppendShape = "bndbox",
    copy_image: bool = True,
    save_image: bool = False,
    draw_bbox: bool = True,
    draw_polygon: bool = True,
    clean_output_before_run: bool = True,
    skip_if_output_exists: bool = False,
) -> dict[str, int]:
    """批处理：推理 + 合并标注写出。"""
    shape = normalize_append_shape(append_shape)
    output_p = Path(output_dir).expanduser().resolve()
    if clean_output_before_run:
        _clear_run_output_dir(str(output_p))

    input_p, image_files = collect_images(str(input_path))
    if not image_files:
        logging.warning("未找到可处理图片: %s", input_path)
        return {
            "images_total": 0,
            "images_ok": 0,
            "images_skipped": 0,
            "images_with_xml": 0,
            "gt_objects_kept": 0,
            "pred_objects_appended": 0,
            "pred_only_images": 0,
        }

    pipeline = None
    stats = {
        "images_total": len(image_files),
        "images_ok": 0,
        "images_skipped": 0,
        "images_with_xml": 0,
        "gt_objects_kept": 0,
        "pred_objects_appended": 0,
        "pred_only_images": 0,
    }
    infer_times: list[float] = []

    try:
        print(f"配置: {resolve_insect_alg_all_path(config_path)}")
        print(f"输入: {input_p}")
        print(f"输出: {output_p}")
        print(f"共 {len(image_files)} 张图片；几何匹配 metric={geom_metric} thr={geom_threshold}")
        print(f"新检测框写出形态: append_shape={shape}")
        if skip_if_output_exists:
            print("已开启增量跳过：输出目录下同 stem 的图片与 xml 均已存在则跳过")

        for idx, img_path in enumerate(image_files, 1):
            rel = img_path.relative_to(input_p) if input_p.is_dir() else Path(img_path.name)
            out_img_dir = output_p / rel.parent
            out_img_path = output_p / rel

            if skip_if_output_exists and outputs_exist_for_incremental_skip(
                str(out_img_dir), rel
            ):
                stats["images_skipped"] += 1
                logging.info(
                    "[%s/%s] 跳过（输出已存在）: %s",
                    idx,
                    len(image_files),
                    rel.as_posix(),
                )
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                logging.warning("[%s/%s] 跳过无法读取: %s", idx, len(image_files), img_path)
                continue

            if pipeline is None:
                pipeline = create_pipeline(config_path, device=None, root_ids=root_ids)
                print(f"根模型: {list(pipeline._roots.keys())}")

            out_img_dir.mkdir(parents=True, exist_ok=True)

            _t0 = time.perf_counter()
            results = pipeline.predict(
                img,
                source_image_stem=rel.stem,
                result_output_dir=str(out_img_dir),
            )
            infer_times.append(time.perf_counter() - _t0)

            src_xml = img_path.with_suffix(".xml")
            has_xml = src_xml.is_file()
            if has_xml:
                stats["images_with_xml"] += 1

            kept, appended = save_merged_annotation_for_image(
                img,
                results,
                src_xml_path=src_xml if has_xml else None,
                out_dir=out_img_dir,
                rel_path=rel,
                geom_metric=geom_metric,
                geom_threshold=geom_threshold,
                append_shape=shape,
            )
            if has_xml:
                stats["gt_objects_kept"] += kept
                stats["pred_objects_appended"] += appended
            else:
                stats["pred_only_images"] += 1
                stats["pred_objects_appended"] += appended

            if copy_image:
                shutil.copy2(img_path, out_img_path)

            if save_image:
                vis_path = out_img_dir / f"{rel.stem}_mark.jpg"
                draw_results(
                    img,
                    results,
                    str(vis_path),
                    draw_bbox=draw_bbox,
                    draw_polygon=draw_polygon,
                    cn_index=pipeline._cn_display_index,
                )

            stats["images_ok"] += 1
            tag = "xml+pred" if has_xml else "pred-only"
            logging.info(
                "[%s/%s] %s kept_gt=%d appended=%d results=%d",
                idx,
                len(image_files),
                tag,
                kept,
                appended,
                len(results),
            )
    finally:
        if pipeline is not None:
            pipeline.release()

    if infer_times:
        total_s = sum(infer_times)
        print(
            f"推理耗时: 总 {total_s:.2f}s, 均 {total_s / len(infer_times):.3f}s/张, "
            f"最大 {max(infer_times):.3f}s"
        )
    print("======== predict_mark 汇总 ========")
    print(
        f"images_ok={stats['images_ok']}/{stats['images_total']}  "
        f"skipped={stats['images_skipped']}  "
        f"with_xml={stats['images_with_xml']}  pred_only={stats['pred_only_images']}"
    )
    print(
        f"gt_kept={stats['gt_objects_kept']}  pred_appended={stats['pred_objects_appended']}"
    )
    return stats


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    logging.info("predict_mark.py start")

    CONFIG_PATH = INSECT_ALG_ALL_JSON_REL
    if platform.system() == "Darwin":
        INPUT_PATH = "/Volumes/shunyao-h1/训练数据/测试集/北京设备全标注"
    else:
        INPUT_PATH = "/data/data-test/北京设备全标注"
    OUTPUT_DIR = str(INPUT_PATH) + "-mark"

    ROOT_IDS: list[str] | None = None
    GEOM_METRIC = "iou"  # "iou" | "ior"
    GEOM_THRESHOLD = 0.4
    # 新追加（及无源 xml 纯推理）检测框写出形态："bndbox" | "polygon"
    APPEND_SHAPE = "bndbox"
    COPY_IMAGE = True
    SAVE_IMAGE = False
    DRAW_BBOX = True
    DRAW_POLYGON = True
    CLEAN_OUTPUT_BEFORE_RUN = False
    # True：输出目录下同 stem 的图片与 xml 均已存在则跳过（增量续跑）；False：全部重跑
    SKIP_IF_OUTPUT_EXISTS = True

    run_predict_mark(
        INPUT_PATH,
        OUTPUT_DIR,
        config_path=CONFIG_PATH,
        root_ids=ROOT_IDS,
        geom_metric=GEOM_METRIC,
        geom_threshold=GEOM_THRESHOLD,
        append_shape=APPEND_SHAPE,
        copy_image=COPY_IMAGE,
        save_image=SAVE_IMAGE,
        draw_bbox=DRAW_BBOX,
        draw_polygon=DRAW_POLYGON,
        clean_output_before_run=CLEAN_OUTPUT_BEFORE_RUN,
        skip_if_output_exists=SKIP_IF_OUTPUT_EXISTS,
    )
