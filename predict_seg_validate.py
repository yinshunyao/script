#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Detail  : YOLO 分割模型推理与可视化验证（独立于 detect+cls 的 predict_size_validate_dual）。
#           - 模型调用：script.predict.model_seg.ModelSegmenter
#           - 业务：多边形绘制、中心点+标签、VOC(xml) 写出、可选与标注 xml 对比统计
#
# 参考 predict_size_validate_dual.py 的目录遍历与保存流程，但不依赖 PredictSize / predict_size_validate_lib。

from __future__ import annotations

import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2

_FILE = Path(__file__).resolve()
_ROOT = _FILE.parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from script.predict.model_seg import ModelSegmenter
from script.predict_seg_lib import (
    bbox_from_row,
    collect_images,
    draw_seg_output_image,
    is_class_match,
    match_pred_gt_polygon,
    normalize_class_name,
    outputs_exist_for_skip,
    parse_voc_objects,
    write_voc_seg_xml,
)


def _save_image_and_xml(
    image_bgr,
    results_visible: list[dict],
    *,
    out_dir: str,
    rel_path: Path,
    draw_polygons: bool,
    draw_bbox: bool,
    draw_center_point_and_label: bool,
    polygon_alpha: float,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    h_img, w_img = image_bgr.shape[:2]
    save_path = os.path.join(out_dir, rel_path.name)

    need_visual = bool(draw_polygons or draw_bbox or draw_center_point_and_label)
    if not need_visual:
        cv2.imwrite(save_path, image_bgr)
    else:
        img_draw = draw_seg_output_image(
            image_bgr,
            results_visible,
            draw_polygons=draw_polygons,
            draw_bbox=draw_bbox,
            draw_center_point_and_label=draw_center_point_and_label,
            polygon_alpha=polygon_alpha,
        )
        cv2.imwrite(save_path, img_draw)

    depth = 3 if image_bgr.ndim >= 3 else 1
    xml_name = Path(rel_path.name).stem + ".xml"
    xml_path = os.path.join(out_dir, xml_name)
    write_voc_seg_xml(
        xml_path,
        folder_name=os.path.basename(os.path.normpath(out_dir)) or "",
        image_filename=rel_path.name,
        width=w_img,
        height=h_img,
        depth=depth,
        results=results_visible,
    )


def _print_stat_summary(title: str, stat: dict[str, int]) -> None:
    tp = int(stat.get("tp", 0))
    fp = int(stat.get("fp", 0))
    fn = int(stat.get("fn", 0))
    ce = int(stat.get("cls_err", 0))
    denom_gt = float(tp + ce + fn)
    denom_pred = float(tp + fp)
    prec = (tp / denom_pred) if denom_pred > 0 else 0.0
    rec = (tp / denom_gt) if denom_gt > 0 else 0.0
    print(
        f"{title}: tp={tp} fp={fp} fn={fn} cls_err={ce} "
        f"precision={prec:.4f} recall={rec:.4f} pairs={stat.get('geom_pairs', 0)}"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # ----------------------- 模型与推理参数 -----------------------
    seg_model_path = os.path.expanduser(
        "~/Documents/code/ai-company/insect/doc/测试结果/分割/best.pt"
    )
    conf_thresh = 0.25
    conf_merge = 0.1
    ior_threshold = 0.5
    clip_size = 960
    overlap_size = 200
    infer_imgsz = 960
    pad_full_image_to_square = True
    detect_nms_iou = 0.65
    detect_max_det = 1000
    detect_nms_agnostic = False
    min_instance_size = 3

    # ----------------------- 绘制开关 -----------------------
    enable_draw_polygons = True
    enable_draw_bbox = False
    enable_draw_center_point_and_label = True
    polygon_fill_alpha = 0.22

    # ----------------------- 验证与类别 -----------------------
    enable_eval_with_xml = True
    val_match_use_mask_iou = True
    val_geom_threshold = 0.5
    CLASS_MERGE_TO_GROUPS: dict[str, list[str]] | None = {
        "insect": ["*"],
    }

    # ----------------------- 输入输出 -----------------------
    input_path = "/Users/shunyaoyin/Documents/code/datasets/insect-data/测试集/北京0518"
    output_dir = input_path + "-seg"
    incremental_skip_done = False
    predict_debug = False
    debug_clip = False

    segmenter = ModelSegmenter(
        model_path=seg_model_path,
        conf_thresh=conf_thresh,
        conf_merge=conf_merge,
        ior_threshold=ior_threshold,
        device=None,
        augment=False,
        nms_iou=detect_nms_iou,
        max_det=detect_max_det,
        nms_agnostic=detect_nms_agnostic,
        imgsz=infer_imgsz,
    )

    input_p, image_files = collect_images(input_path)
    print(f"共找到 {len(image_files)} 张图片")

    stat_total = {"tp": 0, "fp": 0, "fn": 0, "cls_err": 0, "geom_pairs": 0, "img_with_xml": 0}
    stat_by_cls: dict[str, dict[str, int]] = defaultdict(
        lambda: {"gt": 0, "pred": 0, "tp": 0, "fp": 0, "fn": 0, "cls_err": 0}
    )
    skipped = 0

    for idx, img_path in enumerate(image_files, 1):
        rel_path = img_path.relative_to(input_p) if input_p.is_dir() else Path(img_path.name)
        rel_file = Path(rel_path.name)
        out_dir = (
            os.path.join(output_dir, str(rel_path.parent))
            if str(rel_path.parent) != "."
            else output_dir
        )

        if incremental_skip_done and outputs_exist_for_skip(out_dir, rel_file):
            skipped += 1
            print(f"[{idx}/{len(image_files)}] 增量跳过: {rel_path}")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[{idx}/{len(image_files)}] 无法读取，跳过: {img_path}")
            continue

        h_img, w_img = img.shape[:2]
        rows = segmenter.predict(
            img,
            clip_size=clip_size,
            overlap_size=overlap_size,
            padding=True,
            pad_full_image_to_square=pad_full_image_to_square,
            min_size=min_instance_size,
            max_size=None,
            imgsz=infer_imgsz,
            nms_iou=detect_nms_iou,
            max_det=detect_max_det,
            nms_agnostic=detect_nms_agnostic,
            debug=predict_debug,
            debug_clip=debug_clip,
        )

        # 统一字段名
        for r in rows:
            r["cls_name"] = str(r.get("class_name", r.get("cls_name", "")) or "")
            x1, y1, x2, y2 = bbox_from_row(r)
            r["x1"], r["y1"], r["x2"], r["y2"] = x1, y1, x2, y2

        results_visible = list(rows)

        src_xml = img_path.with_suffix(".xml")
        gts = None
        if enable_eval_with_xml and src_xml.is_file():
            try:
                gts = parse_voc_objects(str(src_xml))
            except Exception as e:
                logging.warning("读取标注 xml 失败 %s: %s", src_xml, e)
                gts = None

        if gts:
            stat_total["img_with_xml"] += 1
            matches, matched_p, matched_g = match_pred_gt_polygon(
                results_visible,
                gts,
                img_w=w_img,
                img_h=h_img,
                iou_threshold=val_geom_threshold,
                use_mask_iou=val_match_use_mask_iou,
            )
            stat_total["geom_pairs"] += len(matches)
            pred_to_gt = {pi: gj for pi, gj, _ in matches}

            for gj, g in enumerate(gts):
                gn = normalize_class_name(str(g.get("name", "")), CLASS_MERGE_TO_GROUPS)
                stat_by_cls[gn]["gt"] += 1
            for pi, p in enumerate(results_visible):
                pn = normalize_class_name(str(p.get("cls_name", "")), CLASS_MERGE_TO_GROUPS)
                stat_by_cls[pn]["pred"] += 1

            for pi, p in enumerate(results_visible):
                pn = normalize_class_name(str(p.get("cls_name", "")), CLASS_MERGE_TO_GROUPS)
                if pi not in matched_p:
                    stat_total["fp"] += 1
                    stat_by_cls[pn]["fp"] += 1
                else:
                    gj = pred_to_gt[pi]
                    gt_name = str(gts[gj].get("name", ""))
                    gn = normalize_class_name(gt_name, CLASS_MERGE_TO_GROUPS)
                    if is_class_match(str(p.get("cls_name", "")), gt_name, CLASS_MERGE_TO_GROUPS):
                        stat_total["tp"] += 1
                        stat_by_cls[gn]["tp"] += 1
                    else:
                        stat_total["cls_err"] += 1
                        stat_total["fp"] += 1
                        stat_by_cls[gn]["cls_err"] += 1
                        stat_by_cls[pn]["fp"] += 1

            for gj in range(len(gts)):
                if gj not in matched_g:
                    stat_total["fn"] += 1
                    gn = normalize_class_name(str(gts[gj].get("name", "")), CLASS_MERGE_TO_GROUPS)
                    stat_by_cls[gn]["fn"] += 1

        _save_image_and_xml(
            img,
            results_visible,
            out_dir=out_dir,
            rel_path=rel_path,
            draw_polygons=enable_draw_polygons,
            draw_bbox=enable_draw_bbox,
            draw_center_point_and_label=enable_draw_center_point_and_label,
            polygon_alpha=polygon_fill_alpha,
        )

        if gts:
            print(
                f"[{idx}/{len(image_files)}] {rel_path} "
                f"pred={len(results_visible)} gt={len(gts)} matched={stat_total['geom_pairs']}"
            )
        else:
            print(
                f"[{idx}/{len(image_files)}] {rel_path} pred={len(results_visible)} (无标注 xml)"
            )

    print(f"\n完成。输出目录: {output_dir}")
    if skipped:
        print(f"增量跳过 {skipped} 张")
    if stat_total["img_with_xml"] > 0:
        _print_stat_summary("分割验证汇总", stat_total)
        if stat_by_cls:
            print("\n按类统计（归一名）:")
            for cls_name in sorted(stat_by_cls.keys()):
                s = stat_by_cls[cls_name]
                print(
                    f"  {cls_name}: gt={s['gt']} pred={s['pred']} "
                    f"tp={s['tp']} fp={s['fp']} fn={s['fn']} cls_err={s['cls_err']}"
                )
