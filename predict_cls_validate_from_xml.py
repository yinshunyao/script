#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Detail  : 只调用分类模型做 VOC(xml) 裁剪验证：
#            - 输入目录包含图片与同名 Pascal VOC xml
#            - 解析 xml 的 bndbox，从图片裁剪目标
#            - 可选：配置分割模型时，在 bbox 区域内先分割再按 polygon 抠图送分类
#            - 调用分类模型（YOLO classification）
#            - 比对预测类别与标签是否一致（中文标注 / 拼音类名自动映射，口径同 predict_all 校验）
#            - 按类别统计（含 predict_all 同款一级/二级重点关注 top1/top2），并导出 eval_metrics
#            - 导出混淆矩阵与「易混淆类别对」排行表（按行内混淆比例排序）
#            - 可选落盘：误分类裁剪、分类正确裁剪（均含置信度于文件名）
#            - 默认运行前清空 output_dir，避免与上次结果混合
import csv
import json
import logging
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np

_FILE = Path(__file__).resolve()
_ROOT = _FILE.parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from script.predict.model_cls import ModelCls
from script.predict.model_seg import ModelSegmenter
from script.predict_seg_lib import crop_instance_bgr_from_polygon, resolve_cls_crop_background
from script.predict_size_validate_lib import (
    _export_overall_summary_csv,
    _export_stat_by_cls_csv as export_eval_stat_by_cls_csv,
    _print_overall_stat_summary,
    _print_stat_by_cls as print_eval_stat_by_cls,
    build_eval_class_display_index,
    build_eval_focus_set,
    is_class_match,
    load_eval_label_alias_map,
    merge_stat_by_cls,
    normalize_class_name,
    parse_pascal_voc_objects,
    sum_stat_by_cls_focus,
)
from script.config_paths import DEFAULT_INSECT_ALG_ALL_JSON
from script.predict_all import load_insect_alg_all


def _load_alg_config_json(path: str | Path | None) -> dict | None:
    if not path:
        return None
    try:
        return load_insect_alg_all(path)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        logging.warning("读取 insect_alg_all 失败 %s: %s", path, e)
        return None


def _build_eval_class_merge(
    base: dict[str, list[str]] | None,
    *,
    insect_wildcard: bool = True,
) -> dict[str, list[str]] | None:
    """评估用类别合并表（与 ``predict_all.build_eval_class_merge`` 口径一致）。"""
    if base is None and not insect_wildcard:
        return None
    out: dict[str, list[str]] = dict(base or {})
    if insect_wildcard:
        aliases = [str(a).strip() for a in (out.get("insect") or []) if str(a).strip()]
        if "*" not in aliases:
            aliases.append("*")
        out["insect"] = aliases
    return out or None


def _collect_images(input_path: str) -> tuple[Path, list[Path]]:
    pic_ext = {".jpg", ".jpeg", ".png"}
    input_p = Path(input_path)
    if input_p.is_file():
        image_files = [input_p] if input_p.suffix.lower() in pic_ext else []
    elif input_p.is_dir():
        image_files = sorted(
            p for p in input_p.rglob("*") if p.is_file() and p.suffix.lower() in pic_ext
        )
    else:
        image_files = []
    return input_p, image_files


def _safe_crop_xyxy(img_bgr, x1: int, y1: int, x2: int, y2: int):
    h, w = img_bgr.shape[:2]
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w))
    y2 = max(0, min(int(y2), h))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    crop = img_bgr[y1:y2, x1:x2]
    if crop is None or crop.size == 0:
        return None
    return crop


def _expand_xyxy_pad(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    pad_ratio: float,
    img_w: int,
    img_h: int,
) -> tuple[int, int, int, int]:
    """按外接框长边比例外扩，并裁剪到图像范围内。"""
    bw = max(1, int(x2) - int(x1))
    bh = max(1, int(y2) - int(y1))
    pad = int(round(max(bw, bh) * float(pad_ratio)))
    nx1 = max(0, int(x1) - pad)
    ny1 = max(0, int(y1) - pad)
    nx2 = min(int(img_w), int(x2) + pad)
    ny2 = min(int(img_h), int(y2) + pad)
    if nx2 <= nx1:
        nx2 = min(img_w, nx1 + 1)
    if ny2 <= ny1:
        ny2 = min(img_h, ny1 + 1)
    return nx1, ny1, nx2, ny2


def _seg_det_area(d: dict) -> float:
    """分割实例面积：优先 polygon，否则用外接框。"""
    poly = d.get("polygon")
    if poly and len(poly) >= 3:
        try:
            pts = np.asarray(poly, dtype=np.float32).reshape(-1, 1, 2)
            area = float(cv2.contourArea(pts))
            if area > 0:
                return area
        except (TypeError, ValueError, cv2.error):
            pass
    try:
        x1, y1, x2, y2 = (
            int(d["x1"]),
            int(d["y1"]),
            int(d["x2"]),
            int(d["y2"]),
        )
    except (KeyError, TypeError, ValueError):
        return 0.0
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float((x2 - x1) * (y2 - y1))


def _pick_largest_seg_det(dets: list[dict]) -> dict | None:
    """在 bbox 外扩区域内的分割结果中，取面积最大的实例。"""
    best: dict | None = None
    best_area = 0.0
    best_conf = -1.0
    for d in dets or []:
        if d.get("filter"):
            continue
        area = _seg_det_area(d)
        if area <= 0:
            continue
        conf = float(d.get("conf", 0.0) or 0.0)
        if area > best_area or (area == best_area and conf > best_conf):
            best_area = area
            best_conf = conf
            best = d
    return best


def _crop_for_cls_from_xml_bbox(
    img_bgr,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    *,
    segmenter: ModelSegmenter | None = None,
    seg_bbox_pad_ratio: float = 0.1,
    seg_polygon_pad_ratio: float = 0.05,
    seg_crop_background: tuple[int, int, int] | None = None,
    seg_imgsz: int = 0,
    seg_nms_iou: float | None = None,
) -> tuple[np.ndarray | None, bool]:
    """
    从 xml bbox 得到送分类的裁剪图。

    - ``segmenter is None``：矩形 bbox 裁剪（原行为）。
    - 已配置分割模型：在 bbox 外扩区域内分割，按 polygon 抠图；失败时回退 bbox 裁剪。

    返回 ``(crop_bgr, seg_refined)``，``seg_refined`` 表示是否采用了分割 polygon。
    """
    bbox_crop = _safe_crop_xyxy(img_bgr, x1, y1, x2, y2)
    if segmenter is None:
        return bbox_crop, False
    if bbox_crop is None:
        return None, False

    h_img, w_img = img_bgr.shape[:2]
    px1, py1, px2, py2 = _expand_xyxy_pad(
        x1, y1, x2, y2, seg_bbox_pad_ratio, w_img, h_img
    )
    region = _safe_crop_xyxy(img_bgr, px1, py1, px2, py2)
    if region is None:
        return bbox_crop, False

    rh, rw = region.shape[:2]
    predict_kwargs: dict = {
        "clip_size": max(rw, rh) + 64,
        "overlap_size": 1,
        "padding": True,
    }
    if seg_imgsz > 0:
        predict_kwargs["imgsz"] = int(seg_imgsz)
    if seg_nms_iou is not None:
        predict_kwargs["nms_iou"] = float(seg_nms_iou)

    try:
        dets = segmenter.predict(region, **predict_kwargs)
    except Exception as e:
        logging.warning("bbox 区域分割失败，回退矩形裁剪: %s", e)
        return bbox_crop, False

    best = _pick_largest_seg_det(dets)
    if best is None:
        return bbox_crop, False

    poly = list(best.get("polygon") or [])
    if len(poly) < 3:
        return bbox_crop, False

    refined = crop_instance_bgr_from_polygon(
        region,
        poly,
        pad_ratio=seg_polygon_pad_ratio,
        background_bgr=seg_crop_background,
    )
    if refined is None or refined.size == 0:
        return bbox_crop, False
    return refined, True


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _clear_run_output_dir(out_dir: str) -> None:
    """若输出目录已存在则整目录删除，避免本次结果与上次裁剪图、CSV 混合。"""
    p = Path(out_dir).expanduser().resolve()
    if not p.exists():
        return
    if not p.is_dir():
        logging.warning("输出路径已存在且不是目录，跳过清理: %s", p)
        return
    shutil.rmtree(p)
    logging.info("已清理上次输出目录: %s", p)


def _crop_export_stem(rel_path: Path, obj_index: int, pred: dict | None) -> str:
    """导出裁剪小图文件名：原图 stem + 目标序号 + 模型置信度。"""
    conf = float((pred or {}).get("conf", 0.0) or 0.0)
    return f"{rel_path.stem}__obj{obj_index:03d}__conf{conf:.3f}.jpg"


_FS_UNSAFE_SEGMENT_CHARS = r'\/:*?"<>|' + "\x00"


def _fs_safe_segment(name: str, *, max_len: int = 160) -> str:
    """目录名片段：去掉路径分隔符与非法字符，避免无法创建目录。"""
    s = str(name or "").strip()
    for ch in _FS_UNSAFE_SEGMENT_CHARS:
        s = s.replace(ch, "_")
    s = s.strip(" .")
    if not s:
        s = "_unknown"
    if len(s) > max_len:
        s = s[:max_len]
    return s


def _imwrite_bgr(out_path: str, img_bgr) -> bool:
    """
    写入 BGR 裁剪图。OpenCV 在部分环境下对含中文等非 ASCII 路径会 cv2.imwrite 静默失败；
    失败时用 imencode + 二进制写入（与路径编码无关）。
    """
    if img_bgr is None or getattr(img_bgr, "size", 0) == 0:
        return False
    out_path = str(out_path)
    if cv2.imwrite(out_path, img_bgr):
        return True
    ext = (Path(out_path).suffix or ".jpg").lower()
    if ext not in (".jpg", ".jpeg", ".jp2", ".png", ".bmp", ".tif", ".tiff", ".webp"):
        ext = ".jpg"
    ok_buf, buf = cv2.imencode(ext, img_bgr)
    if not ok_buf or buf is None:
        logging.warning("裁剪图 imencode 失败: %s", out_path)
        return False
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        p.write_bytes(buf.tobytes())
    except OSError as e:
        logging.warning("裁剪图写入失败 %s: %s", out_path, e)
        return False
    return True


def _export_confusion_csv(out_dir: str, cm: dict[tuple[str, str], int]) -> None:
    _ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "confusion_matrix.csv")
    labels = sorted({a for a, _ in cm.keys()} | {b for _, b in cm.keys()})
    idx = {lab: i for i, lab in enumerate(labels)}
    mat = [[0] * len(labels) for _ in range(len(labels))]
    for (a, b), c in cm.items():
        if a not in idx or b not in idx:
            continue
        mat[idx[a]][idx[b]] += int(c)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gt\\pred"] + labels)
        for i, lab in enumerate(labels):
            w.writerow([lab] + [str(mat[i][j]) for j in range(len(labels))])


def _confusion_row_totals(cm: dict[tuple[str, str], int]) -> dict[str, int]:
    """每个 gt 类别在混淆矩阵中的行合计（该 gt 下的样本总数）。"""
    row_sum: dict[str, int] = defaultdict(int)
    for (gt, _pred), c in cm.items():
        row_sum[str(gt)] += int(c)
    return dict(row_sum)


def _export_confusion_pairs_ranked_csv(out_dir: str, cm: dict[tuple[str, str], int]) -> str | None:
    """
    从混淆矩阵提取 gt!=pred 的误分类对，按「行内混淆比例」从高到低排序后落盘。

    行内混淆比例 confusion_rate_in_gt_row = count(gt,pred) / sum_pred count(gt,*)，
    即：在真实为 gt 的样本中，被判成 pred 的比例。
    """
    _ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "confusion_pairs_ranked.csv")
    if not cm:
        return None
    row_totals = _confusion_row_totals(cm)
    total_samples = sum(row_totals.values())
    rows_out: list[dict[str, object]] = []
    for (gt, pred), c in cm.items():
        gt_s, pred_s = str(gt), str(pred)
        if gt_s == pred_s:
            continue
        cnt = int(c)
        if cnt <= 0:
            continue
        rs = int(row_totals.get(gt_s, 0))
        rate_row = (float(cnt) / float(rs)) if rs > 0 else 0.0
        rate_all = (float(cnt) / float(total_samples)) if total_samples > 0 else 0.0
        rows_out.append(
            {
                "gt": gt_s,
                "pred": pred_s,
                "count": cnt,
                "gt_row_total": rs,
                "confusion_rate_in_gt_row": round(rate_row, 6),
                "share_of_all_objects": round(rate_all, 6),
            }
        )
    rows_out.sort(
        key=lambda r: (
            -float(r["confusion_rate_in_gt_row"]),  # 行内混淆比例：高 -> 低
            -int(r["count"]),
            str(r["gt"]),
            str(r["pred"]),
        )
    )
    headers = [
        "gt",
        "pred",
        "count",
        "gt_row_total",
        "confusion_rate_in_gt_row",
        "share_of_all_objects",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(rows_out)
    return out_path


def _print_confusion_pairs_top(
    cm: dict[tuple[str, str], int],
    *,
    title: str,
    top_n: int = 20,
) -> None:
    """终端打印易混淆类别对（与 confusion_pairs_ranked.csv 同序，仅前若干行）。"""
    if not cm or top_n <= 0:
        return
    row_totals = _confusion_row_totals(cm)
    pairs: list[tuple[float, int, str, str]] = []
    for (gt, pred), c in cm.items():
        gt_s, pred_s = str(gt), str(pred)
        if gt_s == pred_s:
            continue
        cnt = int(c)
        if cnt <= 0:
            continue
        rs = int(row_totals.get(gt_s, 0))
        rate_row = (float(cnt) / float(rs)) if rs > 0 else 0.0
        pairs.append((rate_row, cnt, gt_s, pred_s))
    pairs.sort(key=lambda x: (-x[0], -x[1], x[2], x[3]))
    print(f"{title}（前 {min(top_n, len(pairs))} 条，按行内混淆比例从高到低）:")
    print(f"{'gt':<28} {'pred':<28} {'个数':>8} {'行合计':>8} {'行内比例':>10}")
    for rate_row, cnt, gt_s, pred_s in pairs[:top_n]:
        rs = int(row_totals.get(gt_s, 0))
        print(
            f"{gt_s:<28} {pred_s:<28} {cnt:>8} {rs:>8} {rate_row*100:>9.2f}%"
        )


if __name__ == "__main__":
    # /Users/shunyaoyin/miniconda310/miniconda3/envs/yolo11/bin/python3 /Users/shunyaoyin/Documents/code/ai-company/insect/script/predict_cls_validate_from_xml.py
    logging.basicConfig(level=logging.INFO)
    from script.predict_all import (
        compute_competition_counting_summary,
        print_competition_counting_summary,
        resolve_competition_counting_focus,
        resolve_validation_focus_config,
    )

    # ----------------------- 需要你改的参数 -----------------------
    # 输入：图片 + 同名 xml（Pascal VOC）
    input_path = "/Volumes/shunyao-h1/训练数据/测试集/北京设备全标注"
    # 分类模型（Ultralytics YOLO classification）
    # cls_model_path = "/Users/shunyaoyin/Documents/code/ai-company/insect/doc/测试结果/大虫训练总结/20260424-all-large/best.pt"
    cls_model_path = "/Volumes/shunyao-h1/models-test/cls-v3.5-split/cls-3.5.2.pt"
    # 输出
    output_dir = input_path + "-c3.5.2"
    # 只打印关注类别（可选）：仅影响“按类别统计”的打印，不影响统计与 CSV/混淆矩阵落盘
    # 例：FOCUS_CLASS_NAMES = ("bazidilaohu", "caodiming")
    # FOCUS_CLASS_NAMES: tuple[str, ...] | None = (
    #     # 一级
    #     "caodiming", "daozongjuanyeming", "yumiming", "caoditanyee",
    #     "erhuaming",  "laoshinianchong", "laoshinianchong", "feihuang",
    #     "daofeishi", "hefeishi"
    # )
    FOCUS_CLASS_NAMES: tuple[str, ...] | None = None

    # 按类统计排序与导出（口径同 predict_all 校验）
    SORT_STAT_BY_ACC = True
    STANDARD_EVAL_SUBDIR = "eval_metrics"

    # 类别归一（可选）：用于“标签一致性判断”和最终统计的类名归一
    # 说明：和 `script/predict_size_validate_lib.py` 中 merge 语义一样，key 为归一名，values 为别名列表。
    CLASS_MERGE_TO_GROUPS: dict[str, list[str]] | None = None
    # 中文/拼音自动映射（默认开启）：从 insect_info + insect_alg_all 构建，
    # 使 xml 中文 <name> 与模型拼音 class_name 可对齐（口径同 predict_all 校验）。
    INSECT_ALG_ALL_JSON: str | Path | None = DEFAULT_INSECT_ALG_ALL_JSON
    EVAL_USE_CN_PINYIN_ALIAS = True
    EVAL_INSECT_WILDCARD = True

    # 可选分割精炼：None 时保持原流程（xml bbox 矩形裁剪后直接分类）
    # 配置路径时，在 bbox 外扩区域内先跑分割，再按 polygon 抠图送分类
    SEG_MODEL_PATH: str | None = "/Volumes/shunyao-h1/models-test/seg-v3.8/seg-3.8.3.pt"
    # SEG_MODEL_PATH = "/path/to/seg/best.pt"
    seg_conf_thresh: float = 0.25
    seg_imgsz: int = 960
    seg_nms_iou: float | None = 0.5
    seg_bbox_pad_ratio: float = 0.1
    seg_polygon_pad_ratio: float = 0.05
    seg_crop_background: str | list[int] | None = "white"

    # 分类预处理：是否将裁剪补成白底正方形（训练若是白边正方形，建议 True）
    cls_pad_square: bool = True
    # 可选：灰度+CLAHE+Otsu，再扩三通道（只有你训练时用了类似流程才建议打开）
    cls_gray_binarize: bool = False
    # 是否保存误分类裁剪小图（会按 gt/pred 分目录存，便于快速排查）
    save_misclassified_crops: bool = True
    # 是否保存分类正确的裁剪小图（按归一化类别 class= 分目录，文件名含置信度）
    save_correct_crops: bool = True
    # 运行前是否清空整个 output_dir（删除上次验证产物，避免混合）
    clean_output_before_run: bool = True

    # ------------------------------------------------------------

    input_p, image_files = _collect_images(input_path)
    print(f"共找到 {len(image_files)} 张图片")
    if clean_output_before_run:
        _clear_run_output_dir(output_dir)
    _ensure_dir(output_dir)

    label_alias_map: dict[str, str] | None = (
        load_eval_label_alias_map(alg_config_path=INSECT_ALG_ALL_JSON)
        if EVAL_USE_CN_PINYIN_ALIAS
        else None
    )
    if label_alias_map:
        print(f"标签别名映射已加载，条目数={len(label_alias_map)}")
    class_merge_eval = _build_eval_class_merge(
        CLASS_MERGE_TO_GROUPS,
        insect_wildcard=EVAL_INSECT_WILDCARD,
    )
    alg_config = _load_alg_config_json(INSECT_ALG_ALL_JSON)
    validation_focus = resolve_validation_focus_config(alg_config)
    eval_class_display_index = build_eval_class_display_index(alg_config=alg_config)
    report_focus = (
        build_eval_focus_set(
            validation_focus.report_classes,
            merge=class_merge_eval,
            label_alias_map=label_alias_map,
        )
        if validation_focus.report_classes
        else None
    )
    top1_focus = (
        build_eval_focus_set(
            validation_focus.top1_classes,
            merge=class_merge_eval,
            label_alias_map=label_alias_map,
        )
        if validation_focus.top1_classes
        else frozenset()
    )
    top2_focus = (
        build_eval_focus_set(
            validation_focus.top2_classes,
            merge=class_merge_eval,
            label_alias_map=label_alias_map,
        )
        if validation_focus.top2_classes
        else frozenset()
    )
    top3_focus = (
        build_eval_focus_set(
            validation_focus.top3_classes,
            merge=class_merge_eval,
            label_alias_map=label_alias_map,
        )
        if validation_focus.top3_classes
        else frozenset()
    )
    competition_focus = resolve_competition_counting_focus(
        validation_focus,
        class_merge=class_merge_eval,
        label_alias_map=label_alias_map,
    )
    optional_focus = (
        build_eval_focus_set(
            FOCUS_CLASS_NAMES,
            merge=class_merge_eval,
            label_alias_map=label_alias_map,
        )
        if FOCUS_CLASS_NAMES
        else report_focus
    )

    classifier = ModelCls(
        model_path=cls_model_path,
        device=None,
        pad_square=cls_pad_square,
        gray_binarize=cls_gray_binarize,
    )

    segmenter: ModelSegmenter | None = None
    seg_mask_bg = resolve_cls_crop_background(seg_crop_background)
    seg_path = str(SEG_MODEL_PATH or "").strip()
    if seg_path:
        segmenter = ModelSegmenter(
            model_path=seg_path,
            conf_thresh=float(seg_conf_thresh),
            imgsz=int(seg_imgsz or 0),
            nms_iou=seg_nms_iou,
        )
        print(
            f"已启用分割精炼: {seg_path}  "
            f"imgsz={seg_imgsz}  bbox_pad={seg_bbox_pad_ratio}  "
            f"polygon_pad={seg_polygon_pad_ratio}"
        )
    else:
        print("未配置分割模型，使用 xml bbox 矩形裁剪（原流程）")

    stat_by_cls: dict[str, dict[str, int]] = {}

    def _inc(cls_norm: str, key: str, n: int = 1) -> None:
        cls_norm = str(cls_norm or "")
        if cls_norm not in stat_by_cls:
            stat_by_cls[cls_norm] = {
                "gt": 0,
                "pred": 0,
                "tp": 0,
                "fn": 0,
                "fp": 0,
                "cls_err": 0,
            }
        stat_by_cls[cls_norm][key] = int(stat_by_cls[cls_norm].get(key, 0)) + int(n)

    cm: defaultdict[tuple[str, str], int] = defaultdict(int)  # (gt_norm, pred_norm) -> count
    sum_gt = sum_tp = sum_fn = sum_fp = 0
    img_with_xml = 0
    obj_total = 0
    obj_skipped = 0
    obj_seg_refined = 0
    obj_seg_fallback_bbox = 0

    for idx, img_path in enumerate(image_files, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            logging.warning("[%s/%s] 无法读取图片，跳过: %s", idx, len(image_files), img_path)
            continue

        rel_path = img_path.relative_to(input_p) if input_p.is_dir() else Path(img_path.name)
        src_xml = img_path.with_suffix(".xml")
        if not src_xml.is_file():
            logging.info("[%s/%s] 无 xml，跳过: %s", idx, len(image_files), rel_path)
            continue

        try:
            gts = parse_pascal_voc_objects(str(src_xml))
        except (ET.ParseError, OSError, ValueError) as e:
            logging.warning("[%s/%s] 读取 xml 失败 %s: %s", idx, len(image_files), src_xml, e)
            continue

        img_with_xml += 1
        per_img_correct = per_img_wrong = per_img_total = 0

        for oi, g in enumerate(gts):
            gt_raw = str(g.get("name", "") or "").strip()
            if not gt_raw:
                obj_skipped += 1
                continue

            crop, seg_refined = _crop_for_cls_from_xml_bbox(
                img,
                g["x1"],
                g["y1"],
                g["x2"],
                g["y2"],
                segmenter=segmenter,
                seg_bbox_pad_ratio=seg_bbox_pad_ratio,
                seg_polygon_pad_ratio=seg_polygon_pad_ratio,
                seg_crop_background=seg_mask_bg,
                seg_imgsz=seg_imgsz,
                seg_nms_iou=seg_nms_iou,
            )
            if crop is None:
                obj_skipped += 1
                continue
            if segmenter is not None:
                if seg_refined:
                    obj_seg_refined += 1
                else:
                    obj_seg_fallback_bbox += 1

            obj_total += 1
            per_img_total += 1

            pred = classifier.predict(crop)
            pred_raw = str((pred or {}).get("class_name", "") or "").strip() if pred else ""
            if not pred_raw:
                pred_raw = "unknown"

            gt_norm = (
                normalize_class_name(
                    gt_raw,
                    class_merge_eval,
                    label_alias_map=label_alias_map,
                )
                or gt_raw
            )
            pred_norm = (
                normalize_class_name(
                    pred_raw,
                    class_merge_eval,
                    label_alias_map=label_alias_map,
                )
                or pred_raw
            )

            _inc(gt_norm, "gt", 1)
            _inc(pred_norm, "pred", 1)
            ok = bool(
                is_class_match(
                    pred_raw,
                    gt_raw,
                    class_merge_eval,
                    None,
                    label_alias_map=label_alias_map,
                )
            )
            if ok:
                _inc(gt_norm, "tp", 1)
                sum_tp += 1
                per_img_correct += 1
                if save_correct_crops:
                    stem = _crop_export_stem(rel_path, oi, pred)
                    out_ok = os.path.join(
                        output_dir,
                        "classified_crops",
                        f"class={_fs_safe_segment(gt_norm)}",
                    )
                    _ensure_dir(out_ok)
                    out_file = os.path.join(out_ok, stem)
                    if not _imwrite_bgr(out_file, crop):
                        logging.warning("未能写入正确预测裁剪图: %s", out_file)
            else:
                # 分类模型每个 GT 都会给出一个预测：
                # - 对 GT 类别来说是 FN（漏报/没报对）
                # - 对 Pred 类别来说是 FP（误报/报成了别的类）
                _inc(gt_norm, "fn", 1)
                _inc(pred_norm, "fp", 1)
                sum_fn += 1
                sum_fp += 1
                per_img_wrong += 1

            cm[(gt_norm, pred_norm)] += 1
            sum_gt += 1

            if save_misclassified_crops and (not ok):
                stem = _crop_export_stem(rel_path, oi, pred)

                # 原有索引：先按 GT，再按 Pred
                out_sub_gt_first = os.path.join(
                    output_dir,
                    "misclassified_crops",
                    f"gt={_fs_safe_segment(gt_norm)}",
                    f"pred={_fs_safe_segment(pred_norm)}",
                )
                _ensure_dir(out_sub_gt_first)
                out_mis = os.path.join(out_sub_gt_first, stem)
                if not _imwrite_bgr(out_mis, crop):
                    logging.warning("未能写入误分类裁剪图: %s", out_mis)

                # 新增反向索引：先按 Pred，再按 GT（便于从预测类别反查）
                out_sub_pred_first = os.path.join(
                    output_dir,
                    "misclassified_crops_by_pred",
                    f"pred={_fs_safe_segment(pred_norm)}",
                    f"gt={_fs_safe_segment(gt_norm)}",
                )
                _ensure_dir(out_sub_pred_first)
                out_mis2 = os.path.join(out_sub_pred_first, stem)
                if not _imwrite_bgr(out_mis2, crop):
                    logging.warning("未能写入误分类裁剪图: %s", out_mis2)

        print(
            f"[{idx}/{len(image_files)}] {rel_path}  objs={per_img_total}  "
            f"correct={per_img_correct}  wrong={per_img_wrong}"
        )

    print("======== 分类模型 VOC 裁剪验证 ========")
    seg_line = ""
    if segmenter is not None:
        seg_line = (
            f"  seg_polygon={obj_seg_refined}  seg_fallback_bbox={obj_seg_fallback_bbox}"
        )
    print(
        f"images_with_xml={img_with_xml}  obj_total={obj_total}  obj_skipped={obj_skipped}"
        f"{seg_line}"
    )

    stat_by_cls_merged = merge_stat_by_cls(
        dict(stat_by_cls),
        merge=class_merge_eval,
        label_alias_map=label_alias_map,
    )
    stat_total = {
        "tp": int(sum_tp),
        "fp": int(sum_fp),
        "fn": int(sum_fn),
        "cls_err": 0,
        "geom_pairs": int(sum_gt),
    }
    _print_overall_stat_summary("分类裁剪验证汇总", stat_total)
    print_eval_stat_by_cls(
        "按合并类统计",
        stat_by_cls_merged,
        sort_by_acc=SORT_STAT_BY_ACC,
        class_display_index=eval_class_display_index,
        focus=optional_focus,
    )
    _print_overall_stat_summary(
        "一级重点关注(top1)汇总",
        sum_stat_by_cls_focus(stat_by_cls_merged, top1_focus),
    )
    print_eval_stat_by_cls(
        "一级重点关注(top1)按类统计",
        stat_by_cls_merged,
        sort_by_acc=SORT_STAT_BY_ACC,
        class_display_index=eval_class_display_index,
        focus=top1_focus,
    )
    _print_overall_stat_summary(
        "二级重点关注(top2)汇总",
        sum_stat_by_cls_focus(stat_by_cls_merged, top2_focus),
    )
    print_eval_stat_by_cls(
        "二级重点关注(top2)按类统计",
        stat_by_cls_merged,
        sort_by_acc=SORT_STAT_BY_ACC,
        class_display_index=eval_class_display_index,
        focus=top2_focus,
    )
    _print_overall_stat_summary(
        "三级重点关注(top3)汇总",
        sum_stat_by_cls_focus(stat_by_cls_merged, top3_focus),
    )
    print_eval_stat_by_cls(
        "三级重点关注(top3)按类统计",
        stat_by_cls_merged,
        sort_by_acc=SORT_STAT_BY_ACC,
        class_display_index=eval_class_display_index,
        focus=top3_focus,
    )
    counting_summary = compute_competition_counting_summary(
        stat_by_cls_merged,
        competition_focus or frozenset(),
        run_model=validation_focus.run_model,
    )
    print_competition_counting_summary(
        counting_summary,
        class_display_index=eval_class_display_index,
    )

    eval_root = os.path.join(output_dir, STANDARD_EVAL_SUBDIR)
    _export_overall_summary_csv(eval_root, "all", stat_total)
    export_eval_stat_by_cls_csv(
        eval_root,
        "all",
        stat_by_cls_merged,
        sort_by_acc=SORT_STAT_BY_ACC,
        class_display_index=eval_class_display_index,
        focus=optional_focus,
    )
    _export_overall_summary_csv(
        eval_root,
        "top1",
        sum_stat_by_cls_focus(stat_by_cls_merged, top1_focus),
    )
    export_eval_stat_by_cls_csv(
        eval_root,
        "top1",
        stat_by_cls_merged,
        sort_by_acc=SORT_STAT_BY_ACC,
        class_display_index=eval_class_display_index,
        focus=top1_focus,
    )
    _export_overall_summary_csv(
        eval_root,
        "top2",
        sum_stat_by_cls_focus(stat_by_cls_merged, top2_focus),
    )
    export_eval_stat_by_cls_csv(
        eval_root,
        "top2",
        stat_by_cls_merged,
        sort_by_acc=SORT_STAT_BY_ACC,
        class_display_index=eval_class_display_index,
        focus=top2_focus,
    )
    _export_overall_summary_csv(
        eval_root,
        "top3",
        sum_stat_by_cls_focus(stat_by_cls_merged, top3_focus),
    )
    export_eval_stat_by_cls_csv(
        eval_root,
        "top3",
        stat_by_cls_merged,
        sort_by_acc=SORT_STAT_BY_ACC,
        class_display_index=eval_class_display_index,
        focus=top3_focus,
    )

    _export_confusion_csv(output_dir, dict(cm))
    pairs_path = _export_confusion_pairs_ranked_csv(output_dir, dict(cm))
    _print_confusion_pairs_top(dict(cm), title="易混淆类别对")
    print(f"输出目录: {output_dir}")
    print(f"评估统计目录: {eval_root} (all / top1 / top2 / top3)")
    if pairs_path:
        print(f"易混淆类别对排行表: {pairs_path}")

