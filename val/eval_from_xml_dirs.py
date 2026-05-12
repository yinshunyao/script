#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate inference XMLs against GT XMLs (Pascal VOC).

Use-case:
- You already have GT annotation XMLs in a directory
- You already have model inference output XMLs in another directory
- You want to compute basic VOC-style metrics (TP/FP/FN/cls_err) using the same
  matching + class-normalization rules as `script/predict_size_validate*.py`

This file is designed to:
- expose reusable functions for other modules to import
- be runnable directly (config via variables under __main__, no CLI by default)
"""

from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import cv2

from script.predict_size_validate import (
    draw_main_output_image,
    is_class_match,
    is_metric_ignored_other,
    match_pred_gt,
    match_pred_gt_ior,
    normalize_class_name,
    parse_pascal_voc_objects,
)


@dataclass(frozen=True)
class EvalConfig:
    geom_metric: str = "iou"  # "iou" | "ior"
    geom_threshold: float = 0.5
    # When > 0: boxes with diagonal pixel length smaller than this threshold are ignored (not compared).
    # Set to 0/None to disable.
    min_box_diag_px: float | None = None
    class_merge_to_groups: dict[str, list[str]] | None = None
    focus_classes: tuple[str, ...] | None = None
    ignore_missing_pred_xml: bool = True
    # Default off: save overlay images using the same palette as predict_size_validate.draw_main_output_image
    # (green=TP, gray=class mismatch, red=geom-unmatched pred, pink=missed GT).
    visualize_matching: bool = False
    # When visualize_matching is True: output root. If None and out_dir is set, uses out_dir / "matching_visualization".
    visualize_out_dir: str | Path | None = None


class _EvalOneCore(NamedTuple):
    stats: dict[str, int]
    by_cls: dict[str, dict[str, int]]
    gts: list[dict[str, Any]]
    preds: list[dict[str, Any]]
    matches: list[tuple[int, int, float]]
    matched_p: set[int]
    matched_g: set[int]


def _iter_xml_files(root_dir: str | Path) -> list[Path]:
    p = Path(root_dir)
    if not p.exists():
        return []
    if p.is_file():
        return [p] if p.suffix.lower() == ".xml" else []
    return sorted(x for x in p.rglob("*.xml") if x.is_file())


def _build_rel_map(root_dir: str | Path) -> dict[str, Path]:
    """
    Build mapping: relative_posix_path -> absolute_path
    Relative path is computed from root_dir, keeping subfolders.
    """
    root = Path(root_dir)
    out: dict[str, Path] = {}
    for f in _iter_xml_files(root):
        try:
            rel = f.relative_to(root).as_posix()
        except Exception:
            rel = f.name
        out[rel] = f
    return out


def _to_pred_rows(voc_objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Convert VOC object list to "pred row" format expected by matchers:
    {"cls_name": ..., "x1","y1","x2","y2"}
    """
    out: list[dict[str, Any]] = []
    for o in voc_objects or []:
        out.append(
            {
                "cls_name": str(o.get("name", "") or ""),
                "x1": int(o.get("x1", 0) or 0),
                "y1": int(o.get("y1", 0) or 0),
                "x2": int(o.get("x2", 0) or 0),
                "y2": int(o.get("y2", 0) or 0),
            }
        )
    return out


def _focus_set_from_cfg(
    focus_classes: tuple[str, ...] | None, class_merge_to_groups: dict[str, list[str]] | None
) -> frozenset[str]:
    if not focus_classes:
        return frozenset()
    items = [str(x).strip() for x in focus_classes if str(x).strip()]
    return frozenset(items)


def _get_wildcard_group_key(merge: dict[str, list[str]] | None) -> str | None:
    """
    Return the group key that contains wildcard '*' (e.g. {'insect': ['*']}).
    If multiple wildcard groups exist, return the first encountered key.
    """
    if not merge:
        return None
    for k, aliases in (merge or {}).items():
        for a in aliases or []:
            if str(a).strip() == "*":
                return str(k)
    return None


def _keep_name_by_focus(raw_name: str, focus: frozenset[str], merge: dict[str, list[str]] | None) -> bool:
    if not focus:
        return True
    raw = str(raw_name or "").strip()
    if not raw:
        return False
    wildcard_key = _get_wildcard_group_key(merge)
    # If focus includes the wildcard group key (e.g. "insect"), keep all non-empty names.
    # Note: "other" is filtered out earlier by is_metric_ignored_other().
    if wildcard_key and wildcard_key in focus:
        return True
    if raw in focus:
        return True
    norm = normalize_class_name(raw, merge)
    return bool(norm) and norm in focus


def _box_diag_px(x1: int, y1: int, x2: int, y2: int) -> float:
    dx = float(int(x2) - int(x1))
    dy = float(int(y2) - int(y1))
    return (dx * dx + dy * dy) ** 0.5


def _keep_box_by_diag(
    *,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    min_diag_px: float | None,
) -> bool:
    thr = float(min_diag_px or 0.0)
    if thr <= 0:
        return True
    return _box_diag_px(x1, y1, x2, y2) >= thr


def _inc(stat_by_cls: dict[str, dict[str, int]], cls_norm: str, key: str, n: int = 1) -> None:
    cls_norm = str(cls_norm or "")
    if cls_norm not in stat_by_cls:
        stat_by_cls[cls_norm] = {"gt": 0, "pred": 0, "tp": 0, "fp": 0, "fn": 0, "cls_err": 0}
    stat_by_cls[cls_norm][key] = int(stat_by_cls[cls_norm].get(key, 0)) + int(n)


def _resolve_image_next_to_xml(xml_path: str | Path) -> Path | None:
    """同一目录下与 XML 同名的常见图片扩展名。"""
    xp = Path(xml_path)
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".bmp", ".BMP"):
        cand = xp.with_suffix(ext)
        if cand.is_file():
            return cand
    return None


def _pred_boxes_to_draw_rows(preds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """与 draw_main_output_image / _visible_index 兼容的 pred 行（identity 与 results_visible 共用）。"""
    rows: list[dict[str, Any]] = []
    for p in preds or []:
        rows.append(
            {
                "cls_name": str(p.get("cls_name", "") or ""),
                "x1": int(p.get("x1", 0) or 0),
                "y1": int(p.get("y1", 0) or 0),
                "x2": int(p.get("x2", 0) or 0),
                "y2": int(p.get("y2", 0) or 0),
                "class_name": "",
                "cls_conf": 0.0,
                "conf": 0.0,
                "filter": False,
            }
        )
    return rows


def eval_one_image_core(
    *,
    gt_xml_path: str | Path,
    pred_xml_path: str | Path | None,
    config: EvalConfig,
) -> _EvalOneCore:
    """
    单图评估核心：统计 + GT/Pred 过滤后列表及几何匹配结果（供 evaluate_one_image 与可视化共用）。
    """
    s = {"tp": 0, "fp": 0, "fn": 0, "cls_err": 0, "geom_pairs": 0, "img_with_xml": 1}
    by_cls: dict[str, dict[str, int]] = {}

    merge = config.class_merge_to_groups
    focus = _focus_set_from_cfg(config.focus_classes, merge)
    min_diag_px = config.min_box_diag_px

    gts_all = parse_pascal_voc_objects(str(gt_xml_path))
    gts = [
        g
        for g in (gts_all or [])
        if (not is_metric_ignored_other(str(g.get("name", "") or ""), merge))
        and _keep_name_by_focus(str(g.get("name", "") or ""), focus, merge)
        and _keep_box_by_diag(
            x1=int(g.get("x1", 0) or 0),
            y1=int(g.get("y1", 0) or 0),
            x2=int(g.get("x2", 0) or 0),
            y2=int(g.get("y2", 0) or 0),
            min_diag_px=min_diag_px,
        )
    ]

    preds: list[dict[str, Any]] = []
    if pred_xml_path is not None and Path(pred_xml_path).is_file():
        preds_all = parse_pascal_voc_objects(str(pred_xml_path))
        preds = [
            p
            for p in _to_pred_rows(preds_all)
            if (not is_metric_ignored_other(str(p.get("cls_name", "") or ""), merge))
            and _keep_name_by_focus(str(p.get("cls_name", "") or ""), focus, merge)
            and _keep_box_by_diag(
                x1=int(p.get("x1", 0) or 0),
                y1=int(p.get("y1", 0) or 0),
                x2=int(p.get("x2", 0) or 0),
                y2=int(p.get("y2", 0) or 0),
                min_diag_px=min_diag_px,
            )
        ]

    for g in gts:
        _inc(by_cls, normalize_class_name(str(g.get("name", "") or ""), merge), "gt", 1)
    for p in preds:
        _inc(by_cls, normalize_class_name(str(p.get("cls_name", "") or ""), merge), "pred", 1)

    matches: list[tuple[int, int, float]] = []
    matched_p: set[int] = set()
    matched_g: set[int] = set()

    if not gts:
        s["fp"] = len(preds)
        for p in preds:
            _inc(by_cls, normalize_class_name(str(p.get("cls_name", "") or ""), merge), "fp", 1)
        return _EvalOneCore(s, by_cls, gts, preds, matches, matched_p, matched_g)

    metric = str(config.geom_metric or "iou").lower().strip()
    thr = float(config.geom_threshold)
    if metric == "ior":
        matches, matched_p, matched_g = match_pred_gt_ior(preds, gts, thr)
    else:
        matches, matched_p, matched_g = match_pred_gt(preds, gts, thr, metric="iou")

    s["geom_pairs"] = int(len(matches or []))
    pred_to_gt = {int(pi): int(gj) for pi, gj, _sc in (matches or [])}

    for pi, p in enumerate(preds):
        if pi not in matched_p:
            s["fp"] += 1
            _inc(by_cls, normalize_class_name(str(p.get("cls_name", "") or ""), merge), "fp", 1)
            continue

        gj = pred_to_gt[int(pi)]
        gt = gts[int(gj)]
        pred_raw = str(p.get("cls_name", "") or "")
        gt_raw = str(gt.get("name", "") or "")
        pred_norm = normalize_class_name(pred_raw, merge)
        gt_norm = normalize_class_name(gt_raw, merge)

        if is_class_match(pred_raw, gt_raw, merge):
            s["tp"] += 1
            _inc(by_cls, gt_norm, "tp", 1)
        else:
            s["cls_err"] += 1
            s["fp"] += 1
            _inc(by_cls, gt_norm, "cls_err", 1)
            _inc(by_cls, pred_norm, "fp", 1)

    fn = int(len(gts) - len(matched_g))
    s["fn"] = fn
    if fn > 0:
        for gj, gt in enumerate(gts):
            if gj not in matched_g:
                _inc(by_cls, normalize_class_name(str(gt.get("name", "") or ""), merge), "fn", 1)

    return _EvalOneCore(s, by_cls, gts, preds, matches, matched_p, matched_g)


def evaluate_one_image(
    *,
    gt_xml_path: str | Path,
    pred_xml_path: str | Path | None,
    config: EvalConfig,
) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    """
    Evaluate one GT XML against one Pred XML.

    Returns:
    - aggregate stats dict: tp/fp/fn/cls_err/geom_pairs/img_with_xml
    - per-class stats dict (normalized class name)
    """
    core = eval_one_image_core(gt_xml_path=gt_xml_path, pred_xml_path=pred_xml_path, config=config)
    return core.stats, core.by_cls


def _merge_stat_by_cls(dst: dict[str, dict[str, int]], src: dict[str, dict[str, int]]) -> None:
    for cls_name, s in (src or {}).items():
        if cls_name not in dst:
            dst[cls_name] = {"gt": 0, "pred": 0, "tp": 0, "fp": 0, "fn": 0, "cls_err": 0}
        for k in ("gt", "pred", "tp", "fp", "fn", "cls_err"):
            dst[cls_name][k] = int(dst[cls_name].get(k, 0)) + int(s.get(k, 0))


def save_eval_visualization(
    *,
    image_path: str | Path,
    out_image_path: str | Path,
    gts: list[dict[str, Any]],
    preds: list[dict[str, Any]],
    matches: list[tuple[int, int, float]],
    matched_p: set[int],
    class_merge_to_groups: dict[str, list[str]] | None,
) -> bool:
    """
    使用 predict_size_validate.draw_main_output_image 的 val_xml 配色：
    绿=正确，灰=类型错误，红=预测框与 GT 几何未匹配（脚本内题注为「误报」），粉=漏报 GT。
    """
    ip = Path(image_path)
    op = Path(out_image_path)
    op.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.imread(str(ip))
    if bgr is None:
        logging.warning("无法读取图片，跳过可视化: %s", ip)
        return False

    pred_rows = _pred_boxes_to_draw_rows(preds)
    drawn = draw_main_output_image(
        bgr,
        pred_rows,
        clip_size=0,
        overlap_size=0,
        predict_debug=False,
        label_mode="detailed",
        val_xml_mode=True,
        results_visible=pred_rows,
        gts=gts,
        matches=matches,
        matched_p=matched_p,
        merge=class_merge_to_groups,
    )
    cv2.imwrite(str(op), drawn)
    return True


def evaluate_xml_dirs(
    *,
    gt_xml_dir: str | Path,
    pred_xml_dir: str | Path,
    config: EvalConfig | None = None,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    Evaluate a directory of GT XMLs against a directory of prediction XMLs.

    Pairing rule:
    - use relative path under gt_xml_dir, lookup the same relative path under pred_xml_dir.

    Returns a summary dict suitable for json serialization.
    """
    cfg = config or EvalConfig()
    gt_root = Path(gt_xml_dir)
    pred_root = Path(pred_xml_dir)

    gt_map = _build_rel_map(gt_root)
    pred_map = _build_rel_map(pred_root)

    viz_root: Path | None = None
    if cfg.visualize_matching:
        viz_root = Path(cfg.visualize_out_dir) if cfg.visualize_out_dir else None
        if viz_root is None and out_dir is not None:
            viz_root = Path(out_dir) / "matching_visualization"
        if viz_root is None:
            logging.warning("visualize_matching=True 但未设置 visualize_out_dir 且 out_dir 为空，跳过作图")

    total = {"tp": 0, "fp": 0, "fn": 0, "cls_err": 0, "geom_pairs": 0, "img_with_xml": 0}
    stat_by_cls: dict[str, dict[str, int]] = {}

    missing_pred: list[str] = []
    parse_errors: list[dict[str, str]] = []
    viz_written = 0
    viz_skipped_no_image = 0

    for rel, gt_path in gt_map.items():
        pred_path = pred_map.get(rel)
        if pred_path is None:
            missing_pred.append(rel)
            if not cfg.ignore_missing_pred_xml:
                # treat as empty pred; still evaluate (all GT become FN)
                pred_path = None
            else:
                pred_path = None

        try:
            core = eval_one_image_core(gt_xml_path=gt_path, pred_xml_path=pred_path, config=cfg)
            s, by_cls = core.stats, core.by_cls
        except Exception as e:
            parse_errors.append({"rel": rel, "gt": str(gt_path), "pred": str(pred_path or ""), "error": str(e)})
            continue

        total["img_with_xml"] += 1
        for k in ("tp", "fp", "fn", "cls_err", "geom_pairs"):
            total[k] += int(s.get(k, 0))
        _merge_stat_by_cls(stat_by_cls, by_cls)

        if viz_root is not None:
            img_src = _resolve_image_next_to_xml(gt_path)
            if img_src is None:
                viz_skipped_no_image += 1
                logging.debug("GT XML 旁无同名图片，跳过可视化: %s", gt_path)
            else:
                dst = viz_root / Path(rel).with_suffix(".jpg")
                if save_eval_visualization(
                    image_path=img_src,
                    out_image_path=dst,
                    gts=core.gts,
                    preds=core.preds,
                    matches=core.matches,
                    matched_p=core.matched_p,
                    class_merge_to_groups=cfg.class_merge_to_groups,
                ):
                    viz_written += 1

    summary = {
        "meta": {
            "gt_xml_dir": str(gt_root),
            "pred_xml_dir": str(pred_root),
            "geom_metric": cfg.geom_metric,
            "geom_threshold": cfg.geom_threshold,
            "min_box_diag_px": (float(cfg.min_box_diag_px) if (cfg.min_box_diag_px or 0) > 0 else 0.0),
            "focus_classes": list(cfg.focus_classes or []),
            "ignore_missing_pred_xml": bool(cfg.ignore_missing_pred_xml),
            "visualize_matching": bool(cfg.visualize_matching),
            "visualize_out_dir": (str(viz_root.resolve()) if viz_root is not None else ""),
            "visualization_saved_count": viz_written,
            "visualization_skipped_no_image_count": viz_skipped_no_image,
        },
        "counts": total,
        "missing_pred_xml": missing_pred,
        "parse_errors": parse_errors,
        "stat_by_class": stat_by_cls,
    }

    if out_dir is not None:
        out_p = Path(out_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        _write_reports(out_p, summary)

    return summary


def _write_reports(out_dir: Path, summary: dict[str, Any]) -> None:
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # overall csv
    counts = summary.get("counts", {}) or {}
    with open(out_dir / "overall_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["tp", "fp", "fn", "cls_err", "geom_pairs", "img_with_xml"]
        )
        w.writeheader()
        w.writerow({k: int(counts.get(k, 0)) for k in w.fieldnames})

    # per-class csv
    stat_by_cls: dict[str, dict[str, int]] = summary.get("stat_by_class", {}) or {}
    rows: list[dict[str, Any]] = []
    for cls_name, s in stat_by_cls.items():
        gt_n = int(s.get("gt", 0))
        pred_n = int(s.get("pred", 0))
        tp_n = int(s.get("tp", 0))
        fp_n = int(s.get("fp", 0))
        fn_n = int(s.get("fn", 0))
        ce_n = int(s.get("cls_err", 0))
        prec = (tp_n / (tp_n + fp_n)) if (tp_n + fp_n) > 0 else 0.0
        rec = (tp_n / (tp_n + fn_n + ce_n)) if (tp_n + fn_n + ce_n) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        rows.append(
            {
                "class_norm": str(cls_name),
                "gt": gt_n,
                "pred": pred_n,
                "tp": tp_n,
                "fp": fp_n,
                "fn": fn_n,
                "cls_err": ce_n,
                "precision": round(float(prec), 6),
                "recall": round(float(rec), 6),
                "f1": round(float(f1), 6),
            }
        )
    rows.sort(key=lambda r: (-int(r.get("gt", 0)), str(r.get("class_norm", ""))))
    with open(out_dir / "stat_by_class.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "class_norm",
                "gt",
                "pred",
                "tp",
                "fp",
                "fn",
                "cls_err",
                "precision",
                "recall",
                "f1",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    # missing list
    missing = summary.get("missing_pred_xml", []) or []
    if missing:
        with open(out_dir / "missing_pred_xml.txt", "w", encoding="utf-8") as f:
            for rel in missing:
                f.write(str(rel) + "\n")

    errors = summary.get("parse_errors", []) or []
    if errors:
        with open(out_dir / "parse_errors.json", "w", encoding="utf-8") as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)


def _brief_print(summary: dict[str, Any]) -> None:
    meta = summary.get("meta", {}) or {}
    counts = summary.get("counts", {}) or {}
    tp = int(counts.get("tp", 0))
    fp = int(counts.get("fp", 0))
    fn = int(counts.get("fn", 0))
    ce = int(counts.get("cls_err", 0))
    imgs = int(counts.get("img_with_xml", 0))

    denom_pred = float(tp + fp)
    denom_gt = float(tp + fn + ce)
    precision = (tp / denom_pred) if denom_pred > 0 else 0.0
    recall = (tp / denom_gt) if denom_gt > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    print("======== XML 目录评估汇总 ========")
    print(f"GT : {meta.get('gt_xml_dir')}")
    print(f"Pred: {meta.get('pred_xml_dir')}")
    print(f"match: metric={meta.get('geom_metric')} thr={meta.get('geom_threshold')}")
    if meta.get("focus_classes"):
        print(f"focus_classes: {meta.get('focus_classes')}")
    print(f"images(xml): {imgs}")
    print(
        f"tp={tp} fp={fp} fn={fn} cls_err={ce} | "
        f"precision={precision*100:.2f}% recall={recall*100:.2f}% f1={f1*100:.2f}%"
    )
    miss = summary.get("missing_pred_xml", []) or []
    errs = summary.get("parse_errors", []) or []
    if miss:
        print(f"missing pred xml: {len(miss)} (see missing_pred_xml.txt if out_dir is set)")
    if errs:
        print(f"parse errors: {len(errs)} (see parse_errors.json if out_dir is set)")
    meta_vis = summary.get("meta", {}) or {}
    if meta_vis.get("visualize_matching"):
        print(
            f"matching visualization dir: {meta_vis.get('visualize_out_dir') or '(none)'} "
            f"(saved={meta_vis.get('visualization_saved_count', 0)}, "
            f"skipped_no_image={meta_vis.get('visualization_skipped_no_image_count', 0)})"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # ----------------------- 路径配置（按需修改） -----------------------
    GT_XML_DIR = "/Users/shunyaoyin/Documents/code/datasets/insect-data/test-data/福建大赛"
    PRED_XML_DIR = "/Users/shunyaoyin/Documents/code/datasets/insect-data/test-data/福建大赛-d1.5/large"
    # PRED_XML_DIR = "/Users/shunyaoyin/Documents/code/datasets/insect-data/test-data/福建大赛-d0424/large"
    GT_XML_DIR = "/Users/shunyaoyin/Documents/code/datasets/insect-data/test-data/dachong-检出测试集"
    PRED_XML_DIR = "/Users/shunyaoyin/Documents/code/datasets/insect-data/test-data/dachong-检出测试集-d0424/large"
    # PRED_XML_DIR = "/Users/shunyaoyin/Documents/code/datasets/insect-data/test-data/dachong-检出测试集-d1.8/large"

    # ----------------------- 输出 -----------------------
    OUT_DIR = PRED_XML_DIR  # None 表示不落盘，只打印
    # OUT_DIR = None

    # ----------------------- 评估配置 -----------------------
    GEOM_METRIC = "iou"  # "iou" | "ior"
    GEOM_THRESHOLD = 0.5
    # 最小矩形对角线过滤门限（像素）。当 > 0 时：对角线小于该阈值的框不参与对比（GT 与 Pred 都过滤）。
    MIN_BOX_DIAG_PX: float | None = 300  # e.g. 12.0
    CLASS_MERGE_TO_GROUPS: dict[str, list[str]] | None = None
    CLASS_MERGE_TO_GROUPS = {
        "insect": ["*"]
    }
    # 只评估这些类别（支持 raw 名 / normalize 后的组名）；None 表示不过滤
    FOCUS_CLASSES: tuple[str, ...] | None = None
    # GT 有 xml 但 pred 没有 xml：默认忽略（不纳入统计）。设为 False 则按“空预测”计算 FN。
    IGNORE_MISSING_PRED_XML = True

    # 默认关闭：在 GT XML 同级目录查找同名图片，按评估结果着色保存（与 predict_size_validate val_xml 一致）
    VISUALIZE_MATCHING = True
    # 若开启且本项为 None，则在 OUT_DIR 非空时使用 OUT_DIR / "matching_visualization"
    VISUALIZE_OUT_DIR: str | None = OUT_DIR

    cfg = EvalConfig(
        geom_metric=GEOM_METRIC,
        geom_threshold=GEOM_THRESHOLD,
        min_box_diag_px=MIN_BOX_DIAG_PX,
        class_merge_to_groups=CLASS_MERGE_TO_GROUPS,
        focus_classes=FOCUS_CLASSES,
        ignore_missing_pred_xml=IGNORE_MISSING_PRED_XML,
        visualize_matching=VISUALIZE_MATCHING,
        visualize_out_dir=VISUALIZE_OUT_DIR,
    )

    summary = evaluate_xml_dirs(
        gt_xml_dir=GT_XML_DIR,
        pred_xml_dir=PRED_XML_DIR,
        config=cfg,
        out_dir=OUT_DIR,
    )
    _brief_print(summary)
    if OUT_DIR is not None:
        print(f"reports written to: {Path(OUT_DIR).resolve()}")

