#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Detail  : 比对测试集 GT xml 与推理输出 xml，按类别统计正确/错误框的 det_conf、cls_conf 分布。
#             输出：(1) 汇总 CSV xml_fenxi_conf_stats.csv；(2) 区间直方图 CSV
#             xml_fenxi_conf_histogram.csv（按该类实际 min~max 均分桶，correct/wrong 分计，
#             含以 bin_hi 为门限时的累计误报过滤/正确损失数；样本不足类标注 reliability_note）。

from __future__ import annotations

import csv
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_FILE = Path(__file__).resolve()
_INSECT_ROOT = _FILE.parents[2]
if str(_INSECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_INSECT_ROOT))

from script.config_paths import DEFAULT_INSECT_ALG_ALL_JSON
from script.predict_all import build_eval_class_merge, load_insect_alg_all
from script.predict_seg_lib import collect_images
from script.predict_size_validate_lib import (
    ClassTierEquivalence,
    build_eval_class_display_index,
    is_class_match,
    is_metric_ignored_other,
    load_class_tier_equivalence,
    load_eval_label_alias_map,
    match_pred_gt,
    match_pred_gt_ior,
    normalize_class_name,
    parse_pascal_voc_objects,
    parse_pascal_voc_pred_objects,
)


@dataclass
class ConfBucket:
    det: list[float] = field(default_factory=list)
    cls: list[float] = field(default_factory=list)

    def add(self, det_conf: float, cls_conf: float) -> None:
        self.det.append(float(det_conf))
        self.cls.append(float(cls_conf))

    @staticmethod
    def stats(values: list[float]) -> tuple[float | None, float | None, float | None]:
        if not values:
            return None, None, None
        return min(values), max(values), sum(values) / len(values)


# 低于该样本量时，不宜作为 cls/det 门限设定参考（可在 __main__ 覆盖）
THRESHOLD_REF_MIN_SAMPLES = 10


@dataclass
class ClassConfReport:
    cls_key: str
    correct: ConfBucket = field(default_factory=ConfBucket)
    wrong: ConfBucket = field(default_factory=ConfBucket)

    @property
    def n_correct(self) -> int:
        return len(self.correct.det)

    @property
    def n_wrong(self) -> int:
        return len(self.wrong.det)


@dataclass
class EdgeImageConfReport:
    """单类别在边缘图 / 非边缘图上的预测置信度累积。"""

    cls_key: str
    edge: ConfBucket = field(default_factory=ConfBucket)
    non_edge: ConfBucket = field(default_factory=ConfBucket)

    @property
    def n_edge(self) -> int:
        return len(self.edge.det)

    @property
    def n_non_edge(self) -> int:
        return len(self.non_edge.det)


def _gt_objects_to_rows(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for o in objects:
        rows.append(
            {
                "cls_name": str(o.get("name") or ""),
                "name": str(o.get("name") or ""),
                "x1": int(o["x1"]),
                "y1": int(o["y1"]),
                "x2": int(o["x2"]),
                "y2": int(o["y2"]),
            }
        )
    return rows


def _row_confs(row: dict[str, Any]) -> tuple[float, float]:
    det = float(row.get("conf", row.get("det_conf", 0.0)) or 0.0)
    cls_c = float(row.get("cls_conf", det) or det)
    return det, cls_c


def parse_voc_image_size(xml_path: str | Path) -> tuple[int, int]:
    """从 VOC xml 读取图像宽高。"""
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"missing size: {xml_path}")
    w_el = size.find("width")
    h_el = size.find("height")
    if w_el is None or h_el is None or w_el.text is None or h_el.text is None:
        raise ValueError(f"missing width/height: {xml_path}")
    return int(float(w_el.text.strip())), int(float(h_el.text.strip()))


def min_dist_to_image_edge(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> float:
    """检测框到图像四边距离的最小值（像素）。"""
    return float(min(x1, y1, w - x2, h - y2))


def is_edge_image(
    gts: list[dict[str, Any]],
    *,
    w: int,
    h: int,
    edge_reject_distance: float,
) -> bool:
    """
    边缘图：任意 GT 框到图像边缘最小距离 < edge_reject_distance。
    口径与 predict_all 的 edge_reject_distance 一致。
    """
    thr = float(edge_reject_distance)
    if thr <= 0:
        return False
    for g in gts:
        d = min_dist_to_image_edge(int(g["x1"]), int(g["y1"]), int(g["x2"]), int(g["y2"]), w, h)
        if d < thr:
            return True
    return False


def classify_preds_on_image(
    preds: list[dict[str, Any]],
    gts: list[dict[str, Any]],
    *,
    class_merge: dict[str, list[str]] | None,
    geom_metric: str,
    geom_threshold: float,
    label_alias_map: dict[str, str] | None,
    tier_equivalence: ClassTierEquivalence | None,
    fuzzy_only_wildcard: bool,
) -> list[tuple[str, float, float, str]]:
    """
    单图 pred 框分类为 correct / wrong，返回 (norm_cls, det, cls, tag) 列表。
    tag: tp | cls_err | fp
    """
    merge = class_merge
    pred_eval_idx = [
        i
        for i, p in enumerate(preds)
        if not is_metric_ignored_other(str(p.get("cls_name", "") or ""), merge)
    ]
    gt_eval_idx = [
        j
        for j, g in enumerate(gts)
        if not is_metric_ignored_other(str(g.get("cls_name", "") or ""), merge)
    ]
    out: list[tuple[str, float, float, str]] = []
    if not pred_eval_idx:
        return out

    if not gt_eval_idx:
        for i in pred_eval_idx:
            p = preds[i]
            cls_key = normalize_class_name(
                str(p.get("cls_name", "") or ""),
                merge,
                label_alias_map=label_alias_map,
            )
            det, cls_c = _row_confs(p)
            out.append((cls_key, det, cls_c, "fp"))
        return out

    preds_eval = [preds[i] for i in pred_eval_idx]
    gts_eval = [gts[j] for j in gt_eval_idx]
    prefer_class_match: list[list[bool]] = []
    for p in preds_eval:
        row: list[bool] = []
        pred_cls = str(p.get("cls_name", "") or "")
        for g in gts_eval:
            row.append(
                is_class_match(
                    pred_cls,
                    str(g.get("cls_name", "") or ""),
                    merge,
                    None,
                    label_alias_map=label_alias_map,
                    fuzzy_only_wildcard=fuzzy_only_wildcard,
                    tier_equivalence=tier_equivalence,
                )
            )
        prefer_class_match.append(row)

    m = str(geom_metric or "iou").lower().strip()
    thr = float(geom_threshold)
    if m == "ior":
        matches_ev, matched_p_ev, _ = match_pred_gt_ior(
            preds_eval,
            gts_eval,
            thr,
            pred_tier=[0] * len(preds_eval),
            prefer_class_match=prefer_class_match,
        )
    else:
        matches_ev, matched_p_ev, _ = match_pred_gt(
            preds_eval,
            gts_eval,
            thr,
            metric="iou",
            pred_tier=[0] * len(preds_eval),
            prefer_class_match=prefer_class_match,
        )

    pred_to_gt = {
        pred_eval_idx[pi]: gt_eval_idx[gj] for pi, gj, _ in matches_ev
    }
    matched_pred_abs = {pred_eval_idx[i] for i in matched_p_ev}

    for i in pred_eval_idx:
        p = preds[i]
        cls_key = normalize_class_name(
            str(p.get("cls_name", "") or ""),
            merge,
            label_alias_map=label_alias_map,
        )
        det, cls_c = _row_confs(p)
        if i not in matched_pred_abs:
            out.append((cls_key, det, cls_c, "fp"))
            continue
        gj = pred_to_gt[i]
        gt = gts[gj]
        if is_class_match(
            str(p.get("cls_name", "") or ""),
            str(gt.get("cls_name", "") or ""),
            merge,
            None,
            label_alias_map=label_alias_map,
            fuzzy_only_wildcard=fuzzy_only_wildcard,
            tier_equivalence=tier_equivalence,
        ):
            out.append((cls_key, det, cls_c, "tp"))
        else:
            out.append((cls_key, det, cls_c, "cls_err"))
    return out


def analyze_xml_pairs(
    *,
    gt_root: Path,
    pred_root: Path,
    image_files: list[Path],
    class_merge: dict[str, list[str]] | None,
    geom_metric: str,
    geom_threshold: float,
    label_alias_map: dict[str, str] | None,
    tier_equivalence: ClassTierEquivalence | None,
    fuzzy_only_wildcard: bool,
    edge_reject_distance: float = 5.0,
) -> tuple[
    dict[str, ClassConfReport],
    dict[str, EdgeImageConfReport],
    dict[str, int],
]:
    """遍历图片，累积各类别正确/错误置信度，以及边缘/非边缘图上的预测置信度。"""
    by_cls: dict[str, ClassConfReport] = {}
    by_edge_img: dict[str, EdgeImageConfReport] = {}
    counters: dict[str, int] = defaultdict(int)

    def _bucket(cls_key: str) -> ClassConfReport:
        if cls_key not in by_cls:
            by_cls[cls_key] = ClassConfReport(cls_key=cls_key)
        return by_cls[cls_key]

    def _edge_bucket(cls_key: str) -> EdgeImageConfReport:
        if cls_key not in by_edge_img:
            by_edge_img[cls_key] = EdgeImageConfReport(cls_key=cls_key)
        return by_edge_img[cls_key]

    for img_path in image_files:
        if gt_root.is_dir():
            rel = img_path.relative_to(gt_root)
        else:
            rel = Path(img_path.name)
        gt_xml = img_path.with_suffix(".xml")
        pred_xml = pred_root / rel.with_suffix(".xml")
        counters["images"] += 1
        if not gt_xml.is_file():
            counters["skip_no_gt_xml"] += 1
            continue
        if not pred_xml.is_file():
            counters["skip_no_pred_xml"] += 1
            continue
        try:
            gts = _gt_objects_to_rows(parse_pascal_voc_objects(str(gt_xml)))
            preds = parse_pascal_voc_pred_objects(pred_xml)
            img_w, img_h = parse_voc_image_size(pred_xml)
        except (OSError, ValueError, ET.ParseError) as e:
            counters["skip_parse_error"] += 1
            print(f"[跳过] 解析失败 {rel}: {e}")
            continue
        if not gts:
            counters["skip_empty_gt"] += 1
            continue
        counters["paired"] += 1
        edge_img = is_edge_image(
            gts,
            w=img_w,
            h=img_h,
            edge_reject_distance=edge_reject_distance,
        )
        if edge_img:
            counters["edge_images"] += 1
        else:
            counters["non_edge_images"] += 1

        classified = classify_preds_on_image(
            preds,
            gts,
            class_merge=class_merge,
            geom_metric=geom_metric,
            geom_threshold=geom_threshold,
            label_alias_map=label_alias_map,
            tier_equivalence=tier_equivalence,
            fuzzy_only_wildcard=fuzzy_only_wildcard,
        )
        for cls_key, det, cls_c, tag in classified:
            b = _bucket(cls_key)
            if tag == "tp":
                b.correct.add(det, cls_c)
                counters["tp"] += 1
            else:
                b.wrong.add(det, cls_c)
                counters[tag] += 1

            eb = _edge_bucket(cls_key)
            if edge_img:
                eb.edge.add(det, cls_c)
                counters["edge_preds"] += 1
            else:
                eb.non_edge.add(det, cls_c)
                counters["non_edge_preds"] += 1

    return by_cls, by_edge_img, dict(counters)


def _fmt(v: float | None) -> str:
    if v is None:
        return "-"
    return f"{v:.4f}"


def _adaptive_bin_count(n: int, *, min_bins: int = 3, max_bins: int = 10) -> int:
    """按样本量自适应分桶数；样本极少时桶数不超过样本数。"""
    if n <= 0:
        return 0
    if n <= min_bins:
        return n
    import math

    return min(max_bins, max(min_bins, int(round(math.sqrt(n)))))


def _make_bin_edges(vmin: float, vmax: float, n_bins: int) -> list[float]:
    """在 [vmin, vmax] 上均分 n_bins 个闭区间，返回 n_bins+1 个边界。"""
    if n_bins <= 0:
        return []
    if vmin >= vmax:
        pad = 0.001 if vmin == 0 else abs(vmin) * 0.001
        vmin, vmax = vmin - pad, vmax + pad
    step = (vmax - vmin) / n_bins
    edges = [vmin + i * step for i in range(n_bins + 1)]
    edges[0] = vmin
    edges[-1] = vmax
    return edges


def _assign_bin(value: float, edges: list[float]) -> int:
    """左闭右开；最高桶含右端点。"""
    n_bins = len(edges) - 1
    if n_bins <= 0:
        return 0
    if value >= edges[-1]:
        return n_bins - 1
    if value <= edges[0]:
        return 0
    for i in range(n_bins):
        if edges[i] <= value < edges[i + 1]:
            return i
    return n_bins - 1


def _count_in_bins(values: list[float], edges: list[float]) -> list[int]:
    n_bins = max(0, len(edges) - 1)
    counts = [0] * n_bins
    for v in values:
        counts[_assign_bin(float(v), edges)] += 1
    return counts


def assess_threshold_reliability(
    n_correct: int,
    n_wrong: int,
    *,
    min_samples: int = THRESHOLD_REF_MIN_SAMPLES,
) -> tuple[str, str]:
    """
    评估该类样本是否足以支撑门限设定。

    返回 (reliability_code, note_cn)。
    """
    if n_correct == 0 and n_wrong == 0:
        return "empty", "无样本"
    notes: list[str] = []
    if n_correct == 0:
        notes.append(f"无正确样本({n_wrong}误报)")
    elif n_correct < min_samples:
        notes.append(f"正确样本偏少(n={n_correct}<{min_samples})")
    if n_wrong == 0:
        notes.append(f"无误报样本({n_correct}正确)")
    elif n_wrong < min_samples:
        notes.append(f"误报样本偏少(n={n_wrong}<{min_samples})")
    if not notes:
        return "ok", "样本量可支撑门限分析"
    if n_correct == 0 or n_wrong == 0:
        return "single_side", "；".join(notes) + "；仅单侧分布，门限参考价值有限"
    return "low_samples", "；".join(notes) + "；门限仅供参考"


@dataclass
class ConfHistogramBin:
    conf_type: str
    bin_index: int
    bin_lo: float
    bin_hi: float
    n_correct: int
    n_wrong: int
    wrong_le_bin_hi: int
    correct_le_bin_hi: int


def build_class_conf_histogram(
    report: ClassConfReport,
    conf_type: str,
    *,
    min_samples: int = THRESHOLD_REF_MIN_SAMPLES,
) -> tuple[list[ConfHistogramBin], str, str]:
    """
    单类别、单置信度字段（det/cls）的区间直方图。

    区间按该类 correct+wrong 合并后的 min/max 均分；correct 与 wrong 分别计数。
    """
    if conf_type == "det":
        c_vals, w_vals = report.correct.det, report.wrong.det
    elif conf_type == "cls":
        c_vals, w_vals = report.correct.cls, report.wrong.cls
    else:
        raise ValueError(f"unknown conf_type: {conf_type}")

    all_vals = list(c_vals) + list(w_vals)
    reliability_code, reliability_note = assess_threshold_reliability(
        len(c_vals), len(w_vals), min_samples=min_samples
    )
    if not all_vals:
        return [], reliability_code, reliability_note

    n_bins = _adaptive_bin_count(len(all_vals))
    edges = _make_bin_edges(min(all_vals), max(all_vals), n_bins)
    c_counts = _count_in_bins(c_vals, edges)
    w_counts = _count_in_bins(w_vals, edges)

    bins: list[ConfHistogramBin] = []
    wrong_cum = 0
    correct_cum = 0
    for i in range(n_bins):
        wrong_cum += w_counts[i]
        correct_cum += c_counts[i]
        bins.append(
            ConfHistogramBin(
                conf_type=conf_type,
                bin_index=i,
                bin_lo=edges[i],
                bin_hi=edges[i + 1],
                n_correct=c_counts[i],
                n_wrong=w_counts[i],
                wrong_le_bin_hi=wrong_cum,
                correct_le_bin_hi=correct_cum,
            )
        )
    return bins, reliability_code, reliability_note


def print_conf_report(
    by_cls: dict[str, ClassConfReport],
    *,
    class_display_index: dict[str, str] | None = None,
) -> None:
    idx = class_display_index or {}
    rows = sorted(
        by_cls.values(),
        key=lambda r: (-(r.n_correct + r.n_wrong), r.cls_key),
    )
    print(
        "类别           | 正确数 | det_min | det_max | det_avg | cls_min | cls_max | cls_avg "
        "| 错误数 | det_min | det_max | det_avg | cls_min | cls_max | cls_avg"
    )
    print("-" * 120)
    tot_correct = ConfBucket()
    tot_wrong = ConfBucket()
    for r in rows:
        label = (idx.get(r.cls_key) or r.cls_key)[:14]
        c_det_min, c_det_max, c_det_avg = ConfBucket.stats(r.correct.det)
        c_cls_min, c_cls_max, c_cls_avg = ConfBucket.stats(r.correct.cls)
        w_det_min, w_det_max, w_det_avg = ConfBucket.stats(r.wrong.det)
        w_cls_min, w_cls_max, w_cls_avg = ConfBucket.stats(r.wrong.cls)
        tot_correct.det.extend(r.correct.det)
        tot_correct.cls.extend(r.correct.cls)
        tot_wrong.det.extend(r.wrong.det)
        tot_wrong.cls.extend(r.wrong.cls)
        print(
            f"{label:<14} | {r.n_correct:>6} | {_fmt(c_det_min):>7} | {_fmt(c_det_max):>7} | "
            f"{_fmt(c_det_avg):>7} | {_fmt(c_cls_min):>7} | {_fmt(c_cls_max):>7} | {_fmt(c_cls_avg):>7} | "
            f"{r.n_wrong:>6} | {_fmt(w_det_min):>7} | {_fmt(w_det_max):>7} | {_fmt(w_det_avg):>7} | "
            f"{_fmt(w_cls_min):>7} | {_fmt(w_cls_max):>7} | {_fmt(w_cls_avg):>7}"
        )
    print("-" * 120)
    tc_det_min, tc_det_max, tc_det_avg = ConfBucket.stats(tot_correct.det)
    tc_cls_min, tc_cls_max, tc_cls_avg = ConfBucket.stats(tot_correct.cls)
    tw_det_min, tw_det_max, tw_det_avg = ConfBucket.stats(tot_wrong.det)
    tw_cls_min, tw_cls_max, tw_cls_avg = ConfBucket.stats(tot_wrong.cls)
    print(
        f"{'合计':<14} | {len(tot_correct.det):>6} | {_fmt(tc_det_min):>7} | {_fmt(tc_det_max):>7} | "
        f"{_fmt(tc_det_avg):>7} | {_fmt(tc_cls_min):>7} | {_fmt(tc_cls_max):>7} | {_fmt(tc_cls_avg):>7} | "
        f"{len(tot_wrong.det):>6} | {_fmt(tw_det_min):>7} | {_fmt(tw_det_max):>7} | "
        f"{_fmt(tw_det_avg):>7} | {_fmt(tw_cls_min):>7} | {_fmt(tw_cls_max):>7} | {_fmt(tw_cls_avg):>7}"
    )


def export_conf_report_csv(
    path: Path,
    by_cls: dict[str, ClassConfReport],
    *,
    class_display_index: dict[str, str] | None = None,
    min_samples: int = THRESHOLD_REF_MIN_SAMPLES,
) -> None:
    idx = class_display_index or {}
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "cls_key",
        "display_name",
        "n_correct",
        "correct_det_min",
        "correct_det_max",
        "correct_det_avg",
        "correct_cls_min",
        "correct_cls_max",
        "correct_cls_avg",
        "n_wrong",
        "wrong_det_min",
        "wrong_det_max",
        "wrong_det_avg",
        "wrong_cls_min",
        "wrong_cls_max",
        "wrong_cls_avg",
        "reliability_code",
        "reliability_note",
    ]
    rows = sorted(by_cls.values(), key=lambda r: r.cls_key)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            c_det_min, c_det_max, c_det_avg = ConfBucket.stats(r.correct.det)
            c_cls_min, c_cls_max, c_cls_avg = ConfBucket.stats(r.correct.cls)
            w_det_min, w_det_max, w_det_avg = ConfBucket.stats(r.wrong.det)
            w_cls_min, w_cls_max, w_cls_avg = ConfBucket.stats(r.wrong.cls)
            rel_code, rel_note = assess_threshold_reliability(
                r.n_correct, r.n_wrong, min_samples=min_samples
            )
            w.writerow(
                {
                    "cls_key": r.cls_key,
                    "display_name": idx.get(r.cls_key) or r.cls_key,
                    "n_correct": r.n_correct,
                    "correct_det_min": c_det_min,
                    "correct_det_max": c_det_max,
                    "correct_det_avg": c_det_avg,
                    "correct_cls_min": c_cls_min,
                    "correct_cls_max": c_cls_max,
                    "correct_cls_avg": c_cls_avg,
                    "n_wrong": r.n_wrong,
                    "wrong_det_min": w_det_min,
                    "wrong_det_max": w_det_max,
                    "wrong_det_avg": w_det_avg,
                    "wrong_cls_min": w_cls_min,
                    "wrong_cls_max": w_cls_max,
                    "wrong_cls_avg": w_cls_avg,
                    "reliability_code": rel_code,
                    "reliability_note": rel_note,
                }
            )


def export_conf_histogram_csv(
    path: Path,
    by_cls: dict[str, ClassConfReport],
    *,
    class_display_index: dict[str, str] | None = None,
    min_samples: int = THRESHOLD_REF_MIN_SAMPLES,
) -> None:
    """
    按类别导出 det/cls 置信度区间直方图（long format）。

    每行一个区间：correct / wrong 分计数量，并给出以 bin_hi 为门限时
    累计误报过滤数、累计正确损失数（conf <= bin_hi 的框数）。
    """
    idx = class_display_index or {}
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "cls_key",
        "display_name",
        "conf_type",
        "n_correct_total",
        "n_wrong_total",
        "reliability_code",
        "reliability_note",
        "bin_index",
        "bin_lo",
        "bin_hi",
        "n_correct",
        "n_wrong",
        "wrong_le_bin_hi",
        "correct_le_bin_hi",
        "wrong_filtered_if_thr_eq_bin_hi",
        "correct_lost_if_thr_eq_bin_hi",
    ]
    rows_out: list[dict[str, Any]] = []
    for report in sorted(by_cls.values(), key=lambda r: r.cls_key):
        display = idx.get(report.cls_key) or report.cls_key
        rel_code, rel_note = assess_threshold_reliability(
            report.n_correct, report.n_wrong, min_samples=min_samples
        )
        for conf_type in ("det", "cls"):
            bins, _, _ = build_class_conf_histogram(
                report, conf_type, min_samples=min_samples
            )
            if not bins:
                continue
            for b in bins:
                rows_out.append(
                    {
                        "cls_key": report.cls_key,
                        "display_name": display,
                        "conf_type": conf_type,
                        "n_correct_total": report.n_correct,
                        "n_wrong_total": report.n_wrong,
                        "reliability_code": rel_code,
                        "reliability_note": rel_note,
                        "bin_index": b.bin_index,
                        "bin_lo": round(b.bin_lo, 6),
                        "bin_hi": round(b.bin_hi, 6),
                        "n_correct": b.n_correct,
                        "n_wrong": b.n_wrong,
                        "wrong_le_bin_hi": b.wrong_le_bin_hi,
                        "correct_le_bin_hi": b.correct_le_bin_hi,
                        "wrong_filtered_if_thr_eq_bin_hi": b.wrong_le_bin_hi,
                        "correct_lost_if_thr_eq_bin_hi": b.correct_le_bin_hi,
                    }
                )
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)


def print_conf_histogram_report(
    by_cls: dict[str, ClassConfReport],
    *,
    class_display_index: dict[str, str] | None = None,
    min_samples: int = THRESHOLD_REF_MIN_SAMPLES,
    top_n: int = 15,
) -> None:
    """打印误报量靠前类别的 cls 区间分布（便于终端快速浏览）。"""
    idx = class_display_index or {}
    ranked = sorted(
        by_cls.values(),
        key=lambda r: (-r.n_wrong, -r.n_correct, r.cls_key),
    )[:top_n]
    print(
        f"======== 区间直方图（cls，Top{top_n}误报类；"
        f"样本<{min_samples}标注*不宜作门限参考*）========"
    )
    for report in ranked:
        display = (idx.get(report.cls_key) or report.cls_key)[:12]
        rel_code, rel_note = assess_threshold_reliability(
            report.n_correct, report.n_wrong, min_samples=min_samples
        )
        flag = "" if rel_code == "ok" else f" [{rel_note}]"
        bins, _, _ = build_class_conf_histogram(report, "cls", min_samples=min_samples)
        if not bins:
            print(f"{display}({report.cls_key}): 无 cls 样本{flag}")
            continue
        parts = []
        for i, b in enumerate(bins):
            hi_br = "]" if i == len(bins) - 1 else ")"
            parts.append(
                f"[{b.bin_lo:.2f},{b.bin_hi:.2f}{hi_br} c={b.n_correct} w={b.n_wrong}"
            )
        print(
            f"{display}: 正确={report.n_correct} 误报={report.n_wrong}{flag}\n"
            f"  {' | '.join(parts)}"
        )


def print_edge_image_conf_report(
    by_edge: dict[str, EdgeImageConfReport],
    *,
    class_display_index: dict[str, str] | None = None,
    edge_reject_distance: float = 5.0,
) -> None:
    idx = class_display_index or {}
    rows = sorted(
        by_edge.values(),
        key=lambda r: (-(r.n_edge + r.n_non_edge), r.cls_key),
    )
    print(
        f"边缘图判定: 任意 GT 框到图像边缘距离 < {edge_reject_distance:g}px；"
        "统计该图上全部预测框 det_conf / cls_conf"
    )
    print(
        "类别           | 边缘-数 | det_min | det_max | det_avg | cls_min | cls_max | cls_avg "
        "| 非边缘-数 | det_min | det_max | det_avg | cls_min | cls_max | cls_avg"
    )
    print("-" * 120)
    tot_edge = ConfBucket()
    tot_non_edge = ConfBucket()
    for r in rows:
        label = (idx.get(r.cls_key) or r.cls_key)[:14]
        e_det_min, e_det_max, e_det_avg = ConfBucket.stats(r.edge.det)
        e_cls_min, e_cls_max, e_cls_avg = ConfBucket.stats(r.edge.cls)
        n_det_min, n_det_max, n_det_avg = ConfBucket.stats(r.non_edge.det)
        n_cls_min, n_cls_max, n_cls_avg = ConfBucket.stats(r.non_edge.cls)
        tot_edge.det.extend(r.edge.det)
        tot_edge.cls.extend(r.edge.cls)
        tot_non_edge.det.extend(r.non_edge.det)
        tot_non_edge.cls.extend(r.non_edge.cls)
        print(
            f"{label:<14} | {r.n_edge:>7} | {_fmt(e_det_min):>7} | {_fmt(e_det_max):>7} | "
            f"{_fmt(e_det_avg):>7} | {_fmt(e_cls_min):>7} | {_fmt(e_cls_max):>7} | {_fmt(e_cls_avg):>7} | "
            f"{r.n_non_edge:>8} | {_fmt(n_det_min):>7} | {_fmt(n_det_max):>7} | {_fmt(n_det_avg):>7} | "
            f"{_fmt(n_cls_min):>7} | {_fmt(n_cls_max):>7} | {_fmt(n_cls_avg):>7}"
        )
    print("-" * 120)
    te_det_min, te_det_max, te_det_avg = ConfBucket.stats(tot_edge.det)
    te_cls_min, te_cls_max, te_cls_avg = ConfBucket.stats(tot_edge.cls)
    tn_det_min, tn_det_max, tn_det_avg = ConfBucket.stats(tot_non_edge.det)
    tn_cls_min, tn_cls_max, tn_cls_avg = ConfBucket.stats(tot_non_edge.cls)
    print(
        f"{'合计':<14} | {len(tot_edge.det):>7} | {_fmt(te_det_min):>7} | {_fmt(te_det_max):>7} | "
        f"{_fmt(te_det_avg):>7} | {_fmt(te_cls_min):>7} | {_fmt(te_cls_max):>7} | {_fmt(te_cls_avg):>7} | "
        f"{len(tot_non_edge.det):>8} | {_fmt(tn_det_min):>7} | {_fmt(tn_det_max):>7} | "
        f"{_fmt(tn_det_avg):>7} | {_fmt(tn_cls_min):>7} | {_fmt(tn_cls_max):>7} | {_fmt(tn_cls_avg):>7}"
    )


def export_edge_image_conf_report_csv(
    path: Path,
    by_edge: dict[str, EdgeImageConfReport],
    *,
    class_display_index: dict[str, str] | None = None,
) -> None:
    idx = class_display_index or {}
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "cls_key",
        "display_name",
        "n_edge",
        "edge_det_min",
        "edge_det_max",
        "edge_det_avg",
        "edge_cls_min",
        "edge_cls_max",
        "edge_cls_avg",
        "n_non_edge",
        "non_edge_det_min",
        "non_edge_det_max",
        "non_edge_det_avg",
        "non_edge_cls_min",
        "non_edge_cls_max",
        "non_edge_cls_avg",
    ]
    rows = sorted(by_edge.values(), key=lambda r: r.cls_key)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            e_det_min, e_det_max, e_det_avg = ConfBucket.stats(r.edge.det)
            e_cls_min, e_cls_max, e_cls_avg = ConfBucket.stats(r.edge.cls)
            n_det_min, n_det_max, n_det_avg = ConfBucket.stats(r.non_edge.det)
            n_cls_min, n_cls_max, n_cls_avg = ConfBucket.stats(r.non_edge.cls)
            w.writerow(
                {
                    "cls_key": r.cls_key,
                    "display_name": idx.get(r.cls_key) or r.cls_key,
                    "n_edge": r.n_edge,
                    "edge_det_min": e_det_min,
                    "edge_det_max": e_det_max,
                    "edge_det_avg": e_det_avg,
                    "edge_cls_min": e_cls_min,
                    "edge_cls_max": e_cls_max,
                    "edge_cls_avg": e_cls_avg,
                    "n_non_edge": r.n_non_edge,
                    "non_edge_det_min": n_det_min,
                    "non_edge_det_max": n_det_max,
                    "non_edge_det_avg": n_det_avg,
                    "non_edge_cls_min": n_cls_min,
                    "non_edge_cls_max": n_cls_max,
                    "non_edge_cls_avg": n_cls_avg,
                }
            )


if __name__ == "__main__":
    # ----------------------- 需要你改的参数 -----------------------
    # 测试集：图片与同目录 GT xml
    GT_DIR = "/Volumes/shunyao-h1/测试数据/北京田间第1批/辛集市全标注5-6"
    # 推理输出目录：与 GT 保持相同相对路径的预测 xml（OUTPUT_XML=True 且 OUTPUT_XML_DUAL_CONF=True）
    PRED_DIR = "/Volumes/shunyao-h1/测试数据/北京田间第1批/辛集市全标注5-6-3.13.4-3.10.2"


    # GT_DIR = "/Volumes/shunyao-h1/训练数据/北京比赛/北京设备全标注0621"
    # 推理输出目录：与 GT 保持相同相对路径的预测 xml（OUTPUT_XML=True 且 OUTPUT_XML_DUAL_CONF=True）
    # PRED_DIR = "/Volumes/shunyao-h1/训练数据/北京比赛/北京设备全标注0621-3.13.4.768-3.8.11"
    # 几何匹配（口径同 predict_all 校验）
    VAL_BOX_MATCH_METRIC = "iou"  # "iou" | "ior"
    VAL_GEOM_THRESHOLD = 0.25
    # 边缘图判定：任意 GT 框到图像边缘距离 < 该值（与 predict_all edge_reject_distance 一致）
    EDGE_REJECT_DISTANCE = 5.0
    INSECT_ALG_ALL_JSON: str | Path | None = DEFAULT_INSECT_ALG_ALL_JSON
    CLASS_MERGE_TO_GROUPS: dict[str, list[str]] | None = None
    EVAL_INSECT_WILDCARD = True
    EVAL_INSECT_WILDCARD_STRICT = True
    # 统计 CSV 输出；None 则写到 PRED_DIR/xml_fenxi_conf_stats.csv
    OUTPUT_CSV: str | Path | None = None
    # 区间直方图 CSV；None 则写到 PRED_DIR/xml_fenxi_conf_histogram.csv
    OUTPUT_HISTOGRAM_CSV: str | Path | None = None
    # 样本量低于该值时标注「不宜作门限参考」
    THRESHOLD_REF_MIN_SAMPLES = 10

    gt_root = Path(GT_DIR).expanduser().resolve()
    pred_root = Path(PRED_DIR).expanduser().resolve()
    if not pred_root.is_dir():
        raise SystemExit(f"预测目录不存在: {pred_root}")

    alg_config = None
    if INSECT_ALG_ALL_JSON:
        try:
            alg_config = load_insect_alg_all(
                _INSECT_ROOT / "script" / INSECT_ALG_ALL_JSON
                if not Path(str(INSECT_ALG_ALL_JSON)).is_absolute()
                else INSECT_ALG_ALL_JSON
            )
        except (OSError, ValueError) as e:
            print(f"警告: 未加载 insect_alg_all ({e})，中文别名映射不可用")

    class_merge = build_eval_class_merge(
        CLASS_MERGE_TO_GROUPS, insect_wildcard=EVAL_INSECT_WILDCARD
    )
    label_alias_map = load_eval_label_alias_map(alg_config=alg_config)
    tier_equivalence = load_class_tier_equivalence(alg_config=alg_config)
    fuzzy_only_wildcard = bool(EVAL_INSECT_WILDCARD and EVAL_INSECT_WILDCARD_STRICT)
    display_index = build_eval_class_display_index(alg_config=alg_config)

    _, image_files = collect_images(str(gt_root))
    print(f"GT 目录: {gt_root}")
    print(f"预测 xml 目录: {pred_root}")
    print(
        f"匹配: metric={VAL_BOX_MATCH_METRIC} thr={VAL_GEOM_THRESHOLD} "
        f"图片数={len(image_files)}"
    )
    print(
        "正确=几何匹配且类名一致(TP)；错误=多检(FP)+类型错(cls_err)；"
        "按预测类名归并统计 det_conf / cls_conf"
    )

    by_cls, by_edge_img, counters = analyze_xml_pairs(
        gt_root=gt_root,
        pred_root=pred_root,
        image_files=image_files,
        class_merge=class_merge,
        geom_metric=VAL_BOX_MATCH_METRIC,
        geom_threshold=VAL_GEOM_THRESHOLD,
        label_alias_map=label_alias_map,
        tier_equivalence=tier_equivalence,
        fuzzy_only_wildcard=fuzzy_only_wildcard,
        edge_reject_distance=EDGE_REJECT_DISTANCE,
    )

    print(
        f"配对={counters.get('paired', 0)} "
        f"TP={counters.get('tp', 0)} "
        f"类型错={counters.get('cls_err', 0)} "
        f"多检={counters.get('fp', 0)} "
        f"边缘图={counters.get('edge_images', 0)} "
        f"非边缘图={counters.get('non_edge_images', 0)} "
        f"跳过(无GT xml)={counters.get('skip_no_gt_xml', 0)} "
        f"跳过(无预测xml)={counters.get('skip_no_pred_xml', 0)}"
    )
    print("======== 各类别置信度统计 ========")
    print_conf_report(by_cls, class_display_index=display_index)

    csv_path = Path(OUTPUT_CSV) if OUTPUT_CSV else pred_root / "xml_fenxi_conf_stats.csv"
    export_conf_report_csv(
        csv_path,
        by_cls,
        class_display_index=display_index,
        min_samples=THRESHOLD_REF_MIN_SAMPLES,
    )
    print(f"CSV: {csv_path}")

    hist_csv_path = (
        Path(OUTPUT_HISTOGRAM_CSV)
        if OUTPUT_HISTOGRAM_CSV
        else pred_root / "xml_fenxi_conf_histogram.csv"
    )
    export_conf_histogram_csv(
        hist_csv_path,
        by_cls,
        class_display_index=display_index,
        min_samples=THRESHOLD_REF_MIN_SAMPLES,
    )
    print(f"区间直方图 CSV: {hist_csv_path}")
    print_conf_histogram_report(
        by_cls,
        class_display_index=display_index,
        min_samples=THRESHOLD_REF_MIN_SAMPLES,
    )

    print("======== 边缘图 / 非边缘图 各类别置信度统计 ========")
    print_edge_image_conf_report(
        by_edge_img,
        class_display_index=display_index,
        edge_reject_distance=EDGE_REJECT_DISTANCE,
    )
    edge_csv_path = pred_root / "xml_fenxi_edge_image_conf_stats.csv"
    export_edge_image_conf_report_csv(
        edge_csv_path, by_edge_img, class_display_index=display_index
    )
    print(f"边缘图 CSV: {edge_csv_path}")
