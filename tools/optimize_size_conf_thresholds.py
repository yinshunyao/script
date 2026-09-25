#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Detail  : 用低门限推理 XML 离线网格搜索 detect/cls 大小虫门限，最大化生产计数总分。
#            识别数=报出框正确匹配 report_tp（IoU）；不含 filtered/other。

from __future__ import annotations

import csv
import itertools
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_FILE = Path(__file__).resolve()
_INSECT_ROOT = _FILE.parents[2]
if str(_INSECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_INSECT_ROOT))

from script.config_paths import INSECT_ALG_SHENGCHAN_JSON, peek_composed_insect_alg_dict
from script.predict_all import (
    _competition_recognize_count,
    _eval_pred_stat_class_name,
    _is_eval_coarse_class_name,
    _is_other_class_name,
    build_class_size_class_index,
    build_eval_class_merge,
    competition_counting_accuracy_percent,
    compute_competition_counting_summary,
    print_competition_counting_summary,
    resolve_competition_counting_focus,
    resolve_effective_cls_conf_threshold,
    resolve_validation_focus_config,
    try_resolve_threshold_size_class,
)
from script.predict_seg_lib import collect_images
from script.predict_size_validate_lib import (
    ClassTierEquivalence,
    box_iou,
    build_eval_class_display_index,
    build_eval_focus_set,
    is_class_match,
    is_eval_ignored_class,
    is_metric_ignored_other,
    load_class_tier_equivalence,
    load_eval_label_alias_map,
    merge_stat_by_cls,
    normalize_class_name,
    parse_pascal_voc_objects,
    parse_pascal_voc_pred_objects,
)


@dataclass(frozen=True)
class PredBox:
    cls_key: str
    det_conf: float
    cls_conf: float
    size_class: str | None
    fallback_cls_thr: float | None
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass(frozen=True)
class GtBox:
    cls_key: str
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass(frozen=True)
class ImagePair:
    gts: tuple[GtBox, ...]
    preds: tuple[PredBox, ...]


@dataclass(frozen=True)
class PrecomputedImageMatch:
    """单图预计算：全框 IoU / 类匹配；门限搜索只过滤子集再贪心。"""

    pair: ImagePair
    iou: tuple[tuple[float, ...], ...]  # [pred_i][gt_j]
    class_match: tuple[tuple[bool, ...], ...]  # [pred_i][gt_j]


@dataclass(frozen=True)
class ThresholdQuad:
    detect_small: float
    detect_big: float
    cls_small: float
    cls_big: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (
            self.detect_small,
            self.detect_big,
            self.cls_small,
            self.cls_big,
        )

    def sum_thr(self) -> float:
        return sum(self.as_tuple())


@dataclass(frozen=True)
class ClassConfThr:
    """单类检出 / 分类门限（覆盖体型全局门限）。"""

    detect: float
    cls: float

    def sum_thr(self) -> float:
        return self.detect + self.cls


def _nested_cls_cfg(alg: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(alg, dict):
        return {}
    star = ((alg.get("models") or {}).get("detect_big") or {}).get("out") or {}
    if not isinstance(star, dict):
        return {}
    cls = ((star.get("*") or {}).get("models") or {}).get("cls")
    return cls if isinstance(cls, dict) else {}


def _skip_dot_file(path: Path) -> bool:
    return path.name.startswith("._")


def box_kept(
    box: PredBox,
    thr: ThresholdQuad,
    class_overrides: dict[str, ClassConfThr] | None = None,
) -> bool:
    """检测 ``>=``、分类 ``>``；未知体型检测用 min(small,big)，分类用原 cls_conf 回退。"""
    if class_overrides:
        ov = class_overrides.get(box.cls_key)
        if ov is not None:
            if box.det_conf < ov.detect:
                return False
            return box.cls_conf > ov.cls
    size = box.size_class
    if size == "small":
        if box.det_conf < thr.detect_small:
            return False
        return box.cls_conf > thr.cls_small
    if size == "large":
        if box.det_conf < thr.detect_big:
            return False
        return box.cls_conf > thr.cls_big
    det_thr = min(thr.detect_small, thr.detect_big)
    if box.det_conf < det_thr:
        return False
    if box.fallback_cls_thr is None:
        return True
    return box.cls_conf > float(box.fallback_cls_thr)


def _norm_cls(
    raw: str,
    *,
    class_merge: dict[str, list[str]] | None,
    label_alias_map: dict[str, str] | None,
    tier_equivalence: ClassTierEquivalence | None = None,
) -> str:
    """与 predict_all 比赛计数相同：中文别名 → 拼音，再按 class_tier_aliases 并到报出 key。"""
    key = normalize_class_name(
        str(raw or ""),
        class_merge,
        label_alias_map=label_alias_map,
    )
    if key and tier_equivalence is not None:
        key = tier_equivalence.canonical_stat_name(key)
    return key


def expand_focus_for_stat_keys(
    focus: frozenset[str],
    *,
    class_merge: dict[str, list[str]] | None,
    label_alias_map: dict[str, str] | None,
    tier_equivalence: ClassTierEquivalence | None,
) -> frozenset[str]:
    """``merge_stat_by_cls`` 会把 key 并到 canonical；focus 须同时含拼音、中文与 rollup 键。"""
    return build_eval_focus_set(
        focus,
        merge=class_merge,
        label_alias_map=label_alias_map,
        tier_equivalence=tier_equivalence,
    )


def _skip_eval_name(
    raw: str,
    *,
    class_merge: dict[str, list[str]] | None,
    label_alias_map: dict[str, str] | None,
    ignore_classes: frozenset[str],
) -> bool:
    if is_metric_ignored_other(raw, class_merge):
        return True
    return is_eval_ignored_class(
        raw, ignore_classes, class_merge, label_alias_map=label_alias_map
    )


def _is_non_report_pred_row(p: dict[str, Any]) -> bool:
    """
    与 predict_all 比赛识别数一致：filtered / other / 昆虫等粗类不报出、不计识别数。

    ``insect`` 通配开启时 ``is_metric_ignored_other`` 会对 other 放行，这里仍要挡掉。
    """
    if p.get("filtered") or str(p.get("filter_reason") or "").strip():
        return True
    raw = str(p.get("name") or "").strip()
    if not raw:
        return True
    if _is_other_class_name(raw) or _is_eval_coarse_class_name(raw):
        return True
    return False


def _pred_report_class_key(
    p: dict[str, Any],
    *,
    class_merge: dict[str, list[str]] | None,
    label_alias_map: dict[str, str] | None,
    tier_equivalence: ClassTierEquivalence | None,
) -> str:
    """报出类 key：优先 XML ``name``（同 ``_eval_pred_stat_class_name``），再 tier 归一。"""
    key = _eval_pred_stat_class_name(p, class_merge, label_alias_map)
    if not key or _is_other_class_name(key) or _is_eval_coarse_class_name(key):
        return ""
    if tier_equivalence is not None:
        key = tier_equivalence.canonical_stat_name(key)
    return str(key or "")


def load_gt_and_pred_boxes(
    *,
    gt_root: Path,
    pred_root: Path,
    image_files: list[Path],
    alg: dict[str, Any],
    class_merge: dict[str, list[str]] | None,
    label_alias_map: dict[str, str] | None,
    ignore_classes: frozenset[str],
    size_class_by_name: dict[str, str],
    generic_cls_conf: float | None,
    cls_cfg: dict[str, Any],
    tier_equivalence: ClassTierEquivalence | None = None,
) -> tuple[list[ImagePair], dict[str, int]]:
    """加载逐图 GT/报出框；filtered/other 不入池（与 predict_all 识别口径一致）。"""
    pairs: list[ImagePair] = []
    counters: dict[str, int] = defaultdict(int)
    for img_path in image_files:
        if _skip_dot_file(img_path):
            continue
        counters["images"] += 1
        if gt_root.is_dir():
            try:
                rel = img_path.relative_to(gt_root)
            except ValueError:
                rel = Path(img_path.name)
        else:
            rel = Path(img_path.name)
        gt_xml = img_path.with_suffix(".xml")
        pred_xml = pred_root / rel.with_suffix(".xml")
        if not gt_xml.is_file():
            counters["skip_no_gt_xml"] += 1
            continue
        if not pred_xml.is_file():
            counters["skip_no_pred_xml"] += 1
            continue
        counters["gt_xml"] += 1
        try:
            gts_raw = parse_pascal_voc_objects(str(gt_xml))
        except (OSError, ValueError) as e:
            counters["skip_gt_parse"] += 1
            print(f"[跳过] GT 解析失败 {rel}: {e}")
            continue
        gt_boxes: list[GtBox] = []
        for g in gts_raw:
            raw = str(g.get("name") or "")
            if _skip_eval_name(
                raw,
                class_merge=class_merge,
                label_alias_map=label_alias_map,
                ignore_classes=ignore_classes,
            ):
                continue
            key = _norm_cls(
                raw,
                class_merge=class_merge,
                label_alias_map=label_alias_map,
                tier_equivalence=tier_equivalence,
            )
            if not key:
                continue
            gt_boxes.append(
                GtBox(
                    cls_key=key,
                    x1=int(g["x1"]),
                    y1=int(g["y1"]),
                    x2=int(g["x2"]),
                    y2=int(g["y2"]),
                )
            )
            counters["gt_box"] += 1
        try:
            preds = parse_pascal_voc_pred_objects(pred_xml)
        except (OSError, ValueError) as e:
            counters["skip_pred_parse"] += 1
            print(f"[跳过] 预测解析失败 {rel}: {e}")
            continue
        counters["pred_xml"] += 1
        pred_boxes: list[PredBox] = []
        for p in preds:
            if _is_non_report_pred_row(p):
                counters["pred_filtered"] += 1
                continue
            raw = str(p.get("name") or "").strip()
            if _skip_eval_name(
                raw,
                class_merge=class_merge,
                label_alias_map=label_alias_map,
                ignore_classes=ignore_classes,
            ):
                counters["pred_ignored"] += 1
                continue
            key = _pred_report_class_key(
                p,
                class_merge=class_merge,
                label_alias_map=label_alias_map,
                tier_equivalence=tier_equivalence,
            )
            if not key:
                counters["pred_ignored"] += 1
                continue
            det = float(p.get("det_conf", p.get("conf", 0.0)) or 0.0)
            cls_c = float(p.get("cls_conf", det) or det)
            size = try_resolve_threshold_size_class(
                p, None, size_class_by_name
            )
            fallback: float | None = None
            if size not in ("small", "large"):
                fallback = resolve_effective_cls_conf_threshold(
                    alg,
                    key,
                    generic_cls_conf,
                    cls_cfg=cls_cfg,
                    size_class=None,
                    size_class_by_name=size_class_by_name,
                )
            pred_boxes.append(
                PredBox(
                    cls_key=key,
                    det_conf=det,
                    cls_conf=cls_c,
                    size_class=size,
                    fallback_cls_thr=fallback,
                    x1=int(p["x1"]),
                    y1=int(p["y1"]),
                    x2=int(p["x2"]),
                    y2=int(p["y2"]),
                )
            )
            counters["pred_box"] += 1
        pairs.append(ImagePair(gts=tuple(gt_boxes), preds=tuple(pred_boxes)))
    return pairs, dict(counters)


def _empty_stat() -> dict[str, int]:
    return {
        "gt": 0,
        "pred": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "cls_err": 0,
        "report_tp": 0,
    }


def precompute_image_matches(
    pairs: list[ImagePair],
    *,
    class_merge: dict[str, list[str]] | None = None,
    label_alias_map: dict[str, str] | None = None,
    tier_equivalence: ClassTierEquivalence | None = None,
) -> list[PrecomputedImageMatch]:
    """一次算全图 pred×gt 的 IoU 与类匹配，供门限网格反复复用。"""
    out: list[PrecomputedImageMatch] = []
    for pair in pairs:
        n_p, n_g = len(pair.preds), len(pair.gts)
        if n_p == 0 or n_g == 0:
            out.append(
                PrecomputedImageMatch(pair=pair, iou=tuple(), class_match=tuple())
            )
            continue
        iou_rows: list[tuple[float, ...]] = []
        cm_rows: list[tuple[bool, ...]] = []
        for p in pair.preds:
            pb = (p.x1, p.y1, p.x2, p.y2)
            iou_row: list[float] = []
            cm_row: list[bool] = []
            for g in pair.gts:
                iou_row.append(box_iou(pb, (g.x1, g.y1, g.x2, g.y2)))
                cm_row.append(
                    is_class_match(
                        p.cls_key,
                        g.cls_key,
                        class_merge,
                        None,
                        label_alias_map=label_alias_map,
                        tier_equivalence=tier_equivalence,
                    )
                )
            iou_rows.append(tuple(iou_row))
            cm_rows.append(tuple(cm_row))
        out.append(
            PrecomputedImageMatch(
                pair=pair,
                iou=tuple(iou_rows),
                class_match=tuple(cm_rows),
            )
        )
    return out


def _greedy_match_kept(
    kept_pis: list[int],
    iou: tuple[tuple[float, ...], ...],
    class_match: tuple[tuple[bool, ...], ...],
    n_gt: int,
    geom_threshold: float,
) -> list[tuple[int, int]]:
    """与 ``match_pred_gt`` 同序：同类优先，再 IoU；返回 (原 pred 下标, gt 下标)。"""
    candidates: list[tuple[tuple[int, float], int, int]] = []
    thr = float(geom_threshold)
    for pi in kept_pis:
        iou_row = iou[pi]
        cm_row = class_match[pi]
        for gj in range(n_gt):
            score = iou_row[gj]
            if score < thr:
                continue
            cm_penalty = 0 if cm_row[gj] else 1
            candidates.append(((cm_penalty, -score), pi, gj))
    candidates.sort(key=lambda x: x[0])
    used_p: set[int] = set()
    used_g: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _key, pi, gj in candidates:
        if pi in used_p or gj in used_g:
            continue
        used_p.add(pi)
        used_g.add(gj)
        matches.append((pi, gj))
    return matches


def build_stat_by_cls_at(
    prepared: list[PrecomputedImageMatch],
    thr: ThresholdQuad,
    *,
    geom_threshold: float = 0.5,
    class_overrides: dict[str, ClassConfThr] | None = None,
) -> dict[str, dict[str, int]]:
    """
    门限过滤后逐图 IoU 匹配：鉴定=gt，报出=pred，识别数用 report_tp（报出同类正确）。
    几何/类匹配矩阵已预计算，此处只做子集过滤 + 贪心。
    """
    stat: dict[str, dict[str, int]] = defaultdict(_empty_stat)
    for item in prepared:
        pair = item.pair
        for g in pair.gts:
            stat[g.cls_key]["gt"] += 1
        kept_pis = [
            i
            for i, p in enumerate(pair.preds)
            if box_kept(p, thr, class_overrides)
        ]
        for pi in kept_pis:
            stat[pair.preds[pi].cls_key]["pred"] += 1
        if not pair.gts:
            continue
        if not kept_pis or not item.iou:
            for g in pair.gts:
                stat[g.cls_key]["fn"] += 1
            continue
        matches = _greedy_match_kept(
            kept_pis,
            item.iou,
            item.class_match,
            len(pair.gts),
            geom_threshold,
        )
        matched_p: set[int] = set()
        matched_g: set[int] = set()
        for pi, gj in matches:
            matched_p.add(pi)
            matched_g.add(gj)
            if item.class_match[pi][gj]:
                gt_key = pair.gts[gj].cls_key
                stat[gt_key]["tp"] += 1
                # 池内仅报出框（已剔 filtered/other），同类匹配即比赛识别正确
                stat[gt_key]["report_tp"] += 1
            else:
                gt_key = pair.gts[gj].cls_key
                pred_key = pair.preds[pi].cls_key
                stat[gt_key]["cls_err"] += 1
                stat[pred_key]["fp"] += 1
        for pi in kept_pis:
            if pi not in matched_p:
                stat[pair.preds[pi].cls_key]["fp"] += 1
        for gj, g in enumerate(pair.gts):
            if gj not in matched_g:
                stat[g.cls_key]["fn"] += 1
    return {k: dict(v) for k, v in stat.items()}


def recognize_counts_from_stat(
    stat: dict[str, dict[str, int]],
) -> dict[str, int]:
    """与 ``_competition_recognize_count`` 一致：鉴定>0 用报出正确匹配，鉴定=0 用 pred。"""
    out: dict[str, int] = {}
    for k, s in stat.items():
        gt_n = int(s.get("gt", 0))
        pred_n = int(s.get("pred", 0))
        out[k] = _competition_recognize_count(s, gt_n=gt_n, pred_n=pred_n)
    return out


def gt_counts_from_pairs(pairs: list[ImagePair]) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for pair in pairs:
        for g in pair.gts:
            out[g.cls_key] += 1
    return dict(out)


@dataclass(frozen=True)
class ClassDevMetrics:
    """与 predict_size_validate_lib 合并类表同一口径。"""

    total_dev: float
    report_rate: float
    acc_rate: float
    recall_gap: float
    fp_rate: float
    gt_n: int
    pred_n: int
    tp: int
    fp: int
    fn: int
    cls_err: int


def class_dev_metrics_from_stat(
    stat: dict[str, dict[str, int]], cls_key: str
) -> ClassDevMetrics:
    """
    总偏差率 = max(召回缺口, 误报率)；
    召回缺口=(FN+类型错)/标注；误报率=FP/预测；报出率=(TP+类型错)/标注；正确率=TP/标注。
    """
    s = stat.get(cls_key) or {}
    gt_n = int(s.get("gt", 0))
    pred_n = int(s.get("pred", 0))
    tp_n = int(s.get("tp", 0))
    ce_n = int(s.get("cls_err", 0))
    fn_n = int(s.get("fn", 0))
    fp_n = int(s.get("fp", 0))
    denom_gt = float(gt_n)
    denom_pred = float(pred_n)
    report_rate = (float(tp_n + ce_n) / denom_gt) if denom_gt > 0 else 0.0
    acc_rate = (float(tp_n) / denom_gt) if denom_gt > 0 else 0.0
    fp_rate = (float(fp_n) / denom_pred) if denom_pred > 0 else 0.0
    recall_gap = (float(fn_n + ce_n) / denom_gt) if denom_gt > 0 else 0.0
    return ClassDevMetrics(
        total_dev=max(recall_gap, fp_rate),
        report_rate=report_rate,
        acc_rate=acc_rate,
        recall_gap=recall_gap,
        fp_rate=fp_rate,
        gt_n=gt_n,
        pred_n=pred_n,
        tp=tp_n,
        fp=fp_n,
        fn=fn_n,
        cls_err=ce_n,
    )


def score_quad(
    prepared: list[PrecomputedImageMatch],
    thr: ThresholdQuad,
    *,
    focus: frozenset[str],
    ignore_classes: frozenset[str],
    min_insect_count: int,
    class_merge: dict[str, list[str]] | None = None,
    label_alias_map: dict[str, str] | None = None,
    tier_equivalence: ClassTierEquivalence | None = None,
    class_overrides: dict[str, ClassConfThr] | None = None,
) -> tuple[float, dict[str, int], dict[str, dict[str, int]]]:
    raw_stat = build_stat_by_cls_at(
        prepared, thr, class_overrides=class_overrides
    )
    stat = merge_stat_by_cls(
        raw_stat,
        merge=class_merge,
        label_alias_map=label_alias_map,
        tier_equivalence=tier_equivalence,
    )
    summary = compute_competition_counting_summary(
        stat,
        focus,
        run_model="shengchan",
        ignore_classes=ignore_classes,
        min_insect_count=min_insect_count,
    )
    acc = float(summary.avg_accuracy_percent) if summary is not None else 0.0
    return acc, recognize_counts_from_stat(stat), stat


def iter_grid(lo: float, hi: float, step: float) -> list[float]:
    lo_c = int(round(lo * 100))
    hi_c = int(round(hi * 100))
    step_c = int(round(step * 100))
    if step_c <= 0:
        raise ValueError("GRID_STEP 须 > 0")
    return [c / 100.0 for c in range(lo_c, hi_c + 1, step_c)]


def choose_better(
    a: tuple[float, ThresholdQuad],
    b: tuple[float, ThresholdQuad],
) -> tuple[float, ThresholdQuad]:
    if b[0] > a[0]:
        return b
    if b[0] < a[0]:
        return a
    if b[1].sum_thr() < a[1].sum_thr():
        return b
    return a


def choose_better_class_dev(
    a: tuple[ClassDevMetrics, ClassConfThr],
    b: tuple[ClassDevMetrics, ClassConfThr],
) -> tuple[ClassDevMetrics, ClassConfThr]:
    """总偏差率升序 → 报出率降序 → 正确率降序 → 门限和升序。"""
    ma, ta = a
    mb, tb = b
    if mb.total_dev < ma.total_dev:
        return b
    if mb.total_dev > ma.total_dev:
        return a
    if mb.report_rate > ma.report_rate:
        return b
    if mb.report_rate < ma.report_rate:
        return a
    if mb.acc_rate > ma.acc_rate:
        return b
    if mb.acc_rate < ma.acc_rate:
        return a
    if tb.sum_thr() < ta.sum_thr():
        return b
    return a


def eligible_per_class_keys(
    gt_counts: dict[str, int],
    *,
    focus: frozenset[str],
    ignore_classes: frozenset[str],
    conf_by_classes_count: int,
) -> list[str]:
    """鉴定数 ``> conf_by_classes_count`` 且在 focus、非忽略的类。"""
    keys: list[str] = []
    for key, gt_n in gt_counts.items():
        if int(gt_n) <= int(conf_by_classes_count):
            continue
        if key not in focus:
            continue
        if is_eval_ignored_class(key, ignore_classes, None):
            continue
        keys.append(key)
    keys.sort(key=lambda k: (-int(gt_counts[k]), k))
    return keys


def search_global_quad(
    prepared: list[PrecomputedImageMatch],
    *,
    grid: list[float],
    baseline: ThresholdQuad,
    focus: frozenset[str],
    ignore_classes: frozenset[str],
    min_insect_count: int,
    class_merge: dict[str, list[str]] | None,
    label_alias_map: dict[str, str] | None,
    tier_equivalence: ClassTierEquivalence | None,
) -> tuple[list[tuple[float, ThresholdQuad]], tuple[float, ThresholdQuad]]:
    """4D 体型门限网格；返回 (ranked, best)。"""
    base_score, _, _ = score_quad(
        prepared,
        baseline,
        focus=focus,
        ignore_classes=ignore_classes,
        min_insect_count=min_insect_count,
        class_merge=class_merge,
        label_alias_map=label_alias_map,
        tier_equivalence=tier_equivalence,
    )
    combos = [
        ThresholdQuad(ds, db, cs, cb)
        for ds, db, cs, cb in itertools.product(grid, repeat=4)
    ]
    ranked: list[tuple[float, ThresholdQuad]] = []
    best: tuple[float, ThresholdQuad] = (base_score, baseline)
    for thr in combos:
        acc, _, _ = score_quad(
            prepared,
            thr,
            focus=focus,
            ignore_classes=ignore_classes,
            min_insect_count=min_insect_count,
            class_merge=class_merge,
            label_alias_map=label_alias_map,
            tier_equivalence=tier_equivalence,
        )
        ranked.append((acc, thr))
        best = choose_better(best, (acc, thr))
    ranked.sort(key=lambda x: (-x[0], x[1].sum_thr()))
    return ranked, best


def search_per_class_conf(
    prepared: list[PrecomputedImageMatch],
    *,
    grid: list[float],
    base_thr: ThresholdQuad,
    class_keys: list[str],
    class_merge: dict[str, list[str]] | None,
    label_alias_map: dict[str, str] | None,
    tier_equivalence: ClassTierEquivalence | None,
    size_class_by_name: dict[str, str] | None = None,
) -> dict[str, tuple[ClassDevMetrics, ClassConfThr]]:
    """
    对各类独立搜 (detect, cls)；其余框沿用 ``base_thr`` 体型门限。
    目标：该类总偏差率最低（同评估日志：max(召回缺口, 误报率)）。
    返回 cls_key -> (metrics, thr)。
    """
    results: dict[str, tuple[ClassDevMetrics, ClassConfThr]] = {}
    pairs = list(itertools.product(grid, repeat=2))

    def _seed_thr(cls_key: str) -> ClassConfThr:
        size = (size_class_by_name or {}).get(cls_key)
        if size == "small":
            return ClassConfThr(base_thr.detect_small, base_thr.cls_small)
        return ClassConfThr(base_thr.detect_big, base_thr.cls_big)

    def _eval(cls_key: str, ov: dict[str, ClassConfThr] | None) -> ClassDevMetrics:
        raw = build_stat_by_cls_at(prepared, base_thr, class_overrides=ov)
        stat = merge_stat_by_cls(
            raw,
            merge=class_merge,
            label_alias_map=label_alias_map,
            tier_equivalence=tier_equivalence,
        )
        return class_dev_metrics_from_stat(stat, cls_key)

    for cls_key in class_keys:
        seed = _seed_thr(cls_key)
        base_m = _eval(cls_key, None)
        best: tuple[ClassDevMetrics, ClassConfThr] = (base_m, seed)
        for det_v, cls_v in pairs:
            m = _eval(cls_key, {cls_key: ClassConfThr(det_v, cls_v)})
            best = choose_better_class_dev(best, (m, ClassConfThr(det_v, cls_v)))
        final_m = _eval(cls_key, {cls_key: best[1]})
        results[cls_key] = (final_m, best[1])
        print(
            f"  [{cls_key}] gt={final_m.gt_n} pred={final_m.pred_n} "
            f"最优 detect={best[1].detect:.2f} cls={best[1].cls:.2f} "
            f"总偏差={final_m.total_dev*100:.2f}% "
            f"(召回缺口={final_m.recall_gap*100:.2f}% 误报={final_m.fp_rate*100:.2f}% "
            f"报出={final_m.report_rate*100:.2f}% 正确={final_m.acc_rate*100:.2f}%) "
            f"| 基线总偏差={base_m.total_dev*100:.2f}%"
        )
    return results


def print_class_delta(
    *,
    gt_counts: dict[str, int],
    base_recognize: dict[str, int],
    best_recognize: dict[str, int],
    focus: frozenset[str],
    ignore_classes: frozenset[str],
    min_insect_count: int,
    display_index: dict[str, str],
    size_class_by_name: dict[str, str],
) -> None:
    rows: list[tuple[str, int, int, int, float, float, str]] = []
    keys = set(gt_counts) | set(base_recognize) | set(best_recognize)
    for key in keys:
        if key not in focus:
            continue
        if is_eval_ignored_class(key, ignore_classes, None):
            continue
        gt_n = int(gt_counts.get(key, 0))
        b = int(base_recognize.get(key, 0))
        n = int(best_recognize.get(key, 0))
        if 0 < gt_n < min_insect_count:
            continue
        if gt_n <= 0 and b <= 0 and n <= 0:
            continue
        acc_b = competition_counting_accuracy_percent(gt_n, b)
        acc_n = competition_counting_accuracy_percent(gt_n, n)
        size = size_class_by_name.get(key) or "-"
        rows.append((key, gt_n, b, n, acc_b, acc_n, size))
    rows.sort(key=lambda r: (r[5] - r[4], r[1]), reverse=True)
    print(
        "类别           | 体型  | 鉴定 | 基线正确 | 最优正确 | 基线准确率 | 最优准确率 | Δ分"
    )
    print("-" * 96)
    for key, gt_n, b, n, acc_b, acc_n, size in rows:
        label = (display_index.get(key) or key)[:14]
        delta = acc_n - acc_b
        print(
            f"{label:<14} | {size:<5} | {gt_n:>4} | {b:>8} | {n:>8} | "
            f"{acc_b:>8.2f}% | {acc_n:>8.2f}% | {delta:>+6.2f}"
        )


def export_combo_csv(
    path: Path,
    rows: list[tuple[float, ThresholdQuad]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "score",
                "detect_conf_small",
                "detect_conf_big",
                "cls_conf_small",
                "cls_conf_big",
            ],
        )
        w.writeheader()
        for i, (score, thr) in enumerate(rows, start=1):
            w.writerow(
                {
                    "rank": i,
                    "score": round(score, 4),
                    "detect_conf_small": thr.detect_small,
                    "detect_conf_big": thr.detect_big,
                    "cls_conf_small": thr.cls_small,
                    "cls_conf_big": thr.cls_big,
                }
            )


def print_1d_sweeps(
    *,
    prepared: list[PrecomputedImageMatch],
    base: ThresholdQuad,
    grid: list[float],
    focus: frozenset[str],
    ignore_classes: frozenset[str],
    min_insect_count: int,
    class_merge: dict[str, list[str]] | None = None,
    label_alias_map: dict[str, str] | None = None,
    tier_equivalence: ClassTierEquivalence | None = None,
) -> None:
    names = (
        ("detect_conf_small", "detect_small"),
        ("detect_conf_big", "detect_big"),
        ("cls_conf_small", "cls_small"),
        ("cls_conf_big", "cls_big"),
    )
    print("======== 一维扫描（其余门限钉在基线） ========")
    for label, field in names:
        parts: list[str] = []
        for v in grid:
            kw = {
                "detect_small": base.detect_small,
                "detect_big": base.detect_big,
                "cls_small": base.cls_small,
                "cls_big": base.cls_big,
            }
            kw[field] = v
            acc, _, _ = score_quad(
                prepared,
                ThresholdQuad(**kw),
                focus=focus,
                ignore_classes=ignore_classes,
                min_insect_count=min_insect_count,
                class_merge=class_merge,
                label_alias_map=label_alias_map,
                tier_equivalence=tier_equivalence,
            )
            parts.append(f"{v:.2f}={acc:.2f}")
        print(f"{label}: " + "  ".join(parts))


def print_near_optimal_plateau(
    ranked: list[tuple[float, ThresholdQuad]],
    *,
    best_score: float,
    delta: float,
    top_n: int,
) -> None:
    near = [(s, t) for s, t in ranked if s >= best_score - delta]
    print(
        f"======== 近最优平台（总分 ≥ {best_score - delta:.2f}，共 {len(near)} 组） ========"
    )
    if not near:
        return
    ds_vals = sorted({t.detect_small for _, t in near})
    db_vals = sorted({t.detect_big for _, t in near})
    cs_vals = sorted({t.cls_small for _, t in near})
    cb_vals = sorted({t.cls_big for _, t in near})
    print(
        f"detect_conf_small 取值: {', '.join(f'{v:.2f}' for v in ds_vals)}"
    )
    print(
        f"detect_conf_big 取值: {', '.join(f'{v:.2f}' for v in db_vals)}"
    )
    print(
        f"cls_conf_small 取值: {', '.join(f'{v:.2f}' for v in cs_vals)}"
    )
    print(
        f"cls_conf_big 取值: {', '.join(f'{v:.2f}' for v in cb_vals)}"
    )
    show = near[: max(top_n, 1)]
    print("rank | 总分   | d_small | d_big | c_small | c_big")
    for i, (acc, thr) in enumerate(show, start=1):
        print(
            f"{i:>4} | {acc:6.2f} | {thr.detect_small:7.2f} | "
            f"{thr.detect_big:5.2f} | {thr.cls_small:7.2f} | {thr.cls_big:5.2f}"
        )


def print_per_class_results(
    results: dict[str, tuple[ClassDevMetrics, ClassConfThr]],
    *,
    display_index: dict[str, str],
    size_class_by_name: dict[str, str],
) -> None:
    print("======== 单类最优门限（conf_by_classes，总偏差率最低） ========")
    print(
        "类别           | 体型  | 鉴定 | 预测 | 总偏差 | 召回缺口 | 误报率 | "
        "报出率 | 正确率 | detect | cls"
    )
    print("-" * 108)
    for key, (m, thr) in results.items():
        label = (display_index.get(key) or key)[:14]
        size = size_class_by_name.get(key) or "-"
        print(
            f"{label:<14} | {size:<5} | {m.gt_n:>4} | {m.pred_n:>4} | "
            f"{m.total_dev*100:6.2f}% | {m.recall_gap*100:7.2f}% | "
            f"{m.fp_rate*100:6.2f}% | {m.report_rate*100:6.2f}% | "
            f"{m.acc_rate*100:6.2f}% | {thr.detect:6.2f} | {thr.cls:5.2f}"
        )


def run_optimize(
    *,
    gt_dir: str | Path,
    pred_dir: str | Path,
    alg_json: Path,
    ignore_classes: list[str],
    min_insect_count: int,
    eval_insect_wildcard: bool,
    grid_lo: float,
    grid_hi: float,
    grid_step: float,
    baseline: ThresholdQuad,
    top_n: int,
    near_best_delta: float,
    output_csv: str | Path | None,
    conf_by_classes: bool,
    conf_by_classes_count: int,
) -> None:
    gt_root = Path(gt_dir).expanduser().resolve()
    pred_root = Path(pred_dir).expanduser().resolve()
    if not pred_root.is_dir():
        raise SystemExit(f"预测目录不存在: {pred_root}")

    alg = peek_composed_insect_alg_dict(alg_json)
    if not alg:
        raise SystemExit(f"未能加载算法配置: {alg_json}")
    alg["run_model"] = "shengchan"
    cls_cfg = _nested_cls_cfg(alg)
    generic_cls = cls_cfg.get("cls_conf")
    try:
        generic_cls_conf = float(generic_cls) if generic_cls is not None else None
    except (TypeError, ValueError):
        generic_cls_conf = None

    class_merge = build_eval_class_merge(
        None, insect_wildcard=eval_insect_wildcard
    )
    label_alias_map = load_eval_label_alias_map(alg_config=alg)
    tier_equivalence = load_class_tier_equivalence(alg_config=alg)
    size_class_by_name = build_class_size_class_index(alg)
    display_index = build_eval_class_display_index(alg_config=alg)
    competition_focus = expand_focus_for_stat_keys(
        resolve_competition_counting_focus(
            resolve_validation_focus_config(alg),
            class_merge=class_merge,
            label_alias_map=label_alias_map,
            tier_equivalence=tier_equivalence,
        ),
        class_merge=class_merge,
        label_alias_map=label_alias_map,
        tier_equivalence=tier_equivalence,
    )
    ignore_set = build_eval_focus_set(
        ignore_classes,
        merge=class_merge,
        label_alias_map=label_alias_map,
    )

    _, image_files = collect_images(str(gt_root))
    print(f"GT 目录: {gt_root}")
    print(f"预测 xml 目录: {pred_root}")
    print(
        f"网格: {grid_lo:.2f}~{grid_hi:.2f} step={grid_step:.2f}  "
        f"基线=({baseline.detect_small:.2f},{baseline.detect_big:.2f},"
        f"{baseline.cls_small:.2f},{baseline.cls_big:.2f})"
    )
    print(
        "规则: 检测 conf>=门限 保留；分类 conf>门限 否则不报出；"
        "识别数=报出框正确匹配(不含 filtered/other)；"
        "只能模拟不低于本次 XML 出框下限的门限"
    )
    if conf_by_classes:
        print(
            f"单类门限: conf_by_classes=ON  "
            f"鉴定数>{conf_by_classes_count} 的类单独搜 detect/cls"
            f"（目标=总偏差率 min）"
        )

    pairs, counters = load_gt_and_pred_boxes(
        gt_root=gt_root,
        pred_root=pred_root,
        image_files=image_files,
        alg=alg,
        class_merge=class_merge,
        label_alias_map=label_alias_map,
        ignore_classes=ignore_set,
        size_class_by_name=size_class_by_name,
        generic_cls_conf=generic_cls_conf,
        cls_cfg=cls_cfg,
        tier_equivalence=tier_equivalence,
    )
    gt_counts = gt_counts_from_pairs(pairs)
    all_preds = [p for pair in pairs for p in pair.preds]
    n_small = sum(1 for b in all_preds if b.size_class == "small")
    n_large = sum(1 for b in all_preds if b.size_class == "large")
    n_unk = sum(
        1 for b in all_preds if b.size_class not in ("small", "large")
    )
    print(
        f"配对图={counters.get('pred_xml', 0)} GT框={counters.get('gt_box', 0)} "
        f"预测框(未过滤)={counters.get('pred_box', 0)} "
        f"small={n_small} large={n_large} unknown={n_unk} "
        f"跳过无预测xml={counters.get('skip_no_pred_xml', 0)}"
    )
    if all_preds:
        print(
            f"XML 置信度范围 det=[{min(b.det_conf for b in all_preds):.3f},"
            f"{max(b.det_conf for b in all_preds):.3f}] "
            f"cls=[{min(b.cls_conf for b in all_preds):.3f},"
            f"{max(b.cls_conf for b in all_preds):.3f}]"
        )

    prepared = precompute_image_matches(
        pairs,
        class_merge=class_merge,
        label_alias_map=label_alias_map,
        tier_equivalence=tier_equivalence,
    )
    print(f"已预计算 IoU/类匹配矩阵: {len(prepared)} 图")

    base_score, base_recognize, _ = score_quad(
        prepared,
        baseline,
        focus=competition_focus,
        ignore_classes=ignore_set,
        min_insect_count=min_insect_count,
        class_merge=class_merge,
        label_alias_map=label_alias_map,
        tier_equivalence=tier_equivalence,
    )
    print(f"基线生产总分: {base_score:.2f}%")

    grid = iter_grid(grid_lo, grid_hi, grid_step)
    print(f"搜索组合数: {len(grid) ** 4}")
    ranked, best = search_global_quad(
        prepared,
        grid=grid,
        baseline=baseline,
        focus=competition_focus,
        ignore_classes=ignore_set,
        min_insect_count=min_insect_count,
        class_merge=class_merge,
        label_alias_map=label_alias_map,
        tier_equivalence=tier_equivalence,
    )
    best_score, best_thr = best
    _, best_recognize, best_stat = score_quad(
        prepared,
        best_thr,
        focus=competition_focus,
        ignore_classes=ignore_set,
        min_insect_count=min_insect_count,
        class_merge=class_merge,
        label_alias_map=label_alias_map,
        tier_equivalence=tier_equivalence,
    )
    print(
        "最优: "
        f"detect_conf_small={best_thr.detect_small:.2f} "
        f"detect_conf_big={best_thr.detect_big:.2f} "
        f"cls_conf_small={best_thr.cls_small:.2f} "
        f"cls_conf_big={best_thr.cls_big:.2f}  "
        f"总分={best_score:.2f}%  Δ={best_score - base_score:+.2f}"
    )
    print(f"======== Top{top_n} 组合 ========")
    print("rank | 总分   | d_small | d_big | c_small | c_big")
    for i, (acc, thr) in enumerate(ranked[:top_n], start=1):
        print(
            f"{i:>4} | {acc:6.2f} | {thr.detect_small:7.2f} | "
            f"{thr.detect_big:5.2f} | {thr.cls_small:7.2f} | {thr.cls_big:5.2f}"
        )

    print_near_optimal_plateau(
        ranked,
        best_score=best_score,
        delta=near_best_delta,
        top_n=top_n,
    )
    print_1d_sweeps(
        prepared=prepared,
        base=baseline,
        grid=grid,
        focus=competition_focus,
        ignore_classes=ignore_set,
        min_insect_count=min_insect_count,
        class_merge=class_merge,
        label_alias_map=label_alias_map,
        tier_equivalence=tier_equivalence,
    )

    print("======== 最优 vs 基线（鉴定≥下限或无鉴定误报、参与平均） ========")
    print_class_delta(
        gt_counts=gt_counts,
        base_recognize=base_recognize,
        best_recognize=best_recognize,
        focus=competition_focus,
        ignore_classes=ignore_set,
        min_insect_count=min_insect_count,
        display_index=display_index,
        size_class_by_name=size_class_by_name,
    )
    print("======== 最优组合完整计分表 ========")
    print_competition_counting_summary(
        compute_competition_counting_summary(
            best_stat,
            competition_focus,
            run_model="shengchan",
            ignore_classes=ignore_set,
            min_insect_count=min_insect_count,
        ),
        class_display_index=display_index,
    )

    per_class: dict[str, tuple[ClassDevMetrics, ClassConfThr]] = {}
    if conf_by_classes:
        targets = eligible_per_class_keys(
            gt_counts,
            focus=competition_focus,
            ignore_classes=ignore_set,
            conf_by_classes_count=conf_by_classes_count,
        )
        print(
            f"======== 单类门限搜索（{len(targets)} 类，"
            f"鉴定>{conf_by_classes_count}；目标=总偏差率最低） ========"
        )
        per_class = search_per_class_conf(
            prepared,
            grid=grid,
            base_thr=best_thr,
            class_keys=targets,
            class_merge=class_merge,
            label_alias_map=label_alias_map,
            tier_equivalence=tier_equivalence,
            size_class_by_name=size_class_by_name,
        )
        print_per_class_results(
            per_class,
            display_index=display_index,
            size_class_by_name=size_class_by_name,
        )

    csv_path = (
        Path(output_csv) if output_csv else pred_root / "size_conf_threshold_search.csv"
    )
    export_combo_csv(csv_path, ranked[:200])
    print(f"CSV(Top200): {csv_path}")
    print(
        "建议写入 insect_alg_shengchan.json: "
        f"detect_conf_small={best_thr.detect_small}, "
        f"detect_conf_big={best_thr.detect_big}, "
        f"cls_conf_small={best_thr.cls_small}, "
        f"cls_conf_big={best_thr.cls_big}"
    )
    if per_class:
        print("单类建议（out.<类名>.detect_conf / models.cls.cls_conf）:")
        for key, (m, thr) in per_class.items():
            label = display_index.get(key) or key
            print(
                f"  {label} ({key}): detect_conf={thr.detect}, cls_conf={thr.cls} "
                f"(总偏差={m.total_dev*100:.2f}%)"
            )


if __name__ == "__main__":
    # /Users/shunyaoyin/miniconda310/miniconda3/envs/yolo11/bin/python3 \\
    #   /Users/shunyaoyin/Documents/code/ai-company/insect/script/tools/optimize_size_conf_thresholds.py
    if sys.platform == "darwin":
        GT_DIR = "/Volumes/shunyao-h1/比赛数据/2026辛集正式比赛/田间比赛对比"
        PRED_DIR = (
            "/Volumes/shunyao-h1/比赛数据/2026辛集正式比赛/"
            "田间比赛对比-2.0.5-3.12.2-close"
        )
    else:
        GT_DIR = "/data/model/dataset/9月新/辛集市田间比赛"
        PRED_DIR = "/data/model/dataset/9月新/辛集市田间比赛-2.0.5-3.12.2-test"

    ALG_JSON = INSECT_ALG_SHENGCHAN_JSON
    IGNORE_CLASSES = ["划蝽", "德国蜉金龟"]
    MIN_INSECT_COUNT = 3
    EVAL_INSECT_WILDCARD = True
    GRID_LO = 0.30
    GRID_HI = 0.80
    GRID_STEP = 0.05
    BASELINE = ThresholdQuad(0.3, 0.3, 0.3, 0.3)
    TOP_N = 15
    NEAR_BEST_DELTA = 0.5
    OUTPUT_CSV: str | Path | None = None
    # 默认关：只做体型 4 门限全局搜；开则对鉴定数>阈值的类再单独搜 detect/cls
    # （单类目标=总偏差率最低，同评估日志 max(召回缺口,误报率)）
    CONF_BY_CLASSES = True
    CONF_BY_CLASSES_COUNT = 10

    run_optimize(
        gt_dir=GT_DIR,
        pred_dir=PRED_DIR,
        alg_json=ALG_JSON,
        ignore_classes=IGNORE_CLASSES,
        min_insect_count=MIN_INSECT_COUNT,
        eval_insect_wildcard=EVAL_INSECT_WILDCARD,
        grid_lo=GRID_LO,
        grid_hi=GRID_HI,
        grid_step=GRID_STEP,
        baseline=BASELINE,
        top_n=TOP_N,
        near_best_delta=NEAR_BEST_DELTA,
        output_csv=OUTPUT_CSV,
        conf_by_classes=CONF_BY_CLASSES,
        conf_by_classes_count=CONF_BY_CLASSES_COUNT,
    )
