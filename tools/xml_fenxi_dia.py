#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Detail  : 比对测试集 GT xml 与推理输出 xml，按类别统计正确/错误框的对角线长度分布。
#             对角线 = sqrt((x2-x1)^2 + (y2-y1)^2)，口径同 predict_size._box_diag_len。
#             输出：(1) 终端汇总表；(2) CSV xml_fenxi_dia_stats.csv。

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
from script.predict_size import PredictSize
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
from script.tools.xml_fenxi import (
    THRESHOLD_REF_MIN_SAMPLES,
    _gt_objects_to_rows,
    assess_threshold_reliability,
)


@dataclass
class DiaBucket:
    values: list[float] = field(default_factory=list)

    def add(self, diag: float) -> None:
        self.values.append(float(diag))

    @staticmethod
    def stats(values: list[float]) -> tuple[float | None, float | None, float | None]:
        if not values:
            return None, None, None
        return min(values), max(values), sum(values) / len(values)


@dataclass
class ClassDiaReport:
    cls_key: str
    correct: DiaBucket = field(default_factory=DiaBucket)
    wrong: DiaBucket = field(default_factory=DiaBucket)

    @property
    def n_correct(self) -> int:
        return len(self.correct.values)

    @property
    def n_wrong(self) -> int:
        return len(self.wrong.values)


def _box_diag_len(row: dict[str, Any]) -> float:
    return PredictSize._box_diag_len(row["x1"], row["y1"], row["x2"], row["y2"])


def classify_preds_on_image_dia(
    preds: list[dict[str, Any]],
    gts: list[dict[str, Any]],
    *,
    class_merge: dict[str, list[str]] | None,
    geom_metric: str,
    geom_threshold: float,
    label_alias_map: dict[str, str] | None,
    tier_equivalence: ClassTierEquivalence | None,
    fuzzy_only_wildcard: bool,
) -> list[tuple[str, float, str]]:
    """
    单图 pred 框分类为 correct / wrong，返回 (norm_cls, diag_px, tag) 列表。
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
    out: list[tuple[str, float, str]] = []
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
            out.append((cls_key, _box_diag_len(p), "fp"))
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
        diag = _box_diag_len(p)
        if i not in matched_pred_abs:
            out.append((cls_key, diag, "fp"))
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
            out.append((cls_key, diag, "tp"))
        else:
            out.append((cls_key, diag, "cls_err"))
    return out


def analyze_xml_pairs_dia(
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
) -> tuple[dict[str, ClassDiaReport], dict[str, int]]:
    """遍历图片，累积各类别正确/错误预测框的对角线长度。"""
    by_cls: dict[str, ClassDiaReport] = {}
    counters: dict[str, int] = defaultdict(int)

    def _bucket(cls_key: str) -> ClassDiaReport:
        if cls_key not in by_cls:
            by_cls[cls_key] = ClassDiaReport(cls_key=cls_key)
        return by_cls[cls_key]

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
        except (OSError, ValueError, ET.ParseError) as e:
            counters["skip_parse_error"] += 1
            print(f"[跳过] 解析失败 {rel}: {e}")
            continue
        if not gts:
            counters["skip_empty_gt"] += 1
            continue
        counters["paired"] += 1

        classified = classify_preds_on_image_dia(
            preds,
            gts,
            class_merge=class_merge,
            geom_metric=geom_metric,
            geom_threshold=geom_threshold,
            label_alias_map=label_alias_map,
            tier_equivalence=tier_equivalence,
            fuzzy_only_wildcard=fuzzy_only_wildcard,
        )
        for cls_key, diag, tag in classified:
            b = _bucket(cls_key)
            if tag == "tp":
                b.correct.add(diag)
                counters["tp"] += 1
            else:
                b.wrong.add(diag)
                counters[tag] += 1

    return by_cls, dict(counters)


def _fmt(v: float | None) -> str:
    if v is None:
        return "-"
    return f"{v:.2f}"


def print_dia_report(
    by_cls: dict[str, ClassDiaReport],
    *,
    class_display_index: dict[str, str] | None = None,
) -> None:
    idx = class_display_index or {}
    rows = sorted(
        by_cls.values(),
        key=lambda r: (-(r.n_correct + r.n_wrong), r.cls_key),
    )
    print(
        "类别           | 正确数 | dia_min | dia_max | dia_avg "
        "| 错误数 | dia_min | dia_max | dia_avg"
    )
    print("-" * 96)
    tot_correct = DiaBucket()
    tot_wrong = DiaBucket()
    for r in rows:
        label = (idx.get(r.cls_key) or r.cls_key)[:14]
        c_min, c_max, c_avg = DiaBucket.stats(r.correct.values)
        w_min, w_max, w_avg = DiaBucket.stats(r.wrong.values)
        tot_correct.values.extend(r.correct.values)
        tot_wrong.values.extend(r.wrong.values)
        print(
            f"{label:<14} | {r.n_correct:>6} | {_fmt(c_min):>7} | {_fmt(c_max):>7} | "
            f"{_fmt(c_avg):>7} | {r.n_wrong:>6} | {_fmt(w_min):>7} | {_fmt(w_max):>7} | "
            f"{_fmt(w_avg):>7}"
        )
    print("-" * 96)
    tc_min, tc_max, tc_avg = DiaBucket.stats(tot_correct.values)
    tw_min, tw_max, tw_avg = DiaBucket.stats(tot_wrong.values)
    print(
        f"{'合计':<14} | {len(tot_correct.values):>6} | {_fmt(tc_min):>7} | {_fmt(tc_max):>7} | "
        f"{_fmt(tc_avg):>7} | {len(tot_wrong.values):>6} | {_fmt(tw_min):>7} | {_fmt(tw_max):>7} | "
        f"{_fmt(tw_avg):>7}"
    )


def export_dia_report_csv(
    path: Path,
    by_cls: dict[str, ClassDiaReport],
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
        "correct_dia_min",
        "correct_dia_max",
        "correct_dia_avg",
        "n_wrong",
        "wrong_dia_min",
        "wrong_dia_max",
        "wrong_dia_avg",
        "reliability_code",
        "reliability_note",
    ]
    rows = sorted(by_cls.values(), key=lambda r: r.cls_key)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            c_min, c_max, c_avg = DiaBucket.stats(r.correct.values)
            w_min, w_max, w_avg = DiaBucket.stats(r.wrong.values)
            rel_code, rel_note = assess_threshold_reliability(
                r.n_correct, r.n_wrong, min_samples=min_samples
            )
            w.writerow(
                {
                    "cls_key": r.cls_key,
                    "display_name": idx.get(r.cls_key) or r.cls_key,
                    "n_correct": r.n_correct,
                    "correct_dia_min": c_min,
                    "correct_dia_max": c_max,
                    "correct_dia_avg": c_avg,
                    "n_wrong": r.n_wrong,
                    "wrong_dia_min": w_min,
                    "wrong_dia_max": w_max,
                    "wrong_dia_avg": w_avg,
                    "reliability_code": rel_code,
                    "reliability_note": rel_note,
                }
            )


if __name__ == "__main__":
    # ----------------------- 需要你改的参数 -----------------------
    GT_DIR = "/Volumes/shunyao-h1/训练数据/测试集/模拟试卷"
    PRED_DIR = GT_DIR + "-3.12.1-3.8.11"

    GT_DIR = "/Volumes/shunyao-h1/训练数据/北京比赛/模拟摆拍试卷"
    PRED_DIR = "/Volumes/shunyao-h1/训练数据/北京比赛/模拟摆拍试卷-3.13.3-3.8.11-800"
    VAL_BOX_MATCH_METRIC = "iou"  # "iou" | "ior"
    VAL_GEOM_THRESHOLD = 0.25
    INSECT_ALG_ALL_JSON: str | Path | None = DEFAULT_INSECT_ALG_ALL_JSON
    CLASS_MERGE_TO_GROUPS: dict[str, list[str]] | None = None
    EVAL_INSECT_WILDCARD = True
    EVAL_INSECT_WILDCARD_STRICT = True
    OUTPUT_CSV: str | Path | None = None
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
        "对角线=预测框 sqrt(w^2+h^2)，按预测类名归并统计"
    )

    by_cls, counters = analyze_xml_pairs_dia(
        gt_root=gt_root,
        pred_root=pred_root,
        image_files=image_files,
        class_merge=class_merge,
        geom_metric=VAL_BOX_MATCH_METRIC,
        geom_threshold=VAL_GEOM_THRESHOLD,
        label_alias_map=label_alias_map,
        tier_equivalence=tier_equivalence,
        fuzzy_only_wildcard=fuzzy_only_wildcard,
    )

    print(
        f"配对={counters.get('paired', 0)} "
        f"TP={counters.get('tp', 0)} "
        f"类型错={counters.get('cls_err', 0)} "
        f"多检={counters.get('fp', 0)} "
        f"跳过(无GT xml)={counters.get('skip_no_gt_xml', 0)} "
        f"跳过(无预测xml)={counters.get('skip_no_pred_xml', 0)}"
    )
    print("======== 各类别对角线统计(px) ========")
    print_dia_report(by_cls, class_display_index=display_index)

    csv_path = Path(OUTPUT_CSV) if OUTPUT_CSV else pred_root / "xml_fenxi_dia_stats.csv"
    export_dia_report_csv(
        csv_path,
        by_cls,
        class_display_index=display_index,
        min_samples=THRESHOLD_REF_MIN_SAMPLES,
    )
    print(f"CSV: {csv_path}")
