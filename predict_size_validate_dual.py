#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Detail  : 同时调用小虫(稻飞虱)模型 + 大虫模型做 VOC 验证；输出 small/large/combined 三套结果与汇总统计
# 标准评估模式：在仍使用本脚本几何匹配 + 类别归并的前提下，累积混淆矩阵并导出 per-class 精度/召回等
# （输出形态参考 Ultralytics YOLO val，如 train/train_detect/train-cfg/val_v11_big.py 中的 model.val(project=...)）
import csv
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET
import unicodedata

import cv2

_FILE = Path(__file__).resolve()
_ROOT = _FILE.parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from script.predict_size import PredictSize, write_pascal_voc_xml
from script.predict_size_validate import (
    parse_pascal_voc_objects,
    match_pred_gt,
    normalize_class_name,
    is_class_match,
    is_metric_ignored_other,
    draw_main_output_image,
    box_iou,
    match_pred_gt_ior,
)


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


def _save_image_and_xml(
    image_bgr,
    all_final_rows: list[dict],
    results_visible: list[dict],
    *,
    out_dir: str,
    rel_path: Path,
    clip_size: int,
    overlap_size: int,
    predict_debug: bool,
    label_mode: str,
    val_xml_mode: bool,
    draw_boxes: bool,
    draw_center_point_and_label: bool,
    gts: list[dict] | None,
    matches: list[tuple[int, int, float]] | None,
    matched_p: set[int] | None,
    class_merge_to_groups: dict[str, list[str]] | None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    h_img, w_img = image_bgr.shape[:2]

    save_path = os.path.join(out_dir, rel_path.name)
    need_visual = bool(draw_boxes or draw_center_point_and_label)
    if not need_visual:
        # 关闭可视化：不画框/不写文字，但仍然把原图保存到输出目录（覆盖旧画框图）
        cv2.imwrite(save_path, image_bgr)
    else:
        if draw_boxes:
            if val_xml_mode and gts is not None and matches is not None and matched_p is not None:
                img_draw = draw_main_output_image(
                    image_bgr,
                    all_final_rows,
                    clip_size,
                    overlap_size,
                    predict_debug,
                    label_mode=label_mode,
                    val_xml_mode=True,
                    results_visible=results_visible,
                    gts=gts,
                    matches=matches,
                    matched_p=matched_p,
                    merge=class_merge_to_groups,
                )
            else:
                img_draw = draw_main_output_image(
                    image_bgr,
                    all_final_rows,
                    clip_size,
                    overlap_size,
                    predict_debug,
                    label_mode=label_mode,
                    val_xml_mode=False,
                    results_visible=results_visible,
                    gts=[],
                    matches=[],
                    matched_p=set(),
                    merge=class_merge_to_groups,
                )
        else:
            # 不画框，但仍需额外画中心点/标签
            img_draw = image_bgr.copy()

        if draw_center_point_and_label:
            h_img, w_img = img_draw.shape[:2]
            for r in results_visible or []:
                try:
                    x1 = int(round(float(r.get("x1", 0) or 0)))
                    y1 = int(round(float(r.get("y1", 0) or 0)))
                    x2 = int(round(float(r.get("x2", 0) or 0)))
                    y2 = int(round(float(r.get("y2", 0) or 0)))
                except Exception:
                    continue
                cx = int(round((x1 + x2) / 2))
                cy = int(round((y1 + y2) / 2))
                if cx < 0 or cy < 0 or cx >= w_img or cy >= h_img:
                    continue

                name = str(r.get("cls_name", "") or "").strip()
                conf_val = r.get("cls_conf", None)
                if conf_val is None:
                    conf_val = r.get("conf", 0.0)
                try:
                    conf_f = float(conf_val or 0.0)
                except Exception:
                    conf_f = 0.0
                label = f"{name} {conf_f:.2f}".strip()

                cv2.circle(img_draw, (cx, cy), 3, (0, 255, 255), thickness=-1, lineType=cv2.LINE_AA)

                if label:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.45
                    thickness = 1
                    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                    tx = int(cx - tw / 2)
                    ty = int(cy - 6)
                    tx = max(0, min(tx, w_img - tw - 1))
                    ty = max(th + baseline + 1, min(ty, h_img - 1))
                    cv2.putText(
                        img_draw,
                        label,
                        (tx, ty),
                        font,
                        font_scale,
                        (0, 0, 0),
                        thickness=3,
                        lineType=cv2.LINE_AA,
                    )
                    cv2.putText(
                        img_draw,
                        label,
                        (tx, ty),
                        font,
                        font_scale,
                        (255, 255, 255),
                        thickness=1,
                        lineType=cv2.LINE_AA,
                    )

        cv2.imwrite(save_path, img_draw)

    depth = 3 if image_bgr.ndim >= 3 else 1
    xml_name = Path(rel_path.name).stem + ".xml"
    xml_path = os.path.join(out_dir, xml_name)
    write_pascal_voc_xml(
        xml_path,
        folder_name=os.path.basename(os.path.normpath(out_dir)) or "",
        image_filename=rel_path.name,
        width=w_img,
        height=h_img,
        depth=depth,
        results=results_visible,
    )


def _is_daofeishi_like(name: str) -> bool:
    """
    稻飞虱相关：标注/预测可能出现的若干历史写法做兼容。
    """
    n = str(name or "").strip().lower()
    if not n:
        return False
    # 项目里常见：daofeishi；以及 hefeishi/baibeifeishi/huifeishi 等（有时会被映射/修正）
    if n in {"daofeishi", "hefeishi", "baibeifeishi", "baifeifeishi", "huifeishi"}:
        return True
    return False


def _subset_by_gt(
    preds: list[dict],
    gts: list[dict],
    *,
    keep_gt_fn,
) -> tuple[list[dict], list[dict]]:
    g_keep = [g for g in gts if keep_gt_fn(g.get("name", ""))]
    if not g_keep:
        return [], []
    # preds 不做强行过滤（因为同一模型可能也会误报到该子集外类别），
    # 评估时只对 g_keep 做匹配即可；多余 preds 会自然落到 FP。
    return preds, g_keep


def _disp_w(s: str) -> int:
    # 终端显示宽度：中日韩宽字符按 2 计
    w = 0
    for ch in str(s):
        if unicodedata.east_asian_width(ch) in ("W", "F"):
            w += 2
        else:
            w += 1
    return w


def _ljust_disp(s: str, width: int) -> str:
    pad = max(0, width - _disp_w(s))
    return str(s) + (" " * pad)


def _rjust_disp(s: str, width: int) -> str:
    pad = max(0, width - _disp_w(s))
    return (" " * pad) + str(s)


def _print_stat_by_cls(
    title: str,
    stat_by_cls: dict[str, dict[str, int]],
    *,
    sort_by_acc: bool = True,
) -> None:
    if not stat_by_cls:
        return
    headers = [
        "类别(归一)",
        "标注数",
        "预测数",
        "正确TP",
        "报出率",
        "类型错",
        "正确率",
        "漏检FN",
        "漏检率",
        "类错率",
        "多检FP",
        "误报率",
        "总偏差率",
    ]

    # sort keys: total_dev_rate asc -> report_rate desc -> acc_rate desc
    rows_with_sort: list[tuple[str, float, float, float, list[str]]] = []
    total = {"gt": 0, "pred": 0, "tp": 0, "cls_err": 0, "fn": 0, "fp": 0}
    for cls_name in stat_by_cls.keys():
        s = stat_by_cls[cls_name]
        gt_n = int(s.get("gt", 0))
        pred_n = int(s.get("pred", 0))
        tp_n = int(s.get("tp", 0))
        ce_n = int(s.get("cls_err", 0))
        fn_n = int(s.get("fn", 0))
        fp_n = int(s.get("fp", 0))

        denom_gt = float(gt_n)
        denom_matched = float(tp_n + ce_n)
        denom_pred = float(pred_n)
        report_rate = (float(tp_n) / denom_gt) if denom_gt > 0 else 0.0
        acc_rate = (float(tp_n) / denom_matched) if denom_matched > 0 else 0.0
        fp_rate = (float(fp_n) / denom_pred) if denom_pred > 0 else 0.0
        # 漏检率：仅「几何上无任何匹配框」的 GT 占比（与类型错互斥，不去重相加）
        miss_fn_rate = (float(fn_n) / denom_gt) if denom_gt > 0 else 0.0
        # 类错率：有框但分错（相对标注数）
        cls_err_rate = (float(ce_n) / denom_gt) if denom_gt > 0 else 0.0
        # 召回缺口(相对标注) = 漏检 + 类型错，与误报率分母不同；总偏差率取 max 避免简单相加超过 100%
        recall_gap = (float(fn_n + ce_n) / denom_gt) if denom_gt > 0 else 0.0
        total_dev_rate = max(recall_gap, fp_rate)

        row = [
            str(cls_name),
            str(gt_n),
            str(pred_n),
            str(tp_n),
            f"{report_rate*100:.2f}%",
            str(ce_n),
            f"{acc_rate*100:.2f}%",
            str(fn_n),
            f"{miss_fn_rate*100:.2f}%",
            f"{cls_err_rate*100:.2f}%",
            str(fp_n),
            f"{fp_rate*100:.2f}%",
            f"{total_dev_rate*100:.2f}%",
        ]
        rows_with_sort.append((str(cls_name), float(total_dev_rate), float(report_rate), float(acc_rate), row))
        total["gt"] += gt_n
        total["pred"] += pred_n
        total["tp"] += tp_n
        total["cls_err"] += ce_n
        total["fn"] += fn_n
        total["fp"] += fp_n

    if sort_by_acc:
        rows_with_sort.sort(
            key=lambda x: (
                x[1],  # total_dev_rate asc
                -x[2],  # report_rate desc
                -x[3],  # acc_rate desc
                -int(stat_by_cls.get(x[0], {}).get("gt", 0)),  # tie-breaker: support desc
                x[0],
            )
        )
    else:
        rows_with_sort.sort(key=lambda x: x[0])
    rows = [r for _cls, _dev, _rep, _acc, r in rows_with_sort]

    tg = int(total["gt"])
    tpr = int(total["pred"])
    ttp = int(total["tp"])
    tce = int(total["cls_err"])
    tfn = int(total["fn"])
    tfp = int(total["fp"])
    dgt = float(tg)
    dpr = float(tpr)
    sum_miss_fn = (float(tfn) / dgt) if dgt > 0 else 0.0
    sum_cls_err_r = (float(tce) / dgt) if dgt > 0 else 0.0
    sum_recall_gap = (float(tfn + tce) / dgt) if dgt > 0 else 0.0
    sum_fp_r = (float(tfp) / dpr) if dpr > 0 else 0.0
    sum_total_dev = max(sum_recall_gap, sum_fp_r)

    all_lines = [headers] + rows + [
        [
            "合计",
            str(tg),
            str(tpr),
            str(ttp),
            "",
            str(tce),
            "",
            str(tfn),
            f"{sum_miss_fn*100:.2f}%",
            f"{sum_cls_err_r*100:.2f}%",
            str(tfp),
            f"{sum_fp_r*100:.2f}%",
            f"{sum_total_dev*100:.2f}%",
        ]
    ]
    widths = [0] * len(headers)
    for line in all_lines:
        for i, cell in enumerate(line):
            widths[i] = max(widths[i], _disp_w(str(cell)))

    def _fmt_line(items: list[str]) -> str:
        out = []
        for i, it in enumerate(items):
            if i == 0:
                out.append(_ljust_disp(it, widths[i]))
            else:
                out.append(_rjust_disp(it, widths[i]))
        return " | ".join(out)

    sort_hint = (
        "总偏差率升序(max(召回缺口,误报率))，其次报出率降序，再次正确率降序；"
        "漏检率=FN/标注，类错率=类型错/标注，与多检误报分列"
        if sort_by_acc
        else "标注类别名升序"
    )
    print(f"{title}（按{sort_hint}）:")
    print(_fmt_line(headers))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(_fmt_line([str(x) for x in r]))
    print("-+-".join("-" * w for w in widths))
    print(_fmt_line(all_lines[-1]))


def _row_output_suppressed(
    r: dict,
    suppress: frozenset[str] | set[str],
    class_merge_to_groups: dict[str, list[str]] | None,
) -> bool:
    """
    输出暂隐：cls_name 与配置集合比对时同时支持「预测原始名」与 normalize_class_name 后的组名。
    """
    if not suppress:
        return False
    raw = str(r.get("cls_name", "") or "").strip()
    if not raw:
        return False
    if raw in suppress:
        return True
    norm = normalize_class_name(raw, class_merge_to_groups)
    return bool(norm) and norm in suppress


def _apply_output_suppress(
    results: list[dict],
    all_rows: list[dict],
    suppress: frozenset[str],
    class_merge_to_groups: dict[str, list[str]] | None,
) -> tuple[list[dict], list[dict]]:
    if not suppress:
        return results, all_rows
    return (
        [r for r in results if not _row_output_suppressed(r, suppress, class_merge_to_groups)],
        [r for r in all_rows if not _row_output_suppressed(r, suppress, class_merge_to_groups)],
    )


def _apply_output_suppress_combined(
    results: list[dict],
    all_rows: list[dict],
    suppress: frozenset[str],
    merge_small: dict[str, list[str]] | None,
    merge_large: dict[str, list[str]] | None,
) -> tuple[list[dict], list[dict]]:
    if not suppress:
        return results, all_rows

    def _merge_for(r: dict) -> dict[str, list[str]] | None:
        return merge_small if r.get("model_src") == "small" else merge_large

    return (
        [r for r in results if not _row_output_suppressed(r, suppress, _merge_for(r))],
        [r for r in all_rows if not _row_output_suppressed(r, suppress, _merge_for(r))],
    )


def _apply_focus_filter(
    results: list[dict],
    all_rows: list[dict],
    focus: frozenset[str],
    class_merge_to_groups: dict[str, list[str]] | None,
) -> tuple[list[dict], list[dict]]:
    """
    只保留关注列表中的虫子（用于“统计/分析/画图/写xml”全链路）。

    focus 内同时支持：
    - 原始类别名（如 xml/pred 里的 name/cls_name）
    - normalize_class_name 后的归一名（与 CLASS_MERGE_TO_GROUPS_* 的组名一致）
    """
    if not focus:
        return results, all_rows

    def _keep_name(raw_name: str) -> bool:
        raw = str(raw_name or "").strip()
        if not raw:
            return False
        if raw in focus:
            return True
        norm = normalize_class_name(raw, class_merge_to_groups)
        return bool(norm) and norm in focus

    def _keep_row(r: dict) -> bool:
        return _keep_name(str(r.get("cls_name", "") or ""))

    return ([r for r in results if _keep_row(r)], [r for r in all_rows if _keep_row(r)])


def _apply_focus_filter_combined(
    results: list[dict],
    all_rows: list[dict],
    focus: frozenset[str],
    merge_small: dict[str, list[str]] | None,
    merge_large: dict[str, list[str]] | None,
) -> tuple[list[dict], list[dict]]:
    if not focus:
        return results, all_rows

    def _merge_for(r: dict) -> dict[str, list[str]] | None:
        return merge_small if r.get("model_src") == "small" else merge_large

    def _keep_row(r: dict) -> bool:
        raw = str(r.get("cls_name", "") or "").strip()
        if not raw:
            return False
        if raw in focus:
            return True
        norm = normalize_class_name(raw, _merge_for(r))
        return bool(norm) and norm in focus

    return ([r for r in results if _keep_row(r)], [r for r in all_rows if _keep_row(r)])


# 混淆矩阵：行=真实类别(GT)，列=预测类别；与 YOLO val 常见约定一致地增加「无检测」「多余预测」虚拟类
_STD_EVAL_BG_COL = "__no_pred__"  # FN：该 GT 未与任一预测框匹配
_STD_EVAL_BG_ROW = "__extra_pred__"  # FP：该预测未与任一 GT 匹配


def _std_eval_collect_confusion(
    cm: dict[tuple[str, str], int],
    preds: list[dict],
    gts: list[dict] | None,
    matches: list[tuple[int, int, float]] | None,
    matched_p: set[int] | None,
    class_merge_to_groups: dict[str, list[str]] | None,
) -> None:
    """
    按几何匹配结果更新 (gt_norm, pred_norm) 计数；未匹配 GT 记入 (gt, __no_pred__)，未匹配 pred 记入 (__extra_pred__, pred)。
    ``other`` 类不参与混淆矩阵与 per-class 指标（与 _eval_one 一致）。
    """
    if gts is None:
        return
    mlist = list(matches or [])
    m_p = matched_p if matched_p is not None else set()
    if not gts:
        for p in preds:
            if is_metric_ignored_other(str(p.get("cls_name", "") or ""), class_merge_to_groups):
                continue
            pn = normalize_class_name(str(p.get("cls_name", "") or ""), class_merge_to_groups) or "?"
            cm[(_STD_EVAL_BG_ROW, pn)] += 1
        return
    matched_g = {int(j) for _, j, _ in mlist}
    for i, j, _ in mlist:
        if is_metric_ignored_other(str(preds[int(i)].get("cls_name", "") or ""), class_merge_to_groups):
            continue
        if is_metric_ignored_other(str(gts[int(j)].get("name", "") or ""), class_merge_to_groups):
            continue
        gn = normalize_class_name(str(gts[int(j)].get("name", "") or ""), class_merge_to_groups) or "?"
        pn = normalize_class_name(str(preds[int(i)].get("cls_name", "") or ""), class_merge_to_groups) or "?"
        cm[(gn, pn)] += 1
    for pi in range(len(preds)):
        if pi not in m_p:
            if is_metric_ignored_other(str(preds[pi].get("cls_name", "") or ""), class_merge_to_groups):
                continue
            pn = normalize_class_name(str(preds[pi].get("cls_name", "") or ""), class_merge_to_groups) or "?"
            cm[(_STD_EVAL_BG_ROW, pn)] += 1
    for gj in range(len(gts)):
        if gj not in matched_g:
            if is_metric_ignored_other(str(gts[gj].get("name", "") or ""), class_merge_to_groups):
                continue
            gn = normalize_class_name(str(gts[gj].get("name", "") or ""), class_merge_to_groups) or "?"
            cm[(gn, _STD_EVAL_BG_COL)] += 1


def _std_eval_labels_from_cm(cm: dict[tuple[str, str], int]) -> list[str]:
    rows = {a for (a, _) in cm.keys()}
    cols = {b for (_, b) in cm.keys()}
    ordered = [x for x in sorted(rows | cols) if x not in (_STD_EVAL_BG_ROW, _STD_EVAL_BG_COL)]
    # 虚拟行列放末尾，便于阅读
    if _STD_EVAL_BG_ROW in rows or _STD_EVAL_BG_ROW in cols:
        ordered.append(_STD_EVAL_BG_ROW)
    if _STD_EVAL_BG_COL in rows or _STD_EVAL_BG_COL in cols:
        ordered.append(_STD_EVAL_BG_COL)
    return ordered


def _std_eval_matrix_and_metrics(
    cm: dict[tuple[str, str], int],
    *,
    labels: list[str],
) -> tuple[list[list[int]], list[dict], dict[str, float]]:
    """返回 count 矩阵、每类指标行、整体 micro 指标。"""
    n = len(labels)
    idx = {lab: i for i, lab in enumerate(labels)}
    mat = [[0] * n for _ in range(n)]
    for (a, b), c in cm.items():
        if a not in idx or b not in idx:
            continue
        mat[idx[a]][idx[b]] += int(c)

    per_rows: list[dict] = []
    tp_sum = fp_micro = fn_micro = 0
    for lab in labels:
        if lab in (_STD_EVAL_BG_ROW, _STD_EVAL_BG_COL):
            continue
        i = idx[lab]
        tp = mat[i][i]
        fn = sum(mat[i][k] for k in range(n) if k != i)
        fp = sum(mat[k][i] for k in range(n) if k != i)
        support = tp + fn
        pred_pos = tp + fp
        rec = (tp / support) if support > 0 else 0.0
        prec = (tp / pred_pos) if pred_pos > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_rows.append(
            {
                "class": lab,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "support_gt": support,
                "support_pred": pred_pos,
                "precision": round(prec, 6),
                "recall": round(rec, 6),
                "f1": round(f1, 6),
            }
        )
        tp_sum += tp
        fp_micro += fp
        fn_micro += fn

    denom_pr = tp_sum + fp_micro
    denom_re = tp_sum + fn_micro
    overall = {
        "micro_precision": (tp_sum / denom_pr) if denom_pr > 0 else 0.0,
        "micro_recall": (tp_sum / denom_re) if denom_re > 0 else 0.0,
        "macro_precision": (
            sum(r["precision"] for r in per_rows) / len(per_rows) if per_rows else 0.0
        ),
        "macro_recall": (sum(r["recall"] for r in per_rows) / len(per_rows) if per_rows else 0.0),
        "macro_f1": (sum(r["f1"] for r in per_rows) / len(per_rows) if per_rows else 0.0),
    }
    if overall["micro_precision"] + overall["micro_recall"] > 0:
        p, r = overall["micro_precision"], overall["micro_recall"]
        overall["micro_f1"] = 2 * p * r / (p + r)
    else:
        overall["micro_f1"] = 0.0
    for k in ("micro_precision", "micro_recall", "micro_f1", "macro_precision", "macro_recall", "macro_f1"):
        overall[k] = round(float(overall[k]), 6)

    return mat, per_rows, overall


def _std_eval_save_branch(
    out_root: str,
    branch: str,
    cm: dict[tuple[str, str], int],
    stat_block: dict[str, int],
    meta: dict,
) -> None:
    branch_dir = os.path.join(out_root, branch)
    os.makedirs(branch_dir, exist_ok=True)
    labels = _std_eval_labels_from_cm(cm)
    mat, per_rows, overall = _std_eval_matrix_and_metrics(cm, labels=labels)

    csv_path = os.path.join(branch_dir, "confusion_matrix.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gt\\pred"] + labels)
        for i, row_lab in enumerate(labels):
            w.writerow([row_lab] + [str(mat[i][j]) for j in range(len(labels))])

    per_path = os.path.join(branch_dir, "per_class_precision_recall.csv")
    with open(per_path, "w", newline="", encoding="utf-8") as f:
        if per_rows:
            w = csv.DictWriter(f, fieldnames=list(per_rows[0].keys()))
            w.writeheader()
            w.writerows(per_rows)

    summary = {
        "branch": branch,
        "note": "VOC-style matching + normalized class names; __no_pred__=FN __extra_pred__=FP",
        "meta": meta,
        "aggregate_counts": {k: int(stat_block.get(k, 0)) for k in ("tp", "fp", "fn", "cls_err", "geom_pairs", "img_with_xml")},
        "overall_prf": overall,
        "confusion_labels": labels,
    }
    with open(os.path.join(branch_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if not labels:
        logging.warning("标准评估 %s: 无混淆计数，已跳过 heatmap", branch)
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        arr = np.array(mat, dtype=float)
        row_sums = arr.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        norm = arr / row_sums

        fig, axes = plt.subplots(1, 2, figsize=(max(10, len(labels) * 0.45), max(5, len(labels) * 0.38)))
        for ax, data, title in (
            (axes[0], arr, "counts"),
            (axes[1], norm, "row-normalized (recall diag)"),
        ):
            im = ax.imshow(data, interpolation="nearest", cmap=plt.cm.Blues)
            ax.set_title(title)
            tick_marks = range(len(labels))
            ax.set_xticks(tick_marks)
            ax.set_yticks(tick_marks)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
            ax.set_yticklabels(labels, fontsize=7)
            ax.set_ylabel("GT")
            ax.set_xlabel("Pred")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(f"Confusion matrix — {branch}")
        fig.tight_layout()
        fig.savefig(os.path.join(branch_dir, "confusion_matrix.png"), dpi=160)
        plt.close(fig)
    except Exception as e:
        logging.warning("跳过混淆矩阵图（需 matplotlib/numpy）: %s", e)


def _export_stat_by_cls_csv(
    out_root: str,
    branch: str,
    stat_by_cls: dict[str, dict[str, int]],
    *,
    sort_by_acc: bool = True,
) -> None:
    if not stat_by_cls:
        return
    branch_dir = os.path.join(out_root, branch)
    os.makedirs(branch_dir, exist_ok=True)

    headers = [
        "class_norm",
        "gt",
        "pred",
        "tp",
        "report_rate",
        "cls_err",
        "acc_rate",
        "fn",
        "miss_fn_rate",
        "cls_err_rate",
        "recall_gap",
        "fp",
        "fp_rate",
        "total_dev_rate",
    ]

    rows_with_sort: list[tuple[str, float, float, float, dict]] = []
    for cls_name in stat_by_cls.keys():
        s = stat_by_cls[cls_name]
        gt_n = int(s.get("gt", 0))
        pred_n = int(s.get("pred", 0))
        tp_n = int(s.get("tp", 0))
        ce_n = int(s.get("cls_err", 0))
        fn_n = int(s.get("fn", 0))
        fp_n = int(s.get("fp", 0))

        denom_gt = float(gt_n)
        denom_matched = float(tp_n + ce_n)
        denom_pred = float(pred_n)
        report_rate = (float(tp_n) / denom_gt) if denom_gt > 0 else 0.0
        acc_rate = (float(tp_n) / denom_matched) if denom_matched > 0 else 0.0
        fp_rate = (float(fp_n) / denom_pred) if denom_pred > 0 else 0.0
        miss_fn_rate = (float(fn_n) / denom_gt) if denom_gt > 0 else 0.0
        cls_err_rate = (float(ce_n) / denom_gt) if denom_gt > 0 else 0.0
        recall_gap = (float(fn_n + ce_n) / denom_gt) if denom_gt > 0 else 0.0
        total_dev_rate = max(recall_gap, fp_rate)

        row = {
            "class_norm": str(cls_name),
            "gt": gt_n,
            "pred": pred_n,
            "tp": tp_n,
            "report_rate": round(report_rate, 6),
            "cls_err": ce_n,
            "acc_rate": round(acc_rate, 6),
            "fn": fn_n,
            "miss_fn_rate": round(miss_fn_rate, 6),
            "cls_err_rate": round(cls_err_rate, 6),
            "recall_gap": round(recall_gap, 6),
            "fp": fp_n,
            "fp_rate": round(fp_rate, 6),
            "total_dev_rate": round(total_dev_rate, 6),
        }
        rows_with_sort.append((str(cls_name), float(total_dev_rate), float(report_rate), float(acc_rate), row))

    if sort_by_acc:
        rows_with_sort.sort(
            key=lambda x: (
                x[1],  # total_dev_rate asc
                -x[2],  # report_rate desc
                -x[3],  # acc_rate desc
                -int(stat_by_cls.get(x[0], {}).get("gt", 0)),  # support desc
                x[0],
            )
        )
    else:
        rows_with_sort.sort(key=lambda x: x[0])

    out_path = os.path.join(branch_dir, "stat_by_class.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for _cls, _dev, _rep, _acc, row in rows_with_sort:
            w.writerow(row)


def _export_group_summary_csv(
    out_root: str,
    branch: str,
    stat_by_cls: dict[str, dict[str, int]],
    *,
    c1_set: set[str],
    c2_set: set[str],
) -> None:
    if not stat_by_cls:
        return
    branch_dir = os.path.join(out_root, branch)
    os.makedirs(branch_dir, exist_ok=True)

    def _bucket(cls_norm: str) -> str:
        if cls_norm in c1_set:
            return "一类害虫"
        if cls_norm in c2_set:
            return "二类害虫"
        return "其他虫子"

    group_total: dict[str, dict[str, int]] = {
        "一类害虫": {"gt": 0, "pred": 0, "tp": 0, "cls_err": 0, "fn": 0, "fp": 0},
        "二类害虫": {"gt": 0, "pred": 0, "tp": 0, "cls_err": 0, "fn": 0, "fp": 0},
        "其他虫子": {"gt": 0, "pred": 0, "tp": 0, "cls_err": 0, "fn": 0, "fp": 0},
    }
    for cls_norm, s in stat_by_cls.items():
        b = _bucket(str(cls_norm))
        for k in ("gt", "pred", "tp", "cls_err", "fn", "fp"):
            group_total[b][k] += int(s.get(k, 0))

    headers = [
        "group",
        "gt",
        "pred",
        "tp",
        "fp",
        "fn",
        "cls_err",
        "miss_fn_rate",
        "cls_err_rate",
        "recall_gap",
        "fp_rate",
        "total_dev_rate",
    ]
    rows: list[dict] = []
    for group in ("一类害虫", "二类害虫", "其他虫子"):
        s = group_total[group]
        gt_n = int(s.get("gt", 0))
        pred_n = int(s.get("pred", 0))
        tp_n = int(s.get("tp", 0))
        ce_n = int(s.get("cls_err", 0))
        fn_n = int(s.get("fn", 0))
        fp_n = int(s.get("fp", 0))
        denom_gt = float(tp_n + ce_n + fn_n)
        denom_pred = float(tp_n + fp_n)
        miss_fn_rate = (float(fn_n) / denom_gt) if denom_gt > 0 else 0.0
        cls_err_rate = (float(ce_n) / denom_gt) if denom_gt > 0 else 0.0
        recall_gap = (float(fn_n + ce_n) / denom_gt) if denom_gt > 0 else 0.0
        fp_rate = (float(fp_n) / denom_pred) if denom_pred > 0 else 0.0
        total_dev = max(recall_gap, fp_rate)
        rows.append(
            {
                "group": group,
                "gt": gt_n,
                "pred": pred_n,
                "tp": tp_n,
                "fp": fp_n,
                "fn": fn_n,
                "cls_err": ce_n,
                "miss_fn_rate": round(miss_fn_rate, 6),
                "cls_err_rate": round(cls_err_rate, 6),
                "recall_gap": round(recall_gap, 6),
                "fp_rate": round(fp_rate, 6),
                "total_dev_rate": round(total_dev, 6),
            }
        )

    out_path = os.path.join(branch_dir, "group_summary_c1_c2_other.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(rows)


def _export_hn_report_csv(
    out_root: str,
    branch: str,
    stat_by_cls: dict[str, dict[str, int]],
    *,
    c1_set: set[str],
    hn_c2_set: set[str],
) -> None:
    if not stat_by_cls:
        return
    branch_dir = os.path.join(out_root, branch)
    os.makedirs(branch_dir, exist_ok=True)

    def _hn_acc(pred_n: int, gt_n: int) -> float:
        pred_n = int(pred_n)
        gt_n = int(gt_n)
        if gt_n <= 0:
            return 0.0
        acc = (1.0 - abs(float(pred_n - gt_n)) / float(gt_n)) * 100.0
        if acc < 0:
            return 0.0
        return float(acc)

    def _bucket(cls_norm: str) -> str:
        if cls_norm in c1_set:
            return "一类害虫"
        if cls_norm in hn_c2_set:
            return "湖南二类害虫"
        return "其他"

    hn_rows: list[dict] = []
    for cls_norm, s in stat_by_cls.items():
        cls_norm = str(cls_norm)
        gt_n = int(s.get("gt", 0))
        pred_n = int(s.get("pred", 0))
        if gt_n <= 0 and pred_n <= 0:
            continue
        hn_rows.append(
            {
                "group": _bucket(cls_norm),
                "class_norm": cls_norm,
                "gt": gt_n,
                "pred": pred_n,
                "acc_percent": round(_hn_acc(pred_n, gt_n), 6),
            }
        )
    bucket_rank = {"一类害虫": 0, "湖南二类害虫": 1, "其他": 9}
    hn_rows.sort(key=lambda x: (bucket_rank.get(x["group"], 9), x["group"], x["class_norm"]))

    out_path = os.path.join(branch_dir, "hn_counting_accuracy.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["group", "class_norm", "gt", "pred", "acc_percent"])
        w.writeheader()
        w.writerows(hn_rows)

    def _avg_acc(bucket: str) -> float:
        vals = [float(r["acc_percent"]) for r in hn_rows if r["group"] == bucket]
        if not vals:
            return 0.0
        return float(sum(vals) / float(len(vals)))

    avg_c1 = _avg_acc("一类害虫")
    avg_hn_c2 = _avg_acc("湖南二类害虫")
    day_acc = 0.6 * avg_c1 + 0.4 * avg_hn_c2
    day_score_30 = day_acc * 0.30
    with open(os.path.join(branch_dir, "hn_counting_accuracy_summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "avg_c1_percent",
                "avg_hn_c2_percent",
                "weighted_day_acc_percent",
                "score_30",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "avg_c1_percent": round(avg_c1, 6),
                "avg_hn_c2_percent": round(avg_hn_c2, 6),
                "weighted_day_acc_percent": round(day_acc, 6),
                "score_30": round(day_score_30, 6),
            }
        )


def _export_overall_summary_csv(out_root: str, branch: str, s: dict[str, int]) -> None:
    branch_dir = os.path.join(out_root, branch)
    os.makedirs(branch_dir, exist_ok=True)
    tp = int(s.get("tp", 0))
    fp = int(s.get("fp", 0))
    fn = int(s.get("fn", 0))
    ce = int(s.get("cls_err", 0))
    geom = int(s.get("geom_pairs", 0))
    denom_gt = float(tp + ce + fn)
    denom_pred = float(tp + fp)
    report_rate = (float(tp + ce) / denom_gt) if denom_gt > 0 else 0.0
    acc_rate = (float(tp) / denom_pred) if denom_pred > 0 else 0.0
    err_rate = (float(fp) / denom_pred) if denom_pred > 0 else 0.0
    miss_fn_rate = (float(fn) / denom_gt) if denom_gt > 0 else 0.0
    cls_err_rate = (float(ce) / denom_gt) if denom_gt > 0 else 0.0
    recall_gap = (float(fn + ce) / denom_gt) if denom_gt > 0 else 0.0
    total_dev = max(recall_gap, err_rate)
    out_path = os.path.join(branch_dir, "overall_summary.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "tp",
                "fp",
                "fn",
                "cls_err",
                "geom_pairs",
                "report_rate",
                "acc_rate",
                "err_rate",
                "miss_fn_rate",
                "cls_err_rate",
                "recall_gap",
                "total_dev_rate",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "cls_err": ce,
                "geom_pairs": geom,
                "report_rate": round(report_rate, 6),
                "acc_rate": round(acc_rate, 6),
                "err_rate": round(err_rate, 6),
                "miss_fn_rate": round(miss_fn_rate, 6),
                "cls_err_rate": round(cls_err_rate, 6),
                "recall_gap": round(recall_gap, 6),
                "total_dev_rate": round(total_dev, 6),
            }
        )


def _dedup_by_cls_iou(rows: list[dict], *, iou_threshold: float) -> list[dict]:
    """
    后处理去重：按 cls_name 分组做贪心 NMS（基于 box_iou）。
    解决“检测阶段不同 cls 都保留，但分类阶段映射到同一 cls_name”导致的重复框问题。
    """
    thr = float(iou_threshold)
    if thr <= 0 or not rows:
        return rows
    out: list[dict] = []
    by_cls: dict[str, list[dict]] = {}
    for r in rows:
        by_cls.setdefault(str(r.get("cls_name", "")), []).append(r)
    for _cls, rs in by_cls.items():
        rs = sorted(rs, key=lambda x: float(x.get("conf", 0.0) or 0.0), reverse=True)
        kept: list[dict] = []
        for r in rs:
            drop = False
            for k in kept:
                iou = float(
                    box_iou(
                        [r["x1"], r["y1"], r["x2"], r["y2"]],
                        [k["x1"], k["y1"], k["x2"], k["y2"]],
                    )
                )
                if iou >= thr:
                    drop = True
                    break
            if not drop:
                kept.append(r)
        out.extend(kept)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from script.insect_info import c1 as C1_KEYS, c2 as C2_MAP
    from script.insect_info import INSECTS as INSECTS_CFG

    # ----------------------- 运行开关 -----------------------
    # 只想跑其中一个时，关掉另一个即可；combined 依赖 small+large 同时开启才有意义
    enable_small: bool = False
    enable_large: bool = True
    enable_combined: bool = True
    # 画框开关：默认画框并保存可视化图片；关闭后不画框，仅保存预测 xml
    enable_draw_boxes: bool = True
    # 额外标注：打开时，在每个预测矩形框中心绘制点，并绘制“名称+置信度”标签（默认关闭）
    enable_draw_center_point_and_label: bool = False
    # detect 灰度检测：打开时，detect 检测阶段将输入图转为灰度（再扩展为 3 通道）进行检测。默认关闭。
    detect_gray: bool = False

    # ----------------------- 类别合并配置（可按需改） -----------------------
    # 大虫：沿用 predict_size_validate.py 内置示例的合并表（也可以在此精简/扩展）
    CLASS_MERGE_TO_GROUPS_LARGE: dict[str, list[str]] | None = {
        # 地老虎组
        # "bazidilaohu": ["bazidilaohu-bei", "bazidilaohu-fu"],
        # "xiaodilaohu": ["xiaodilaohu-bei", "xiaodilaohu-fu"],
        # # 年虫组（举例）
        # "dongfangnianchong": ["dongfangnianchong", "dongfangzhanchong"],
        # "laoshinianchong": ["laoshizhanchong", "laoshinianchong"],
        # "daofeishi": ["baibeifeishi", "hufeishi", "daofeishi"],
        # 粗分类：insect 可匹配任意具体昆虫类别（不依赖 detect/class_name 严格一致）
        "insect": ["*"],
    }
    # 小虫（稻飞虱）基本不需要合并，保持 None 即可；若 xml 里有别名可在此补
    CLASS_MERGE_TO_GROUPS_SMALL: dict[str, list[str]] | None = None

    # ----------------------- 输出暂隐（识别率低的类别） -----------------------
    # 名单中的分类：不写预测 xml、不画框、不做验证匹配、不参与逐图打印；按类汇总统计也不计入这些预测。
    # 可同时写「预测 cls_name 原文」或「类别合并后的组名」（与上面 CLASS_MERGE_* 一致）。
    # OUTPUT_SUPPRESS_CLASS_NAMES: tuple[str, ...] = ("xiaocaie", "xiaoshie")
    # 默认不暂隐，避免把所有框都过滤掉
    OUTPUT_SUPPRESS_CLASS_NAMES: tuple[str, ...] = ()
    _output_suppress_set = frozenset(OUTPUT_SUPPRESS_CLASS_NAMES) if OUTPUT_SUPPRESS_CLASS_NAMES else frozenset()

    # ----------------------- 关注昆虫列表（只分析/统计/画图这些虫） -----------------------
    # 留空/None 表示不过滤；一旦填写，脚本会：
    # - 逐图输出：只打印关注虫子的预测
    # - 保存图片与预测 xml：只绘制/写入关注虫子的框
    # - 评估与汇总统计：只对关注虫子计算 tp/fp/fn/cls_err、混淆矩阵、各类报表
    #
    # 列表里既可以写原始类别名，也可以写“归一后的类别名/合并组名”（normalize_class_name 后的名字）。
    # 例：FOCUS_INSECTS = ("bazidilaohu", "caodiming", "tiancaibaidaiyuming", "erdianweiyee")
    # FOCUS_INSECTS: tuple[str, ...] | None = ("bazidilaohu", "caodiming", "tiancaibaidaiyuming", "erdianweiyee", "yindingyee", "insect")
    FOCUS_INSECTS: tuple[str, ...] | None = None
    _focus_set = frozenset(str(x).strip() for x in (FOCUS_INSECTS or []) if str(x).strip())

    # ----------------------- 小虫模型配置（稻飞虱） -----------------------
    small_cls_list = []
    small_detect_model_path = "/Users/shunyaoyin/Documents/code/models/daofeishi-detect-0405.pt"
    # small_cls_model_path = "/Users/shunyaoyin/Documents/code/models/daofeishi-cls.pt"
    small_cls_model_path = None
    small_size_config_path = None
    small_conf_thresh = 0.55
    small_clip_size = 640
    small_overlap_size = 120
    # 福建参数
    # small_clip_size = 800
    # small_overlap_size = 150
    small_edge_reject_distance = 1
    small_edge_reject_conf_threshold = 0.5
    small_edge_reject_cls_conf_threshold = 0.66
    small_cls_top1_conf_threshold = 0.4
    small_cls_pad_square = False
    small_detect_pad_square_full_image = False
    small_detect_nms_iou = None
    small_detect_max_det = None

    # ----------------------- 大虫模型配置 -----------------------
    large_cls_list = None
    large_detect_model_path = "/Users/shunyaoyin/Documents/code/models/kuangxuan_0424.pt"
    large_detect_model_path = "/Users/shunyaoyin/Documents/code/ai-company/insect/doc/测试结果/大虫框选/v1.8-0509-insect-build-net/best.pt"
    large_detect_model_path = "/Users/shunyaoyin/Documents/code/ai-company/insect/doc/测试结果/大虫框选/v2.0-0512/训练过程/temp.pt"
    # 若有大虫分类模型，填路径；否则 None 仅用 detect 的 class_name
    large_cls_model_path = "/Users/shunyaoyin/Documents/code/ai-company/insect/doc/测试结果/分类测试/v1-20260424-all-large/best.pt"
    # large_cls_model_path = "/Users/shunyaoyin/Documents/code/ai-company/insect/doc/测试结果/分类测试/v2-20260428-all-large/best.pt"
    # large_cls_model_path = "/Users/shunyaoyin/Documents/code/ai-company/insect/doc/测试结果/分类测试/v6-cls/temp.pt"
    large_cls_model_path = None
    large_size_config_path = None
    large_conf_thresh = 0.01
    large_clip_size = 0
    large_overlap_size = 600
    large_edge_reject_distance = 5
    large_edge_reject_conf_threshold = 1
    large_edge_reject_cls_conf_threshold = 0.3
    large_cls_top1_conf_threshold = 0.3
    large_cls_pad_square = True
    # todo 一定注意配置
    large_detect_pad_square_full_image = True
    large_detect_nms_iou = 0.7
    large_detect_max_det = 1000
    # 跨类别 NMS：解决“两个不同 detect 类别但重叠很大”的重复框
    large_detect_nms_agnostic: bool | None = True
    # 分类后去重：按 cls_name 做一次 IoU 去重（避免“同一对象两个几乎相同框”）
    large_post_dedup_iou_threshold: float | None = 0.9

    # ----------------------- 输入输出 -----------------------
    input_path = "/Users/shunyaoyin/Documents/code/datasets/insect-data/test-data/福建大赛"
    # input_path = "/Users/shunyaoyin/Documents/code/datasets/insect-data/test-data/dachong-测试数据集"
    # input_path = "/Users/shunyaoyin/Documents/code/datasets/insect-data/test-data/虫情4设备现场测试数据"
    # input_path = "/Users/shunyaoyin/Documents/code/datasets/insect-data/test-data/duankou_1"
    # input_path = "/Users/shunyaoyin/Documents/code/datasets/insect-data/test-data/daofeishi-测试数据集"
    # input_path = "/Users/shunyaoyin/Documents/code/datasets/insect-data/test-data/dachong-标准测试集"
    # input_path = "/Users/shunyaoyin/Documents/code/datasets/insect-data/test-data/比赛-北京"
    # input_path = "/Users/shunyaoyin/Documents/code/datasets/insect-data/红河生产/0428"
    # input_path = "/Users/shunyaoyin/Documents/code/datasets/insect-data/红河生产/0428"
    # 检出测试
    input_path = "/Users/shunyaoyin/Documents/code/datasets/insect-data/test-data/dachong-检出测试集"
    # input_path = "/Users/shunyaoyin/Documents/code/datasets/insect-data/test-data/dachong-检测测试集泛化"
    input_path = "/Users/shunyaoyin/Documents/code/datasets/insect-data/框选all/生产"
    # input_path = "/Users/shunyaoyin/Documents/code/datasets/insect-data/框选all/生产偏多"
    # input_path = "/Users/shunyaoyin/Documents/code/datasets/insect-data/框选all/all-0330-0331-beijing"
    # input_path = "/Users/shunyaoyin/Documents/code/datasets/insect-data/生产收集/草地螟"
    # output_dir = input_path + "-daofeishi"
    output_dir = input_path + "-v2.0"
    predict_debug = False
    debug_clip = False
    label_mode = "minimal"  # "minimal" | "detailed"
    sort_stat_by_acc: bool = True

    # 几何匹配：默认 IoU；也可改成 "ior"
    # 说明：小虫目标很小，IoU=0.5 有时偏严（轻微偏移也会导致 IoU 掉很快）。
    # 可分别给 small/large/combined 设不同阈值，便于排查。
    val_box_match_metric_small = "ior"  # "iou" | "ior"
    val_box_match_metric_large = "iou"
    val_box_match_metric_combined = "iou"
    val_geom_threshold_small = 0.5 if val_box_match_metric_small.lower().strip() == "iou" else 0.6
    val_geom_threshold_large = 0.5 if val_box_match_metric_large.lower().strip() == "iou" else 0.8
    val_geom_threshold_combined = 0.5 if val_box_match_metric_combined.lower().strip() == "iou" else 0.8
    # FN 排查：若 small 出现漏检，打印每个漏检 gt 的 best match 分数与对应 pred
    debug_match_detail_small: bool = True

    # 标准评估：在保持原 PredictSize 推理与几何匹配的前提下，将混淆矩阵、per-class 精度/召回等写入 output_dir 子目录
    # （与 train/train_detect/train-cfg/val_v11_big.py 中 YOLO model.val 的「指标落盘」思路一致，指标由本脚本 VOC 匹配统计）
    enable_standard_eval: bool = True
    standard_eval_subdir: str = "eval_metrics"

    if enable_combined and (not enable_small or not enable_large):
        logging.warning("enable_combined=True 但 small/large 未同时开启，已自动关闭 combined")
        enable_combined = False

    small_predictor = None
    large_predictor = None
    if enable_small:
        small_predictor = PredictSize(
            detect_model_path=small_detect_model_path,
            size_config_path=small_size_config_path,
            cls_list=small_cls_list,
            cls_model_path=small_cls_model_path,
            offset_rate=1.2,
            conf_thresh=small_conf_thresh,
            device=None,
            augment=False,
        )
    if enable_large:
        large_predictor = PredictSize(
            detect_model_path=large_detect_model_path,
            size_config_path=large_size_config_path,
            cls_list=large_cls_list,
            cls_model_path=large_cls_model_path,
            offset_rate=1.2,
            conf_thresh=large_conf_thresh,
            device=None,
            augment=False,
        )

    input_p, image_files = _collect_images(input_path)
    print(f"共找到 {len(image_files)} 张图片")

    def _stat_init() -> dict[str, int]:
        return {"tp": 0, "fp": 0, "fn": 0, "cls_err": 0, "geom_pairs": 0, "img_with_xml": 0}

    stat_small = _stat_init() if enable_small else {}
    stat_large = _stat_init() if enable_large else {}
    stat_combined = _stat_init() if enable_combined else {}

    stat_by_cls_small: dict[str, dict[str, int]] = {} if enable_small else {}
    stat_by_cls_large: dict[str, dict[str, int]] = {} if enable_large else {}
    stat_by_cls_combined: dict[str, dict[str, int]] = {} if enable_combined else {}

    std_cm_small: defaultdict[tuple[str, str], int] | None = (
        defaultdict(int) if (enable_standard_eval and enable_small) else None
    )
    std_cm_large: defaultdict[tuple[str, str], int] | None = (
        defaultdict(int) if (enable_standard_eval and enable_large) else None
    )
    std_cm_combined: defaultdict[tuple[str, str], int] | None = (
        defaultdict(int) if (enable_standard_eval and enable_combined) else None
    )

    def _inc(stat_by_cls: dict[str, dict[str, int]], cls_name: str, key: str, n: int = 1) -> None:
        cls_name = str(cls_name or "")
        if cls_name not in stat_by_cls:
            stat_by_cls[cls_name] = {"tp": 0, "fp": 0, "fn": 0, "cls_err": 0, "gt": 0, "pred": 0}
        stat_by_cls[cls_name][key] = int(stat_by_cls[cls_name].get(key, 0)) + int(n)

    for idx, img_path in enumerate(image_files, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[{idx}/{len(image_files)}] 无法读取图片，跳过: {img_path}")
            continue

        rel_path = img_path.relative_to(input_p) if input_p.is_dir() else Path(img_path.name)
        src_xml = img_path.with_suffix(".xml")

        # -------- 同时跑小虫 / 大虫模型 --------
        all_small_rows = []
        small_results = []
        if enable_small:
            assert small_predictor is not None
            all_small_rows = small_predictor.predict(
                img,
                clip_size=small_clip_size,
                overlap_size=small_overlap_size,
                edge_reject_distance=small_edge_reject_distance,
                edge_reject_conf_threshold=small_edge_reject_conf_threshold,
                edge_reject_cls_conf_threshold=small_edge_reject_cls_conf_threshold,
                cls_top1_conf_threshold=small_cls_top1_conf_threshold,
                output=None,
                image_name=rel_path.name,
                debug=predict_debug,
                debug_clip=debug_clip,
                cls_pad_square=small_cls_pad_square,
                detect_pad_square_full_image=small_detect_pad_square_full_image,
                detect_gray=detect_gray,
                detect_nms_iou=small_detect_nms_iou,
                detect_max_det=small_detect_max_det,
                return_full_final=True,
            )
            small_results = [r for r in all_small_rows if not r.get("filter")]
            if _output_suppress_set:
                small_results, all_small_rows = _apply_output_suppress(
                    small_results, all_small_rows, _output_suppress_set, CLASS_MERGE_TO_GROUPS_SMALL
                )
            if _focus_set:
                small_results, all_small_rows = _apply_focus_filter(
                    small_results, all_small_rows, _focus_set, CLASS_MERGE_TO_GROUPS_SMALL
                )

        all_large_rows = []
        large_results = []
        if enable_large:
            assert large_predictor is not None
            all_large_rows = large_predictor.predict(
                img,
                clip_size=large_clip_size,
                overlap_size=large_overlap_size,
                edge_reject_distance=large_edge_reject_distance,
                edge_reject_conf_threshold=large_edge_reject_conf_threshold,
                edge_reject_cls_conf_threshold=large_edge_reject_cls_conf_threshold,
                cls_top1_conf_threshold=large_cls_top1_conf_threshold,
                output=None,
                image_name=rel_path.name,
                debug=predict_debug,
                debug_clip=debug_clip,
                cls_pad_square=large_cls_pad_square,
                detect_pad_square_full_image=large_detect_pad_square_full_image,
                detect_gray=detect_gray,
                detect_nms_iou=large_detect_nms_iou,
                detect_max_det=large_detect_max_det,
                detect_nms_agnostic=large_detect_nms_agnostic,
                return_full_final=True,
            )
            large_results = [r for r in all_large_rows if not r.get("filter")]
            if large_post_dedup_iou_threshold is not None:
                large_results = _dedup_by_cls_iou(
                    large_results, iou_threshold=float(large_post_dedup_iou_threshold)
                )
            if _output_suppress_set:
                large_results, all_large_rows = _apply_output_suppress(
                    large_results, all_large_rows, _output_suppress_set, CLASS_MERGE_TO_GROUPS_LARGE
                )
            if _focus_set:
                large_results, all_large_rows = _apply_focus_filter(
                    large_results, all_large_rows, _focus_set, CLASS_MERGE_TO_GROUPS_LARGE
                )

        # combined：直接拼接（保留来源，便于排查）
        all_combined_rows = []
        combined_results = []
        if enable_combined:
            for r in all_small_rows:
                r.setdefault("model_src", "small")
            for r in all_large_rows:
                r.setdefault("model_src", "large")
            all_combined_rows = list(all_small_rows) + list(all_large_rows)
            combined_results = [r for r in all_combined_rows if not r.get("filter")]
            if large_post_dedup_iou_threshold is not None:
                combined_results = _dedup_by_cls_iou(
                    combined_results, iou_threshold=float(large_post_dedup_iou_threshold)
                )
            if _output_suppress_set:
                combined_results, all_combined_rows = _apply_output_suppress_combined(
                    combined_results,
                    all_combined_rows,
                    _output_suppress_set,
                    CLASS_MERGE_TO_GROUPS_SMALL,
                    CLASS_MERGE_TO_GROUPS_LARGE,
                )
            if _focus_set:
                combined_results, all_combined_rows = _apply_focus_filter_combined(
                    combined_results,
                    all_combined_rows,
                    _focus_set,
                    CLASS_MERGE_TO_GROUPS_SMALL,
                    CLASS_MERGE_TO_GROUPS_LARGE,
                )

        # -------- 读取标注并评估（若无 xml 仍保存预测结果图/xml 便于看分布） --------
        gts: list[dict] | None = None
        if src_xml.is_file():
            try:
                gts = parse_pascal_voc_objects(str(src_xml))
            except (ET.ParseError, OSError, ValueError) as e:
                logging.warning("读取源 xml 失败 %s: %s", src_xml, e)
                gts = None

        # 输出目录：small/large/combined 保持同样子目录结构
        out_small_dir = os.path.join(output_dir, "small", str(rel_path.parent)) if str(rel_path.parent) != "." else os.path.join(output_dir, "small")
        out_large_dir = os.path.join(output_dir, "large", str(rel_path.parent)) if str(rel_path.parent) != "." else os.path.join(output_dir, "large")
        out_comb_dir = os.path.join(output_dir, "combined", str(rel_path.parent)) if str(rel_path.parent) != "." else os.path.join(output_dir, "combined")

        def _eval_one(
            preds_visible: list[dict],
            all_rows: list[dict],
            *,
            gts_local: list[dict] | None,
            class_merge_to_groups: dict[str, list[str]] | None,
            subset_keep_gt_fn=None,
            stat_by_cls: dict[str, dict[str, int]] | None = None,
            geom_metric: str = "iou",
            geom_threshold: float = 0.5,
        ) -> tuple[dict[str, int], list[tuple[int, int, float]] | None, set[int] | None, list[dict] | None]:
            s = {"tp": 0, "fp": 0, "fn": 0, "cls_err": 0, "geom_pairs": 0, "img_with_xml": 0}
            matches = None
            matched_p = None
            used_gts = gts_local
            if gts_local is None:
                return s, matches, matched_p, used_gts

            # 关注列表：先对 GT 侧做过滤，保证“评估/统计”只围绕关注虫子
            if _focus_set:
                def _keep_gt(name: str) -> bool:
                    raw = str(name or "").strip()
                    if not raw:
                        return False
                    if raw in _focus_set:
                        return True
                    norm = normalize_class_name(raw, class_merge_to_groups)
                    return bool(norm) and norm in _focus_set

                gts_local = [g for g in gts_local if _keep_gt(str(g.get("name", "") or ""))]
                used_gts = gts_local

            if subset_keep_gt_fn is not None:
                _preds, _gts = _subset_by_gt(preds_visible, gts_local, keep_gt_fn=subset_keep_gt_fn)
                used_gts = _gts
            else:
                _preds, _gts = preds_visible, gts_local
                used_gts = gts_local

            merge = class_merge_to_groups

            def _pred_skip(r: dict) -> bool:
                return is_metric_ignored_other(str(r.get("cls_name", "") or ""), merge)

            def _gt_skip(g: dict) -> bool:
                return is_metric_ignored_other(str(g.get("name", "") or ""), merge)

            _pred_eval_idx = [i for i in range(len(_preds)) if not _pred_skip(_preds[i])]
            _gts_eval_idx = [j for j in range(len(_gts)) if not _gt_skip(_gts[j])]
            _preds_eval = [_preds[i] for i in _pred_eval_idx]
            _gts_eval = [_gts[j] for j in _gts_eval_idx]

            if not _gts:
                # 子集内没有 gt：非 other 的预测计为 FP
                s["fp"] = sum(1 for i in range(len(_preds)) if not _pred_skip(_preds[i]))
                s["img_with_xml"] = 1
                if stat_by_cls is not None:
                    for i in range(len(_preds)):
                        if _pred_skip(_preds[i]):
                            continue
                        pn = normalize_class_name(_preds[i].get("cls_name", ""), merge)
                        _inc(stat_by_cls, pn, "pred", 1)
                        _inc(stat_by_cls, pn, "fp", 1)
                return s, matches, matched_p, used_gts

            if not _gts_eval:
                # 子集内仅含 other 等「不参与指标」的 GT：不计 FN，仅统计非 other 预测为 FP
                s["img_with_xml"] = 1
                s["fp"] = sum(1 for i in range(len(_preds)) if not _pred_skip(_preds[i]))
                if stat_by_cls is not None:
                    for i in range(len(_preds)):
                        if _pred_skip(_preds[i]):
                            continue
                        pn = normalize_class_name(_preds[i].get("cls_name", ""), merge)
                        _inc(stat_by_cls, pn, "pred", 1)
                        _inc(stat_by_cls, pn, "fp", 1)
                matches = []
                matched_p = set()
                return s, matches, matched_p, used_gts

            m = str(geom_metric or "iou").lower().strip()
            thr = float(geom_threshold)
            if m == "ior":
                matches_ev, matched_p_ev, matched_g_ev = match_pred_gt_ior(_preds_eval, _gts_eval, thr)
            else:
                matches_ev, matched_p_ev, matched_g_ev = match_pred_gt(_preds_eval, _gts_eval, thr, metric="iou")

            matches = [
                (_pred_eval_idx[pi], _gts_eval_idx[gj], sc) for pi, gj, sc in matches_ev
            ]
            matched_p = {_pred_eval_idx[i] for i in matched_p_ev}
            matched_g = {_gts_eval_idx[j] for j in matched_g_ev}

            s["geom_pairs"] = len(matches)
            s["img_with_xml"] = 1

            pred_to_gt = {i: j for i, j, _ in matches}
            if stat_by_cls is not None:
                for j in _gts_eval_idx:
                    _inc(stat_by_cls, normalize_class_name(_gts[j].get("name", ""), merge), "gt", 1)
                for i in _pred_eval_idx:
                    _inc(stat_by_cls, normalize_class_name(_preds[i].get("cls_name", ""), merge), "pred", 1)

            for i in range(len(_preds)):
                if _pred_skip(_preds[i]):
                    continue
                if i not in matched_p:
                    s["fp"] += 1
                    if stat_by_cls is not None:
                        _inc(stat_by_cls, normalize_class_name(_preds[i].get("cls_name", ""), merge), "fp", 1)
                else:
                    j = pred_to_gt[i]
                    pred_cls = _preds[i].get("cls_name", "")
                    gt_name = _gts[j].get("name", "")
                    pred_norm = normalize_class_name(pred_cls, merge)
                    gt_norm = normalize_class_name(gt_name, merge)
                    if is_class_match(pred_cls, gt_name, merge):
                        s["tp"] += 1
                        if stat_by_cls is not None:
                            _inc(stat_by_cls, gt_norm, "tp", 1)
                    else:
                        s["cls_err"] += 1
                        s["fp"] += 1
                        if stat_by_cls is not None:
                            _inc(stat_by_cls, gt_norm, "cls_err", 1)
                            _inc(stat_by_cls, pred_norm, "fp", 1)
            s["fn"] = len(_gts_eval) - len(matched_g_ev)
            if stat_by_cls is not None:
                for j in _gts_eval_idx:
                    if j not in matched_g:
                        _inc(stat_by_cls, normalize_class_name(_gts[j].get("name", ""), merge), "fn", 1)
            return s, matches, matched_p, used_gts

        # small：只对 daofeishi-like gt 做子集评估（但仍输出全图预测，便于查误报）
        small_s, small_matches, small_matched_p, small_used_gts = (
            ({"tp": 0, "fp": 0, "fn": 0, "cls_err": 0, "geom_pairs": 0, "img_with_xml": 0}, None, None, None)
        )
        if enable_small:
            small_s, small_matches, small_matched_p, small_used_gts = _eval_one(
                small_results,
                all_small_rows,
                gts_local=gts,
                class_merge_to_groups=CLASS_MERGE_TO_GROUPS_SMALL,
                subset_keep_gt_fn=_is_daofeishi_like,
                stat_by_cls=stat_by_cls_small,
                geom_metric=val_box_match_metric_small,
                geom_threshold=val_geom_threshold_small,
            )
            if std_cm_small is not None:
                _std_eval_collect_confusion(
                    std_cm_small,
                    small_results,
                    small_used_gts,
                    small_matches,
                    small_matched_p,
                    CLASS_MERGE_TO_GROUPS_SMALL,
                )
        # large：对非 daofeishi gt 子集评估
        large_s, large_matches, large_matched_p, large_used_gts = (
            ({"tp": 0, "fp": 0, "fn": 0, "cls_err": 0, "geom_pairs": 0, "img_with_xml": 0}, None, None, None)
        )
        if enable_large:
            large_s, large_matches, large_matched_p, large_used_gts = _eval_one(
                large_results,
                all_large_rows,
                gts_local=gts,
                class_merge_to_groups=CLASS_MERGE_TO_GROUPS_LARGE,
                subset_keep_gt_fn=lambda n: (not _is_daofeishi_like(n)),
                stat_by_cls=stat_by_cls_large,
                geom_metric=val_box_match_metric_large,
                geom_threshold=val_geom_threshold_large,
            )
            if std_cm_large is not None:
                _std_eval_collect_confusion(
                    std_cm_large,
                    large_results,
                    large_used_gts,
                    large_matches,
                    large_matched_p,
                    CLASS_MERGE_TO_GROUPS_LARGE,
                )
        # combined：对全部 gt 评估（整体效果）
        comb_s, comb_matches, comb_matched_p, comb_used_gts = (
            ({"tp": 0, "fp": 0, "fn": 0, "cls_err": 0, "geom_pairs": 0, "img_with_xml": 0}, None, None, None)
        )
        if enable_combined:
            comb_s, comb_matches, comb_matched_p, comb_used_gts = _eval_one(
                combined_results,
                all_combined_rows,
                gts_local=gts,
                class_merge_to_groups=CLASS_MERGE_TO_GROUPS_LARGE,
                subset_keep_gt_fn=None,
                stat_by_cls=stat_by_cls_combined,
                geom_metric=val_box_match_metric_combined,
                geom_threshold=val_geom_threshold_combined,
            )
            if std_cm_combined is not None:
                _std_eval_collect_confusion(
                    std_cm_combined,
                    combined_results,
                    comb_used_gts,
                    comb_matches,
                    comb_matched_p,
                    CLASS_MERGE_TO_GROUPS_LARGE,
                )

        # small FN 细节排查：打印每个未匹配 gt 的 best score
        if enable_small and debug_match_detail_small and gts is not None and small_used_gts and small_s.get("fn", 0) > 0:
            # 计算哪些 gt 没被匹配到
            matched_g = set()
            if small_matches:
                for _pi, gj, _sc in small_matches:
                    matched_g.add(int(gj))

            def _score_iou(p: dict, g: dict) -> float:
                return float(
                    box_iou(
                        [p["x1"], p["y1"], p["x2"], p["y2"]],
                        [g["x1"], g["y1"], g["x2"], g["y2"]],
                    )
                )

            def _score_ior(p: dict, g: dict) -> float:
                # 复用 match_pred_gt_ior 的阈值语义：与 model_detect.ior 一致
                from script.predict.model_detect import ior as _ior

                return float(
                    _ior(
                        [p["x1"], p["y1"], p["x2"], p["y2"]],
                        [g["x1"], g["y1"], g["x2"], g["y2"]],
                    )
                )

            metric = val_box_match_metric_small.lower().strip()
            scorer = _score_ior if metric == "ior" else _score_iou
            thr = float(val_geom_threshold_small)
            print(f"  [debug] small 漏检明细：metric={metric} thr={thr}")
            for gj, g in enumerate(small_used_gts):
                if is_metric_ignored_other(str(g.get("name", "") or ""), CLASS_MERGE_TO_GROUPS_SMALL):
                    continue
                if gj in matched_g:
                    continue
                best = (-1.0, None)
                for pi, p in enumerate(small_results):
                    try:
                        sc = scorer(p, g)
                    except Exception:
                        continue
                    if sc > best[0]:
                        best = (sc, pi)
                bsc, bpi = best
                if bpi is None:
                    print(f"    miss gt[{gj}]={g.get('name','')} box=({g.get('x1')},{g.get('y1')},{g.get('x2')},{g.get('y2')}) | best=None")
                else:
                    p = small_results[int(bpi)]
                    print(
                        f"    miss gt[{gj}]={g.get('name','')} box=({g.get('x1')},{g.get('y1')},{g.get('x2')},{g.get('y2')}) | "
                        f"best={bsc:.3f} pred[{bpi}]={p.get('cls_name','')} box=({p.get('x1')},{p.get('y1')},{p.get('x2')},{p.get('y2')})"
                    )

        # 汇总累加
        if enable_small:
            for k in stat_small.keys():
                stat_small[k] += int(small_s.get(k, 0))
        if enable_large:
            for k in stat_large.keys():
                stat_large[k] += int(large_s.get(k, 0))
        if enable_combined:
            for k in stat_combined.keys():
                stat_combined[k] += int(comb_s.get(k, 0))

        # 保存可视化与预测 xml（xml 始终写 preds_visible；验证图在有 xml 时会按 tp/fp/fn 着色）
        if enable_small:
            _save_image_and_xml(
                img,
                all_small_rows,
                small_results,
                out_dir=out_small_dir,
                rel_path=Path(rel_path.name),
                clip_size=small_clip_size,
                overlap_size=small_overlap_size,
                predict_debug=predict_debug,
                label_mode=label_mode,
                val_xml_mode=(gts is not None),
                draw_boxes=enable_draw_boxes,
                draw_center_point_and_label=enable_draw_center_point_and_label,
                gts=small_used_gts,
                matches=small_matches,
                matched_p=small_matched_p,
                class_merge_to_groups=CLASS_MERGE_TO_GROUPS_SMALL,
            )
        if enable_large:
            _save_image_and_xml(
                img,
                all_large_rows,
                large_results,
                out_dir=out_large_dir,
                rel_path=Path(rel_path.name),
                clip_size=large_clip_size,
                overlap_size=large_overlap_size,
                predict_debug=predict_debug,
                label_mode=label_mode,
                val_xml_mode=(gts is not None),
                draw_boxes=enable_draw_boxes,
                draw_center_point_and_label=enable_draw_center_point_and_label,
                gts=large_used_gts,
                matches=large_matches,
                matched_p=large_matched_p,
                class_merge_to_groups=CLASS_MERGE_TO_GROUPS_LARGE,
            )
        if enable_combined:
            _save_image_and_xml(
                img,
                all_combined_rows,
                combined_results,
                out_dir=out_comb_dir,
                rel_path=Path(rel_path.name),
                clip_size=max(int(small_clip_size), int(large_clip_size)),
                overlap_size=max(int(small_overlap_size), int(large_overlap_size)),
                predict_debug=predict_debug,
                label_mode=label_mode,
                val_xml_mode=(gts is not None),
                draw_boxes=enable_draw_boxes,
                draw_center_point_and_label=enable_draw_center_point_and_label,
                gts=comb_used_gts,
                matches=comb_matches,
                matched_p=comb_matched_p,
                class_merge_to_groups=CLASS_MERGE_TO_GROUPS_LARGE,
            )

        parts = [f"[{idx}/{len(image_files)}] {rel_path}"]
        if enable_small:
            parts.append(
                f"small(pred={len(small_results)} tp={small_s['tp']} fp={small_s['fp']} fn={small_s['fn']} cls_err={small_s['cls_err']})"
            )
        if enable_large:
            parts.append(
                f"large(pred={len(large_results)} tp={large_s['tp']} fp={large_s['fp']} fn={large_s['fn']} cls_err={large_s['cls_err']})"
            )
        if enable_combined:
            parts.append(
                f"combined(pred={len(combined_results)} tp={comb_s['tp']} fp={comb_s['fp']} fn={comb_s['fn']} cls_err={comb_s['cls_err']})"
            )
        parts.append(f"(xml={src_xml.is_file()})")
        print("  ".join(parts))
        # 逐条打印（参考 predict_size_validate.py）
        if enable_small and small_results:
            print("  small results:")
            for r in small_results:
                print(
                    f"    [{r.get('cls_name','')}] conf={float(r.get('cls_conf', 0) or 0):.2f}  "
                    f"det_conf={float(r.get('conf', 0) or 0):.2f}  "
                    f"box=({r.get('x1')},{r.get('y1')},{r.get('x2')},{r.get('y2')})"
                )
        if enable_large and large_results:
            print("  large results:")
            for r in large_results:
                print(
                    f"    [{r.get('cls_name','')}] conf={float(r.get('cls_conf', 0) or 0):.2f}  "
                    f"det_conf={float(r.get('conf', 0) or 0):.2f}  "
                    f"box=({r.get('x1')},{r.get('y1')},{r.get('x2')},{r.get('y2')})"
                )
        if enable_combined and combined_results:
            print("  combined results:")
            for r in combined_results:
                print(
                    f"    [{r.get('model_src','?')}|{r.get('cls_name','')}] conf={float(r.get('cls_conf', 0) or 0):.2f}  "
                    f"det_conf={float(r.get('conf', 0) or 0):.2f}  "
                    f"box=({r.get('x1')},{r.get('y1')},{r.get('x2')},{r.get('y2')})"
                )

    if small_predictor is not None:
        small_predictor.release()
    if large_predictor is not None:
        large_predictor.release()

    def _summary(name: str, s: dict[str, int]) -> str:
        tp = int(s.get("tp", 0))
        fp = int(s.get("fp", 0))
        fn = int(s.get("fn", 0))
        ce = int(s.get("cls_err", 0))
        geom = int(s.get("geom_pairs", 0))
        denom_gt = float(tp + ce + fn)
        denom_pred = float(tp + fp)
        report_rate = (float(tp + ce) / denom_gt) if denom_gt > 0 else 0.0
        acc_rate = (float(tp) / denom_pred) if denom_pred > 0 else 0.0
        err_rate = (float(fp) / denom_pred) if denom_pred > 0 else 0.0
        miss_fn_rate = (float(fn) / denom_gt) if denom_gt > 0 else 0.0
        cls_err_rate = (float(ce) / denom_gt) if denom_gt > 0 else 0.0
        recall_gap = (float(fn + ce) / denom_gt) if denom_gt > 0 else 0.0
        total_dev = max(recall_gap, err_rate)
        return (
            f"{name}: tp={tp} fp={fp} fn={fn} cls_err={ce} geom_pairs={geom} | "
            f"报出率={report_rate*100:.2f}% 正确率={acc_rate*100:.2f}% 错误率={err_rate*100:.2f}% "
            f"漏检率(仅几何无框)={miss_fn_rate*100:.2f}% 类错率={cls_err_rate*100:.2f}% "
            f"召回缺口={recall_gap*100:.2f}% 总偏差率=max(召回缺口,错误率)={total_dev*100:.2f}%"
        )

    print("======== 双模型验证汇总（仅对有 xml 的图统计 tp/fp/fn/cls_err） ========")
    if enable_small:
        print(_summary("small(稻飞虱子集)", stat_small))
    if enable_large:
        print(_summary("large(大虫子集)", stat_large))
    if enable_combined:
        print(_summary("combined(全量)", stat_combined))

    if enable_small:
        _print_stat_by_cls("small 按类别统计", stat_by_cls_small, sort_by_acc=sort_stat_by_acc)
    if enable_large:
        _print_stat_by_cls("large 按类别统计", stat_by_cls_large, sort_by_acc=sort_stat_by_acc)
    if enable_combined:
        _print_stat_by_cls("combined 按类别统计", stat_by_cls_combined, sort_by_acc=sort_stat_by_acc)

    # -------- 一类/二类/其他 汇总统计（基于 script/insect_info.py 的 c1/c2）--------
    c1_set = set(str(x) for x in (C1_KEYS or []))
    c2_set = set(str(x) for x in (C2_MAP or {}).keys())

    def _bucket(cls_norm: str) -> str:
        if cls_norm in c1_set:
            return "一类害虫"
        if cls_norm in c2_set:
            return "二类害虫"
        return "其他虫子"

    def _group_summary(title: str, stat_by_cls: dict[str, dict[str, int]]) -> None:
        group_total: dict[str, dict[str, int]] = {
            "一类害虫": {"gt": 0, "pred": 0, "tp": 0, "cls_err": 0, "fn": 0, "fp": 0},
            "二类害虫": {"gt": 0, "pred": 0, "tp": 0, "cls_err": 0, "fn": 0, "fp": 0},
            "其他虫子": {"gt": 0, "pred": 0, "tp": 0, "cls_err": 0, "fn": 0, "fp": 0},
        }
        for cls_norm, s in stat_by_cls.items():
            b = _bucket(str(cls_norm))
            for k in ("gt", "pred", "tp", "cls_err", "fn", "fp"):
                group_total[b][k] += int(s.get(k, 0))

        def _fmt_group_line(name: str, s: dict[str, int]) -> str:
            gt_n = int(s.get("gt", 0))
            pred_n = int(s.get("pred", 0))
            tp_n = int(s.get("tp", 0))
            ce_n = int(s.get("cls_err", 0))
            fn_n = int(s.get("fn", 0))
            fp_n = int(s.get("fp", 0))
            denom_gt = float(tp_n + ce_n + fn_n)
            denom_pred = float(tp_n + fp_n)
            miss_fn_rate = (float(fn_n) / denom_gt) if denom_gt > 0 else 0.0
            cls_err_rate = (float(ce_n) / denom_gt) if denom_gt > 0 else 0.0
            recall_gap = (float(fn_n + ce_n) / denom_gt) if denom_gt > 0 else 0.0
            fp_rate = (float(fp_n) / denom_pred) if denom_pred > 0 else 0.0
            total_dev = max(recall_gap, fp_rate)
            return (
                f"{name}: gt={gt_n} pred={pred_n} tp={tp_n} fp={fp_n} fn={fn_n} cls_err={ce_n} | "
                f"漏检率(仅几何)={miss_fn_rate*100:.2f}% 类错率={cls_err_rate*100:.2f}% "
                f"误报率={fp_rate*100:.2f}% 总偏差率=max(召回缺口,误报率)={total_dev*100:.2f}%"
            )

        print(f"{title} 分组统计（按 insect_info 的一类/二类/其他）：")
        print(_fmt_group_line("一类害虫", group_total["一类害虫"]))
        print(_fmt_group_line("二类害虫", group_total["二类害虫"]))
        print(_fmt_group_line("其他虫子", group_total["其他虫子"]))

    if enable_small:
        _group_summary("small", stat_by_cls_small)
    if enable_large:
        _group_summary("large", stat_by_cls_large)
    if enable_combined:
        _group_summary("combined", stat_by_cls_combined)

    # -------- 湖南统计方法：按“识别数量 vs 鉴定数量”计算每种准确率，并加权汇总 --------
    def _hn_acc(pred_n: int, gt_n: int) -> float:
        pred_n = int(pred_n)
        gt_n = int(gt_n)
        if gt_n <= 0:
            return 0.0
        acc = (1.0 - abs(float(pred_n - gt_n)) / float(gt_n)) * 100.0
        if acc < 0:
            return 0.0
        return float(acc)

    hn_c2_set: set[str] = set()
    for k, cfg in (INSECTS_CFG or {}).items():
        try:
            if int(getattr(cfg, "level", 0)) != 2:
                continue
            zones = set(getattr(cfg, "zones", []) or [])
            notes = str(getattr(cfg, "notes", "") or "")
            if ("华中" in zones) or ("湖南" in notes):
                hn_c2_set.add(str(k))
        except Exception:
            continue

    def _hn_report(title: str, stat_by_cls: dict[str, dict[str, int]]) -> None:
        def _is_c1(k: str) -> bool:
            return k in c1_set

        def _is_hn_c2(k: str) -> bool:
            return k in hn_c2_set

        hn_rows: list[tuple[str, str, int, int, float]] = []
        for cls_norm, s in stat_by_cls.items():
            cls_norm = str(cls_norm)
            gt_n = int(s.get("gt", 0))
            pred_n = int(s.get("pred", 0))
            if gt_n <= 0 and pred_n <= 0:
                continue
            bucket = "一类害虫" if _is_c1(cls_norm) else ("湖南二类害虫" if _is_hn_c2(cls_norm) else "其他")
            acc = _hn_acc(pred_n, gt_n)
            hn_rows.append((bucket, cls_norm, gt_n, pred_n, acc))

        bucket_rank = {"一类害虫": 0, "湖南二类害虫": 1, "其他": 9}
        hn_rows.sort(key=lambda x: (bucket_rank.get(x[0], 9), x[0], x[1]))

        print(f"{title} 湖南统计方法（每种虫体的自动识别与计数准确率）：")
        print("分组 | 类别(归一) | 鉴定数量(gt) | 识别数量(pred) | 每种准确率")
        for b, cls_norm, gt_n, pred_n, acc in hn_rows:
            print(f"{b} | {cls_norm} | {gt_n} | {pred_n} | {acc:.2f}%")

        def _avg_acc(rows: list[tuple[str, str, int, int, float]], bucket: str) -> float:
            vals = [r[4] for r in rows if r[0] == bucket]
            if not vals:
                return 0.0
            return float(sum(vals) / float(len(vals)))

        avg_c1 = _avg_acc(hn_rows, "一类害虫")
        avg_hn_c2 = _avg_acc(hn_rows, "湖南二类害虫")
        day_acc = 0.6 * avg_c1 + 0.4 * avg_hn_c2
        day_score_30 = day_acc * 0.30

        print(
            f"{title} 湖南统计方法汇总："
            f"一类均值={avg_c1:.2f}%  湖南二类均值={avg_hn_c2:.2f}%  "
            f"加权当天准确率={day_acc:.2f}%  折算(30分制)={day_score_30:.2f}/30"
        )

    if enable_small:
        _hn_report("small", stat_by_cls_small)
    if enable_large:
        _hn_report("large", stat_by_cls_large)
    if enable_combined:
        _hn_report("combined", stat_by_cls_combined)

    if enable_standard_eval:
        eval_root = os.path.join(output_dir, standard_eval_subdir)
        ref_cfg = "train/train_detect/train-cfg/val_v11_big.py"
        if std_cm_small is not None:
            _std_eval_save_branch(
                eval_root,
                "small",
                dict(std_cm_small),
                stat_small,
                {
                    "reference": ref_cfg,
                    "match_metric": val_box_match_metric_small,
                    "geom_threshold": val_geom_threshold_small,
                    "gt_subset": "daofeishi-like only",
                },
            )
            _export_overall_summary_csv(eval_root, "small", stat_small)
            _export_stat_by_cls_csv(eval_root, "small", stat_by_cls_small, sort_by_acc=sort_stat_by_acc)
            _export_group_summary_csv(
                eval_root,
                "small",
                stat_by_cls_small,
                c1_set=c1_set,
                c2_set=c2_set,
            )
            _export_hn_report_csv(
                eval_root,
                "small",
                stat_by_cls_small,
                c1_set=c1_set,
                hn_c2_set=hn_c2_set,
            )
        if std_cm_large is not None:
            _std_eval_save_branch(
                eval_root,
                "large",
                dict(std_cm_large),
                stat_large,
                {
                    "reference": ref_cfg,
                    "match_metric": val_box_match_metric_large,
                    "geom_threshold": val_geom_threshold_large,
                    "gt_subset": "non-daofeishi",
                },
            )
            _export_overall_summary_csv(eval_root, "large", stat_large)
            _export_stat_by_cls_csv(eval_root, "large", stat_by_cls_large, sort_by_acc=sort_stat_by_acc)
            _export_group_summary_csv(
                eval_root,
                "large",
                stat_by_cls_large,
                c1_set=c1_set,
                c2_set=c2_set,
            )
            _export_hn_report_csv(
                eval_root,
                "large",
                stat_by_cls_large,
                c1_set=c1_set,
                hn_c2_set=hn_c2_set,
            )
        if std_cm_combined is not None:
            _std_eval_save_branch(
                eval_root,
                "combined",
                dict(std_cm_combined),
                stat_combined,
                {
                    "reference": ref_cfg,
                    "match_metric": val_box_match_metric_combined,
                    "geom_threshold": val_geom_threshold_combined,
                    "gt_subset": "all",
                },
            )
            _export_overall_summary_csv(eval_root, "combined", stat_combined)
            _export_stat_by_cls_csv(eval_root, "combined", stat_by_cls_combined, sort_by_acc=sort_stat_by_acc)
            _export_group_summary_csv(
                eval_root,
                "combined",
                stat_by_cls_combined,
                c1_set=c1_set,
                c2_set=c2_set,
            )
            _export_hn_report_csv(
                eval_root,
                "combined",
                stat_by_cls_combined,
                c1_set=c1_set,
                hn_c2_set=hn_c2_set,
            )
        print(f"标准评估输出目录: {eval_root}")

    print(f"输出目录: {output_dir}")
    print("处理完成")

