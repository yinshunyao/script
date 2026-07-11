#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Detail  : YOLO 分割（+ 可选分类）推理与可视化验证。
#           阈值：script/insect_alg_seg.json（相对路径，detect_conf / cls_conf / dia）
#           分类：默认按分割多边形掩码裁剪；cls_crop_from_bbox=True 时按 bbox 矩形裁剪
#           支持 cls_pad_square / cls_gray_binarize
#           有标注 xml 时按 predict_size_validate_lib 口径统计报出率/正确率等；类别合并由 cls_merge.py 派生（cls_hierarchy_util.build_merge_groups_from_cls_merge）

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

from script.cls_hierarchy_util import build_merge_groups_from_cls_merge
from script.insect_cls_map import similar_name as INSECT_SIMILAR_NAME
from script.predict_seg import INSECT_ALG_SEG_JSON_REL, PredictSeg
from script.predict_seg_lib import (
    bbox_from_row,
    collect_images,
    draw_seg_output_image,
    filter_rows_by_bbox_diag_range,
    load_insect_alg,
    match_pred_gt_polygon,
    outputs_exist_for_skip,
    parse_voc_objects,
    resolve_insect_alg_seg_path,
    resolve_seg_dia_range,
    write_voc_seg_xml,
)
from script.predict_size_validate_lib import _print_stat_by_cls
from script.predict_size_validate_lib import (
    is_class_match,
    is_metric_ignored_other,
    normalize_class_name,
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
    cls_output_top_n: int,
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
            cls_output_top_n=cls_output_top_n,
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


def build_eval_class_merge(
    base: dict[str, list[str]] | None = None,
    *,
    insect_wildcard: bool = True,
) -> dict[str, list[str]] | None:
    """
    评估用类别合并表：默认在 ``cls_merge`` 派生的合并组上叠加 ``insect: ['*']``。

    ``*`` 仅参与 ``is_class_match``（粗标 ``insect`` 与任意具体合并类/物种互认正确），
    不会把未配置类别在 normalize 时一律压成 insect（见 predict_size_validate_lib）。
    """
    if base is None and not insect_wildcard:
        return None
    out: dict[str, list[str]] = dict(base or {})
    if insect_wildcard:
        aliases = [str(a).strip() for a in (out.get("insect") or []) if str(a).strip()]
        if "*" not in aliases:
            aliases.append("*")
        out["insect"] = aliases
    return out or None


def _build_bidirectional_equiv_map(m: dict[str, str] | None) -> dict[str, str]:
    """
    等价名映射（双向等价）。

    约定：
    - 对任意 k->v，评估时把 k 与 v 都归一为 v（即 canonical 取 value）。
    - v 本身也会映射到 v，保证 key/value 完全等价。
    """
    if not m:
        return {}
    out: dict[str, str] = {}
    for k, v in m.items():
        kk = str(k or "").strip()
        vv = str(v or "").strip()
        if not kk or not vv:
            continue
        out[kk] = vv
        out[vv] = vv
    return out


_EVAL_EQUIV_NAME_MAP: dict[str, str] = _build_bidirectional_equiv_map(INSECT_SIMILAR_NAME)

try:
    INSECT_CLS_MERGE_MAP: dict[str, list[str]] = build_merge_groups_from_cls_merge()
except (ImportError, FileNotFoundError, ValueError) as _e:
    logging.warning("从 cls_merge.py 构建类别合并组失败，使用空合并组: %s", _e)
    INSECT_CLS_MERGE_MAP = {}


def _canonical_eval_name(raw: str) -> str:
    """
    评估用 canonical 名：
    - 先取叶目录/标注名首段物种键（如 bazidilaohu-ba-bei → bazidilaohu）
    - 再按 insect_cls_map.similar_name 做等价归一（key/value 完全等价）
    """
    s = str(raw or "").strip()
    if not s:
        return s
    species = s.split("-", 1)[0]
    return _EVAL_EQUIV_NAME_MAP.get(species, species)


def _eval_normalize_class_name(raw: str, merge: dict[str, list[str]] | None) -> str:
    return normalize_class_name(_canonical_eval_name(raw), merge)


def _eval_is_class_match(pred_raw: str, gt_raw: str, merge: dict[str, list[str]] | None) -> bool:
    return is_class_match(_canonical_eval_name(pred_raw), _canonical_eval_name(gt_raw), merge)


def _inc_stat_by_cls(stat_by_cls: dict[str, dict[str, int]], cls_name: str, key: str, n: int = 1) -> None:
    cls_name = str(cls_name or "")
    if cls_name not in stat_by_cls:
        stat_by_cls[cls_name] = {"gt": 0, "pred": 0, "tp": 0, "fp": 0, "fn": 0, "cls_err": 0}
    stat_by_cls[cls_name][key] = int(stat_by_cls[cls_name].get(key, 0)) + int(n)


def _print_overall_stat_summary(title: str, s: dict[str, int]) -> None:
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
    print(
        f"{title}: tp={tp} fp={fp} fn={fn} cls_err={ce} geom_pairs={geom} | "
        f"报出率={report_rate * 100:.2f}% 正确率={acc_rate * 100:.2f}% 错误率={err_rate * 100:.2f}% "
        f"漏检率(仅几何无框)={miss_fn_rate * 100:.2f}% 类错率={cls_err_rate * 100:.2f}% "
        f"召回缺口={recall_gap * 100:.2f}% 总偏差率=max(召回缺口,错误率)={total_dev * 100:.2f}%"
    )


def _accumulate_seg_validation(
    stat_total: dict[str, int],
    stat_by_cls: dict[str, dict[str, int]],
    preds: list[dict],
    gts: list[dict],
    *,
    img_w: int,
    img_h: int,
    iou_threshold: float,
    use_mask_iou: bool,
    class_merge: dict[str, list[str]] | None,
) -> None:
    """几何匹配 + 按 merge_map 统计 tp/fp/fn/cls_err（口径对齐 predict_size_validate_lib）。"""
    merge = class_merge

    def _pred_skip(r: dict) -> bool:
        return is_metric_ignored_other(str(r.get("cls_name", "") or ""), merge)

    def _gt_skip(g: dict) -> bool:
        return is_metric_ignored_other(str(g.get("name", "") or ""), merge)

    pred_eval_idx = [i for i, r in enumerate(preds) if not _pred_skip(r)]
    gt_eval_idx = [j for j, g in enumerate(gts) if not _gt_skip(g)]

    if not gts:
        stat_total["fp"] += len(pred_eval_idx)
        for i in pred_eval_idx:
            pn = _eval_normalize_class_name(preds[i].get("cls_name", ""), merge)
            _inc_stat_by_cls(stat_by_cls, pn, "pred", 1)
            _inc_stat_by_cls(stat_by_cls, pn, "fp", 1)
        return

    if not gt_eval_idx:
        stat_total["fp"] += len(pred_eval_idx)
        for i in pred_eval_idx:
            pn = _eval_normalize_class_name(preds[i].get("cls_name", ""), merge)
            _inc_stat_by_cls(stat_by_cls, pn, "pred", 1)
            _inc_stat_by_cls(stat_by_cls, pn, "fp", 1)
        return

    preds_eval = [preds[i] for i in pred_eval_idx]
    gts_eval = [gts[j] for j in gt_eval_idx]
    matches_ev, matched_p_ev, matched_g_ev = match_pred_gt_polygon(
        preds_eval,
        gts_eval,
        img_w=img_w,
        img_h=img_h,
        iou_threshold=iou_threshold,
        use_mask_iou=use_mask_iou,
    )
    matches = [(pred_eval_idx[pi], gt_eval_idx[gj], sc) for pi, gj, sc in matches_ev]
    matched_p = {pred_eval_idx[i] for i in matched_p_ev}
    matched_g = {gt_eval_idx[j] for j in matched_g_ev}
    pred_to_gt = {pi: gj for pi, gj, _ in matches}

    stat_total["geom_pairs"] += len(matches)

    for j in gt_eval_idx:
        gn = _eval_normalize_class_name(gts[j].get("name", ""), merge)
        _inc_stat_by_cls(stat_by_cls, gn, "gt", 1)
    for i in pred_eval_idx:
        pn = _eval_normalize_class_name(preds[i].get("cls_name", ""), merge)
        _inc_stat_by_cls(stat_by_cls, pn, "pred", 1)

    for i in range(len(preds)):
        if _pred_skip(preds[i]):
            continue
        pn = _eval_normalize_class_name(preds[i].get("cls_name", ""), merge)
        if i not in matched_p:
            stat_total["fp"] += 1
            _inc_stat_by_cls(stat_by_cls, pn, "fp", 1)
        else:
            gj = pred_to_gt[i]
            pred_cls = preds[i].get("cls_name", "")
            gt_name = gts[gj].get("name", "")
            gn = _eval_normalize_class_name(gt_name, merge)
            if _eval_is_class_match(str(pred_cls), str(gt_name), merge):
                stat_total["tp"] += 1
                _inc_stat_by_cls(stat_by_cls, gn, "tp", 1)
            else:
                stat_total["cls_err"] += 1
                stat_total["fp"] += 1
                _inc_stat_by_cls(stat_by_cls, gn, "cls_err", 1)
                _inc_stat_by_cls(stat_by_cls, pn, "fp", 1)

    stat_total["fn"] += len(gt_eval_idx) - len(matched_g_ev)
    for j in gt_eval_idx:
        if j not in matched_g:
            gn = _eval_normalize_class_name(gts[j].get("name", ""), merge)
            _inc_stat_by_cls(stat_by_cls, gn, "fn", 1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # ----------------------- 模型与推理参数 -----------------------
    seg_model_path = os.path.expanduser(
        # "/Users/shunyaoyin/Documents/code/ai-company/insect/doc/测试结果/大虫框选/v2.9-0518/best-2.7.1-s.pt"
        # "/Users/shunyaoyin/Documents/code/ai-company/insect/doc/测试结果/大虫框选/v2.11/best-2.11-seg.pt"
        # "/Users/shunyaoyin/Documents/code/ai-company/insect/doc/测试结果/大虫框选/v2.12/best-2.12.2-seg.pt"
        "/Users/shunyaoyin/Documents/code/ai-company/insect/doc/测试结果/大虫框选/v2.12/best-2.12.5-seg.pt"
        # "/Users/shunyaoyin/Documents/code/ai-company/insect/doc/测试结果/大虫框选/v2.12/best-2.12.6-seg.pt"
    )
    # 分类模型：None 则仅输出分割类别；配置后对每个分割实例按 polygon 裁剪再分类
    # cls_model_path = "/Users/shunyaoyin/Documents/code/ai-company/insect/doc/测试结果/分类测试/v1-20260424-all-large/best-1.0.pt"
    # cls_model_path = "/Users/shunyaoyin/Documents/code/ai-company/insect/doc/测试结果/分类测试/v2-20260428-all-large/best-2.0.pt"
    cls_model_path = "/Users/shunyaoyin/Documents/code/ai-company/insect/doc/测试结果/分类测试/v2.3-all/best-2.3.pt"
    # cls_model_path = "/Users/shunyaoyin/Documents/code/ai-company/insect/doc/测试结果/分类测试/v2.5-seg-cls/best.pt"
    # cls_model_path = "/Users/shunyaoyin/Documents/code/ai-company/insect/doc/测试结果/分类测试/v2.6/best-e1.pt"
    cls_model_path = "/Users/shunyaoyin/Documents/code/ai-company/insect/doc/测试结果/分类测试/v2.6/2.6.2/temp.pt"
    cls_model_path = None
    # 仅对这些「分割模型输出的类名」跑分类；None/[] 表示全部实例都分类
    cls_list = None

    conf_thresh = 0.25
    conf_merge = 0.1
    ior_threshold = 0.5
    clip_size = 0
    overlap_size = 200
    # infer_imgsz = 640
    infer_imgsz = 960
    # infer_imgsz = 1280
    pad_full_image_to_square = True
    detect_nms_iou = 0.75
    detect_max_det = 1000
    detect_nms_agnostic = True
    min_instance_size = 3

    # insect_alg_seg.json：相对 script/ 目录；detect_conf / cls_conf / dia
    insect_alg_seg_path = INSECT_ALG_SEG_JSON_REL
    insect_alg_profile: str | None = "insect"
    # dia 过滤：None 表示不按类名限定（profile 有 dia 时过滤全部实例）；否则仅过滤列出的类名
    insect_dia_filter_class_keys: set[str] | None = None
    cls_top1_conf_threshold: float | None = 0.3  # 无 JSON 命中时的全局分类门限

    # 分类前处理（与 train_cls / PredictSize 一致）
    cls_pad_square = True
    cls_gray_binarize = False
    crop_pad_ratio = 0.05
    # 分类裁剪：False=多边形掩码抠图（默认）；True=实例外接框矩形截取
    cls_crop_from_bbox = False

    # ----------------------- 绘制开关 -----------------------
    enable_draw_polygons = True
    enable_draw_bbox = True
    enable_draw_center_point_and_label = True
    cls_output_top_n = 2
    polygon_fill_alpha = 0.22

    # ----------------------- 验证与类别 -----------------------
    enable_eval_with_xml = True
    val_match_use_mask_iou = True
    val_geom_threshold = 0.5
    # 类别合并组（由 cls_merge.py 派生）+ insect 通配：标注/预测任一侧为 insect 时与任意昆虫类互认正确
    class_merge_to_groups: dict[str, list[str]] | None = build_eval_class_merge(
        INSECT_CLS_MERGE_MAP,
        insect_wildcard=True,
    )
    sort_stat_by_acc = True

    # ----------------------- 输入输出 -----------------------
    # input_path = "/Users/shunyaoyin/Documents/code/datasets/insect-data/测试集/生产"
    input_path = "/Users/shunyaoyin/Documents/code/datasets/insect-data/测试集/北京0518"
    # input_path = "/Users/shunyaoyin/Documents/code/datasets/insect-data/测试集/比赛-北京"
    input_path = "/Users/shunyaoyin/Documents/code/datasets/insect-data/测试集/dachong-测试数据集"
    # input_path = "/Users/shunyaoyin/Documents/code/datasets/insect-data/测试集/dachong-标准测试集"
    # input_path = "/Users/shunyaoyin/Documents/code/datasets/insect-data/测试集/dachong-检测测试集泛化"
    output_dir = input_path + "-2.12.5"
    incremental_skip_done = False
    predict_debug = False
    debug_clip = False

    predictor = PredictSeg(
        seg_model_path=seg_model_path,
        cls_model_path=cls_model_path,
        cls_list=cls_list,
        conf_thresh=conf_thresh,
        conf_merge=conf_merge,
        ior_threshold=ior_threshold,
        device=None,
        augment=False,
        cls_pad_square=cls_pad_square,
        cls_gray_binarize=cls_gray_binarize,
        insect_alg_path=insect_alg_seg_path,
        insect_alg_profile=insect_alg_profile,
        seg_imgsz=infer_imgsz,
        nms_iou=detect_nms_iou,
        max_det=detect_max_det,
        nms_agnostic=detect_nms_agnostic,
        crop_pad_ratio=crop_pad_ratio,
        cls_crop_from_bbox=cls_crop_from_bbox,
    )

    input_p, image_files = collect_images(input_path)
    insect_alg = load_insect_alg(insect_alg_seg_path)
    dia_range = resolve_seg_dia_range(
        insect_alg,
        insect_alg_profile=insect_alg_profile,
        cls_list=cls_list,
    )
    print(f"共找到 {len(image_files)} 张图片")
    print(f"算法阈值: {resolve_insect_alg_seg_path(insect_alg_seg_path)}")
    if dia_range is not None:
        print(
            f"对角线过滤: [{dia_range[0]:.0f}, {dia_range[1]:.0f}] px "
            f"(profile={insect_alg_profile!r})"
        )
    print(
        f"分类: {'已启用' if cls_model_path else '未启用'} "
        f"crop={'bbox' if cls_crop_from_bbox else 'polygon'} "
        f"pad_square={cls_pad_square} gray_binarize={cls_gray_binarize}"
    )
    if enable_eval_with_xml and class_merge_to_groups:
        n_groups = len(class_merge_to_groups)
        has_insect_wc = "*" in (class_merge_to_groups.get("insect") or [])
        print(
            f"评估类别合并: {n_groups} 组"
            + ("；insect 通配 [*] 已启用（粗标 insect 与任意昆虫类互认）" if has_insect_wc else "")
        )

    stat_total = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "cls_err": 0,
        "geom_pairs": 0,
        "img_with_xml": 0,
        "dia_filtered": 0,
    }
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
        rows = predictor.predict(
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
            cls_top1_conf_threshold=cls_top1_conf_threshold,
            cls_pad_square=cls_pad_square,
            cls_gray_binarize=cls_gray_binarize,
            cls_crop_from_bbox=cls_crop_from_bbox,
            debug=predict_debug,
            debug_clip=debug_clip,
        )

        for r in rows:
            x1, y1, x2, y2 = bbox_from_row(r)
            r["x1"], r["y1"], r["x2"], r["y2"] = x1, y1, x2, y2

        results_visible = list(rows)
        if dia_range is not None:
            n_before = len(results_visible)
            results_visible, n_dia_drop = filter_rows_by_bbox_diag_range(
                results_visible,
                dia_range[0],
                dia_range[1],
                class_keys=insect_dia_filter_class_keys,
            )
            stat_total["dia_filtered"] += n_dia_drop
            if n_dia_drop and predict_debug:
                print(
                    f"  dia 过滤: {n_before} -> {len(results_visible)} "
                    f"(剔除 {n_dia_drop})"
                )

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
            _accumulate_seg_validation(
                stat_total,
                stat_by_cls,
                results_visible,
                gts,
                img_w=w_img,
                img_h=h_img,
                iou_threshold=val_geom_threshold,
                use_mask_iou=val_match_use_mask_iou,
                class_merge=class_merge_to_groups,
            )

        _save_image_and_xml(
            img,
            results_visible,
            out_dir=out_dir,
            rel_path=rel_path,
            draw_polygons=enable_draw_polygons,
            draw_bbox=enable_draw_bbox,
            draw_center_point_and_label=enable_draw_center_point_and_label,
            cls_output_top_n=cls_output_top_n,
            polygon_alpha=polygon_fill_alpha,
        )

        if gts:
            print(
                f"[{idx}/{len(image_files)}] {rel_path} "
                f"pred={len(results_visible)} gt={len(gts)}"
            )
        else:
            print(
                f"[{idx}/{len(image_files)}] {rel_path} pred={len(results_visible)} (无标注 xml)"
            )

    print(f"\n完成。输出目录: {output_dir}")
    if skipped:
        print(f"增量跳过 {skipped} 张")
    if stat_total.get("dia_filtered", 0) > 0:
        print(f"对角线范围误报过滤共剔除 {stat_total['dia_filtered']} 个实例")
    if stat_total["img_with_xml"] > 0:
        print("======== 分割验证汇总（merge_map 类别合并，仅对有 xml 的图统计） ========")
        _print_overall_stat_summary("分割+分类验证汇总", stat_total)
        _print_stat_by_cls(
            "分割 按合并类统计",
            stat_by_cls,
            sort_by_acc=sort_stat_by_acc,
        )
