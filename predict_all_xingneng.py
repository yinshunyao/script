#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Detail  : 虫情 pipeline 本地性能测试（predict_all 精简版）。
#           图片目录/文件名约定与 test_api.py 一致，便于与 HTTP API 耗时对比。
#           默认关闭校验、可视化与导出，仅统计 pipeline.predict 耗时。

from __future__ import annotations

import json
import logging
import os
import platform
import re
import shutil
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

_FILE = Path(__file__).resolve()
_SCRIPT_DIR = _FILE.parent
_ROOT = _FILE.parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from script.config_paths import (  # noqa: E402
    INSECT_ALG_ALL_JSON_REL,
    INSECT_ALG_LAUNCHER_JSON,
    compose_insect_alg_from_profile,
    is_insect_alg_profile_path,
    load_insect_alg_launcher_dict,
    read_run_model_profile,
    resolve_clip_profiles_enable,
    resolve_detect_seg_batch_size,
    resolve_effective_insect_alg_path,
    resolve_insect_alg_all_path,
    resolve_predict_cfg_value,
    resolve_use_gpu_crop,
)
DEFAULT_INSECT_ALG_ALL_JSON = resolve_insect_alg_all_path()

from script.predict.model_cls import ModelCls
from script.predict.model_cls_crop import cls_infer_pad_square, cls_infer_to_gray
from script.predict.model_gpu_crop import (
    cls_job_can_defer_gpu_crop,
    get_gpu_crop_session,
    gpu_crop_session_scope,
)
from script.predict.model_cls_batch import (
    ClsBatchJob,
    cfg_uses_yolo_cls_batch,
    cls_batch_group_key,
    make_cls_crop,
    resolve_cls_batch_size,
    run_cls_job_batches,
)
from script.predict.model_cls_factory import ClsModel, cls_cache_key, create_classifier
from script.predict.model_trt import resolve_inference_model_path, resolve_trt_switch, rewrite_model_dir_paths, set_inference_global_cfg
from script.predict.model_yolo_cache import clear_yolo_cache
from script.predict.model_detect import ClipProfile, format_clip_slice_label_suffix, resolve_clip_profiles
from script.predict.model_seg import _bbox_from_polygon
from script.predict_seg import PredictSeg
from script.predict_seg_lib import (
    PIC_EXT,
    _ROUTE_PATTERNS_KEY,
    bbox_diag_px,
    bbox_diag_px_from_row,
    build_alg_table_from_out,
    collect_images,
    crop_instance_bgr_from_bbox,
    crop_instance_bgr_from_polygon,
    filter_rows_by_bbox_diag_range,
    resolve_cls_crop_background,
    resolve_cls_pad_color,
    compute_mask_bbox_fill_ratio_from_row,
    mask_rate_passes,
    max_binarize_black_in_big_ratio,
    find_best_in_big_big_row,
    build_in_big_debug_panel,
    save_in_big_debug_panel,
    collect_in_big_debug_parts_for_row,
    resolve_cls_top1_threshold,
    resolve_out_route_entry,
    row_location_to_bbox,
)
from script.predict_size import PredictSize, write_pascal_voc_xml
from script.tools.roi_heyifei import apply_roi_preprocess, roi_circle_from_apply
from script.predict_size_validate_lib import (
    _auto_draw_params,
    _draw_cn_text,
    _export_overall_summary_csv,
    _export_stat_by_cls_csv,
    _print_overall_stat_summary,
    _print_stat_by_cls,
    _save_type_confusion_crops_for_branch,
    _save_type_confusion_crops_for_filename_gt,
    _std_eval_collect_confusion,
    _std_eval_collect_confusion_filename_gt,
    _std_eval_save_branch,
    build_eval_class_display_index,
    build_eval_focus_set,
    build_class_tier_equivalence,
    ClassTierEquivalence,
    is_class_match,
    is_metric_ignored_other,
    build_filename_pseudo_gt_objects,
    infer_gt_label_from_filename_stem,
    load_class_tier_equivalence,
    load_eval_label_alias_map,
    match_pred_gt,
    match_pred_gt_ior,
    merge_stat_by_cls,
    normalize_class_name,
    parse_pascal_voc_objects,
    parse_pascal_voc_pred_objects,
    resolve_eval_label,
    voc_xml_has_dual_confidence,
    resolve_cjk_font_path,
    sum_stat_by_cls_focus,
)

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover
    Image = None
    ImageDraw = None
    ImageFont = None

# --------------------------------------------------------------------------- #
#  配置解析（相对 script/ 目录；开关关闭时可省略子字段）
# --------------------------------------------------------------------------- #

_DETECT_DEFAULTS: dict[str, Any] = {
    "enable": True,
    "model_type": "detect",
    "conf_thresh": 0.3,
    "conf_merge": 0.1,
    "conf_merge_draw": 0.01,
    "iou_threshold": 0.3,
    "ior_threshold": 0.4,
    "clip_size": 640,
    "overlap_size": 120,
    "edge_reject_distance": 5,
    "half": False,
    "augment": False,
    "offset_rate": 1.2,
    "inner_boxes_fp_threshold": 8,
    "bin_dark_ratio_min": 0.2,
    # dia_switch: false 时关闭对角线尺寸过滤；true 时再按 dia_w 与图片宽度决定是否生效
    "dia_switch": True,
    # dia_w: None -> dia 永远生效（保持历史行为）；list[int] -> 仅当图片宽度命中其中之一（默认±1%）时 dia 才生效
    "dia_w": None,
}

_SEGMENT_DEFAULTS: dict[str, Any] = {
    "enable": True,
    "model_type": "segment",
    "conf_thresh": 0.25,
    "conf_merge": 0.1,
    "conf_merge_draw": 0.01,
    "ior_threshold": 0.5,
    "clip_size": 0,
    "overlap_size": 0,
    "augment": False,
    "seg_imgsz": 0,
    # 高分辨率分割掩码（Ultralytics retina_masks）；默认关闭，按需在 JSON 中打开
    "retina_masks": False,
    # 多边形相似实例合并（边缘贴合比例 或 包含率 达标即合并）；默认关闭，开启时按虫体轮廓去重
    "poly_merge": False,
    "poly_merge_edge_px": 5.0,
    "poly_merge_edge_ratio": 0.5,
    "poly_merge_contain_ratio": 0.7,
    "poly_merge_cross_class": True,
    "poly_merge_max_points": 80,
    "to_square": True,
    "pad_full_image_to_square": True,
    "crop_pad_ratio": 0.05,
    "min_instance_size": 3,
    # dia_switch: false 时关闭根级 dia 对角线过滤（与 detect 根行为一致）
    "dia_switch": True,
    # 检测/分割送入 YOLO 前：灰度 CLAHE 对比度增强（ch=1 模型或 to_gray 场景可试开）
    "gray_contrast_enhance": False,
    "gray_clahe_clip": 2.0,
    "gray_clahe_tile": 8,
    "gray_contrast_debug_save": False,
}

_DETECT_DEFAULTS["to_square"] = False


def resolve_clip_profiles_from_cfg(
    cfg: dict[str, Any],
    *,
    global_cfg: dict[str, Any] | None = None,
) -> list[ClipProfile]:
    """
    从根模型 JSON 配置解析滑窗切片列表（单套或多套）。

    - ``clip_profiles_enable: false``（``predict_cfg``）：忽略 ``clip_profiles``，回退顶层 ``clip_size``/``overlap_size``。
    - ``clip_profiles[].enable: false``：跳过该套（见 ``resolve_clip_profiles``）。
    """
    raw_profiles = cfg.get("clip_profiles")
    if raw_profiles is not None and not resolve_clip_profiles_enable(
        cfg, global_cfg=global_cfg
    ):
        raw_profiles = None
    return resolve_clip_profiles(
        clip_size=int(cfg.get("clip_size", 0) or 0),
        overlap_size=int(cfg.get("overlap_size", 0) or 0),
        clip_start=int(cfg.get("clip_start", 0) or 0),
        clip_profiles=raw_profiles,
    )


def resolve_clip_batch_size_from_cfg(
    cfg: dict[str, Any],
    *,
    global_cfg: dict[str, Any] | None = None,
) -> int:
    """detect/seg 滑窗 batch（见 ``resolve_detect_seg_batch_size``）。"""
    return resolve_detect_seg_batch_size(cfg, global_cfg=global_cfg)


_CLS_NODE_DEFAULTS: dict[str, Any] = {
    "enable": True,
    "from_bbox": True,
    "to_square": True,
    "cls_conf": 0.3,
    "gray_binarize": False,
    "to_gray": False,
    "crop_pad_ratio": 0.05,
    # dia_switch: false 时本层 cls 路由 out 项上的 dia 过滤关闭（与根模型 dia_switch 独立）
    "dia_switch": True,
    # cls_batch_size：YOLO 嵌套分类 crop 批量大小（默认 32，见 model_cls_batch）
    "cls_backend": "auto",
    "timm_model": "",
    "image_size": 0,
    # cls_crop_background：默认空 → polygon 不填底；white/black 或 BGR 列表
    # cls_pad_color：默认空 → 白边补方；white/black 或 BGR 列表
}


@dataclass(frozen=True)
class ValidationFocusConfig:
    """验证/绘图关注类：tier 清单来自顶层 ``postprocess``；``report_classes`` 由报出开关决定。"""

    top1_classes: tuple[str, ...]
    top2_classes: tuple[str, ...]
    top3_classes: tuple[str, ...]
    other_classes: tuple[str, ...]
    background_classes: tuple[str, ...]
    report_classes: tuple[str, ...]
    report_all_switch: bool
    run_model: str


@dataclass
class PredictPhaseSample:
    """单张图 ``pipeline.predict`` 分阶段耗时（秒）。"""

    detect_s: float = 0.0
    seg_s: float = 0.0
    cls_batch_s: float = 0.0
    post_s: float = 0.0
    roots_parallel: bool = False

    @property
    def total_s(self) -> float:
        return self.detect_s + self.seg_s + self.cls_batch_s + self.post_s

    @property
    def wall_s(self) -> float:
        """detect/seg 根并发时，根阶段墙钟近似 max(detect, seg)+cls+post。"""
        if self.roots_parallel:
            return max(self.detect_s, self.seg_s) + self.cls_batch_s + self.post_s
        return self.total_s


_PHASE_LABELS: tuple[tuple[str, str], ...] = (
    ("detect_s", "detect"),
    ("seg_s", "seg"),
    ("cls_batch_s", "cls-batch"),
    ("post_s", "post"),
)


def merge_predict_phase_samples(
    samples: list[PredictPhaseSample],
    *,
    overlap: bool = False,
    roots_parallel: bool = False,
) -> PredictPhaseSample:
    if not samples:
        return PredictPhaseSample(roots_parallel=roots_parallel)
    merged = PredictPhaseSample(roots_parallel=roots_parallel)
    for attr, _ in _PHASE_LABELS:
        vals = [float(getattr(s, attr)) for s in samples]
        merged_val = max(vals) if overlap else sum(vals)
        setattr(merged, attr, merged_val)
    return merged


class PredictPhaseRecorder:
    """``pipeline.predict`` 分阶段耗时采集（detect / seg / cls-batch / post）。"""

    def __init__(self, *, enabled: bool = False) -> None:
        self.enabled = bool(enabled)
        self.samples: list[PredictPhaseSample] = []
        self._tls = threading.local()
        self._samples_lock = threading.Lock()

    @property
    def _current(self) -> PredictPhaseSample | None:
        return getattr(self._tls, "current", None)

    @_current.setter
    def _current(self, value: PredictPhaseSample | None) -> None:
        self._tls.current = value

    def begin_image(self) -> None:
        if self.enabled:
            self._current = PredictPhaseSample()

    def end_image(self) -> PredictPhaseSample | None:
        cur = self._current
        if cur is None:
            return None
        with self._samples_lock:
            self.samples.append(cur)
        self._current = None
        return cur

    def add(self, phase: str, seconds: float) -> None:
        if self._current is None or seconds <= 0:
            return
        attr = f"{phase}_s"
        if hasattr(self._current, attr):
            setattr(self._current, attr, getattr(self._current, attr) + float(seconds))

    def last_sample(self) -> PredictPhaseSample | None:
        return self.samples[-1] if self.samples else None


def format_predict_phase_line(sample: PredictPhaseSample | None) -> str:
    if sample is None:
        return ""
    parts = [
        f"{label}={getattr(sample, attr):.3f}s"
        for attr, label in _PHASE_LABELS
        if getattr(sample, attr) > 0.0005
    ]
    if not parts:
        return ""
    suffix = f" sum={sample.total_s:.3f}s]"
    if sample.roots_parallel and abs(sample.wall_s - sample.total_s) > 0.0005:
        suffix = f" sum={sample.total_s:.3f}s wall≈{sample.wall_s:.3f}s∥]"
    return " phases[" + " ".join(parts) + suffix


def print_predict_phase_summary(
    recorder: PredictPhaseRecorder,
    *,
    image_count: int | None = None,
) -> None:
    samples = recorder.samples
    if not samples:
        return
    n = image_count if image_count is not None and image_count > 0 else len(samples)
    n = max(1, min(n, len(samples)))
    totals = {
        attr: sum(getattr(s, attr) for s in samples)
        for attr, _ in _PHASE_LABELS
    }
    grand = sum(totals.values()) or 1e-9
    print("======== 分阶段耗时（pipeline.predict，detect/seg/cls-batch/post） ========")
    for attr, label in _PHASE_LABELS:
        total = totals[attr]
        print(
            f"  {label:9s} 总={total:7.3f}s  均/张={total / n:6.3f}s  "
            f"占比={100.0 * total / grand:5.1f}%"
        )
    print(f"  {'合计':9s} 总={grand:7.3f}s  均/张={grand / n:6.3f}s")
    if any(s.roots_parallel for s in samples):
        wall_grand = sum(s.wall_s for s in samples)
        print(
            f"  {'墙钟∥':9s} 总={wall_grand:7.3f}s  均/张={wall_grand / n:6.3f}s"
            "  （detect/seg 根并发，墙钟≈max(detect,seg)+cls+post）"
        )


_root_run_tls = threading.local()
def _postprocess_block(alg_config: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(alg_config, dict):
        return {}
    block = alg_config.get("postprocess")
    return block if isinstance(block, dict) else {}


def resolve_postprocess_debug(alg_config: dict[str, Any] | None) -> bool:
    """``postprocess.debug``：开启时保存 in_big 等后处理调试图到 ``eval_metrics/debug``。"""
    return bool(_postprocess_block(alg_config).get("debug", False))


def collect_draw_filter_model_sources(
    alg_config: dict[str, Any] | None,
) -> frozenset[str]:
    """
    收集 ``models.{root_id}.draw_filter=true`` 的根模型 id。

    未配置时视为 ``false``，不绘制该模型 filtered 实例。
    """
    if not isinstance(alg_config, dict):
        return frozenset()
    models = alg_config.get("models")
    if not isinstance(models, dict):
        return frozenset()
    out: set[str] = set()
    for root_id, raw in models.items():
        if not isinstance(raw, dict):
            continue
        if bool(raw.get("draw_filter", False)):
            out.add(str(root_id))
    return frozenset(out)


def resolve_eval_metrics_debug_dir(output_root: str | Path) -> Path:
    """``{output_root}/eval_metrics/debug``，与验证汇总 ``eval_metrics`` 同根。"""
    return Path(output_root).expanduser().resolve() / "eval_metrics" / "debug"


def _expand_class_names_with_aliases(
    names: tuple[str, ...] | list[str],
    aliases: dict[str, Any] | None,
) -> frozenset[str]:
    return build_class_tier_equivalence(aliases).expand_names(names)


def _class_matches_report_allowed(
    class_name: str | None,
    allowed: frozenset[str],
    aliases: dict[str, Any] | None,
) -> bool:
    name = str(class_name or "").strip()
    if not name:
        return False
    if name in allowed:
        return True
    tier_equiv = build_class_tier_equivalence(aliases)
    return any(tier_equiv.are_equivalent(name, item) for item in allowed)


def parse_shenchan_map_classes(
    postprocess: dict[str, Any] | None,
) -> dict[str, str]:
    """解析 ``postprocess.shenchan_map_classes``：生产报出前 key→value 类名映射。"""
    if not isinstance(postprocess, dict):
        return {}
    raw = postprocess.get("shenchan_map_classes")
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for key, value in raw.items():
        k = str(key or "").strip()
        v = str(value or "").strip()
        if k and v:
            out[k] = v
    return out


def apply_shengchan_class_name(
    class_name: str | None,
    *,
    class_map: dict[str, str] | None,
) -> str:
    """类名命中 ``shenchan_map_classes`` 时返回 value，否则返回原类名。"""
    name = str(class_name or "").strip()
    if not name or not class_map:
        return name
    return class_map.get(name, name)


def apply_shengchan_class_map_to_row(
    row: dict[str, Any],
    *,
    class_map: dict[str, str] | None,
    cn_index: dict[str, str] | None = None,
) -> bool:
    """
    就地改写单条推理结果的生产映射类名。

    返回是否发生了映射。
    """
    if not class_map:
        return False
    raw_name = str(row.get("name") or row.get("cls_name") or "").strip()
    mapped = apply_shengchan_class_name(raw_name, class_map=class_map)
    if not mapped or mapped == raw_name:
        return False
    row["name"] = mapped
    if "cls_name" in row:
        row["cls_name"] = mapped
    cn = _resolve_cn_display_for_class(mapped, cn_index=cn_index)
    if cn:
        row["cn_name"] = cn
    return True


def resolve_report_allowed_class_names(
    alg_config: dict[str, Any] | None,
) -> frozenset[str] | None:
    """
    报出类白名单。

    - ``report_all_switch=true``：返回 ``None``（不过滤，全部报出；生产映射仍可在报出阶段应用）；
    - ``report_all_switch=false``：仅 ``top1`` + ``top2`` + ``top3`` + ``other``；
      ``background_classes`` 不报出。

    白名单与 ``postprocess`` 的 top1/top2/top3/other **key** 对齐；
    ``out`` 条目的 ``infer_name`` 仅写入 ``infer_name`` 字段，供对外 XML 与外部测试集匹配。
    """
    postprocess = _postprocess_block(alg_config)
    if bool(postprocess.get("report_all_switch", False)):
        return None
    top1 = _dedupe_class_names(
        _parse_class_name_list(postprocess.get("top1_classes"))
    )
    top2 = _dedupe_class_names(
        _parse_class_name_list(postprocess.get("top2_classes"))
    )
    top3 = _dedupe_class_names(
        _parse_class_name_list(postprocess.get("top3_classes"))
    )
    other = _dedupe_class_names(
        _parse_class_name_list(postprocess.get("other_classes"))
    )
    allowed = list(top1) + list(top2) + list(top3) + list(other)
    aliases = postprocess.get("class_tier_aliases")
    return _expand_class_names_with_aliases(allowed, aliases)


def _parse_class_name_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    out: list[str] = []
    for item in value:
        s = str(item or "").strip()
        if s:
            out.append(s)
    return out


def _dedupe_class_names(names: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return tuple(out)


def _is_out_route_key(key: str) -> bool:
    k = str(key or "").strip()
    return not k or k == "*" or k.startswith("[")


def _collect_report_classes_from_out_node(node: Any) -> set[str]:
    """递归收集 ``out`` 树中可报出的叶类名（路由 **key**；``infer_name`` 不参与 tier 清单）。"""
    keys: set[str] = set()
    if not isinstance(node, dict):
        return keys
    out_table = node.get("out")
    if not isinstance(out_table, dict):
        return keys
    for cls_key, entry in out_table.items():
        if not isinstance(entry, dict) or not _out_entry_is_enabled(entry):
            continue
        models = entry.get("models") or {}
        cls_cfg = models.get("cls") if isinstance(models, dict) else None
        if not isinstance(cls_cfg, dict):
            cls_cfg = entry.get("cls")
        if isinstance(cls_cfg, dict):
            cls_out = cls_cfg.get("out")
            if isinstance(cls_out, dict) and cls_out:
                for leaf_key, leaf_entry in cls_out.items():
                    if not isinstance(leaf_entry, dict):
                        continue
                    if not _out_entry_is_enabled(leaf_entry):
                        continue
                    lk = str(leaf_key).strip()
                    if lk and not _is_out_route_key(lk):
                        keys.add(lk)
                keys.update(_collect_report_classes_from_out_node(entry))
                continue
        if not _is_out_route_key(str(cls_key)):
            keys.add(str(cls_key).strip())
        keys.update(_collect_report_classes_from_out_node(entry))
    return keys


def resolve_validation_focus_config(
    alg_config: dict[str, Any] | None,
    *,
    root_ids: list[str] | None = None,
) -> ValidationFocusConfig:
    """
    从 ``insect_alg_all`` 顶层 ``postprocess`` 读取 tier 清单与报出开关。

    ``report_classes`` 由 ``report_all_switch`` 决定（与 ``resolve_report_allowed_class_names`` 一致）；
    ``report_all_switch=true`` 时为 top1+top2+top3+other 全量昆虫类清单。
    """
    if not isinstance(alg_config, dict):
        return ValidationFocusConfig((), (), (), (), (), (), False, "baipai")
    postprocess = _postprocess_block(alg_config)
    report_all_switch = bool(postprocess.get("report_all_switch", False))
    run_model = str(alg_config.get("run_model") or "baipai").strip().lower() or "baipai"
    top1 = _parse_class_name_list(postprocess.get("top1_classes"))
    top2 = _parse_class_name_list(postprocess.get("top2_classes"))
    top3 = _parse_class_name_list(postprocess.get("top3_classes"))
    other = _parse_class_name_list(postprocess.get("other_classes"))
    background = _parse_class_name_list(postprocess.get("background_classes"))
    allowed = resolve_report_allowed_class_names(alg_config)
    if allowed is None:
        report = list(top1) + list(top2) + list(top3) + list(other)
    else:
        report = list(allowed)
    return ValidationFocusConfig(
        top1_classes=_dedupe_class_names(top1),
        top2_classes=_dedupe_class_names(top2),
        top3_classes=_dedupe_class_names(top3),
        other_classes=_dedupe_class_names(other),
        background_classes=_dedupe_class_names(background),
        report_classes=_dedupe_class_names(report),
        report_all_switch=report_all_switch,
        run_model=run_model,
    )


@dataclass(frozen=True)
class CompetitionCountingSummary:
    """北京比赛识别计数准确率汇总（见 ``script/config/北京比赛规则.md``）。"""

    mode: str
    class_rows: tuple[tuple[str, int, int, float], ...]
    avg_accuracy_percent: float
    corrected_accuracy_percent: float | None
    extra_non_specimen_pred: int
    total_specimen_gt: int
    penalty_percent: float


def resolve_competition_counting_focus(
    focus: ValidationFocusConfig,
    *,
    class_merge: dict[str, list[str]] | None = None,
    label_alias_map: dict[str, str] | None = None,
) -> frozenset[str]:
    """
    北京比赛计分关注类：配置报出的所有昆虫（top1+top2+top3+other），不含 background。

    生产 / 摆拍均按 ``report_classes``；二者差异仅在摆拍是否对额外非标样多检做修正惩罚。
    """
    names = list(focus.report_classes)
    if not names:
        return frozenset()
    return build_eval_focus_set(
        names,
        merge=class_merge,
        label_alias_map=label_alias_map,
    )


def _count_report_tier_filtered_preds(
    filtered: list[dict[str, Any]] | None,
) -> int:
    """``report_tier`` 过滤（background 等非报出类）的预测框数；不参与摆拍修正惩罚。"""
    n = 0
    for r in filtered or []:
        if str(r.get("filter_reason") or "").strip() == "report_tier":
            n += 1
    return n


def competition_counting_accuracy_percent(gt: int, pred: int) -> float:
    """
    单类识别计数准确率（%）= (1 - |pred-gt|/gt) × 100。

    鉴定数 gt=0 且识别数 pred≥1 时按 0；公式结果为负时按 0。
    """
    gt_n = int(gt)
    pred_n = int(pred)
    if gt_n <= 0:
        return 0.0 if pred_n >= 1 else 0.0
    acc = (1.0 - abs(float(pred_n - gt_n)) / float(gt_n)) * 100.0
    return max(0.0, float(acc))


def compute_competition_counting_summary(
    stat_by_cls: dict[str, dict[str, int]],
    focus: frozenset[str],
    *,
    run_model: str = "baipai",
) -> CompetitionCountingSummary | None:
    """
    按北京比赛规则汇总报出类 focus 内的计数准确率。

    - **生产**（``run_model=shengchan``）：配置报出各类的计数准确率算术平均；鉴定数=0 且
      识别数≥1 的单类准确率按 0 计并纳入平均；无摆拍修正；
    - **摆拍**（``run_model=baipai``）：仅 **标样类**（鉴定数>0）参与算术平均；鉴定数=0 的
      额外误报类不参与平均，其识别数计入修正惩罚（仅 focus 内 gt=0 且 pred≥1 的报出类；
      background / ``report_tier`` 过滤框不报出，不计入惩罚）。
    """
    if not stat_by_cls or not focus:
        return None
    mode_key = str(run_model or "baipai").strip().lower() or "baipai"
    is_baipai = mode_key == "baipai"
    specimen_rows: list[tuple[str, int, int, float]] = []
    extra_rows: list[tuple[str, int, int, float]] = []
    acc_vals: list[float] = []
    extra_from_stat = 0
    total_specimen_gt = 0
    for cls_name, s in stat_by_cls.items():
        key = str(cls_name)
        if key not in focus:
            continue
        gt_n = int(s.get("gt", 0))
        pred_n = int(s.get("pred", 0))
        if gt_n <= 0 and pred_n <= 0:
            continue
        acc = competition_counting_accuracy_percent(gt_n, pred_n)
        if gt_n > 0:
            specimen_rows.append((key, gt_n, pred_n, acc))
            acc_vals.append(acc)
            total_specimen_gt += gt_n
        elif pred_n >= 1:
            extra_rows.append((key, gt_n, pred_n, acc))
            extra_from_stat += pred_n
            if not is_baipai:
                acc_vals.append(acc)
    class_rows = specimen_rows + extra_rows
    if not class_rows:
        return None
    avg_acc = float(sum(acc_vals) / len(acc_vals)) if acc_vals else 0.0
    penalty = 0.0
    corrected: float | None = None
    extra_total = extra_from_stat if is_baipai else 0
    if is_baipai:
        if total_specimen_gt > 0:
            penalty = float(extra_total) / float(total_specimen_gt) * 100.0
        corrected = max(0.0, avg_acc - penalty)
    mode = report_mode_label(run_model=mode_key)
    class_rows.sort(key=lambda x: (-x[3], -x[1], -x[2], x[0]))
    return CompetitionCountingSummary(
        mode=mode,
        class_rows=tuple(class_rows),
        avg_accuracy_percent=avg_acc,
        corrected_accuracy_percent=corrected,
        extra_non_specimen_pred=extra_total,
        total_specimen_gt=total_specimen_gt,
        penalty_percent=penalty,
    )


def print_competition_counting_summary(
    summary: CompetitionCountingSummary | None,
    *,
    class_display_index: dict[str, str] | None = None,
) -> None:
    """打印北京比赛识别计数准确率（生产 / 摆拍）。"""
    if summary is None:
        return
    idx = class_display_index or {}
    print(
        f"======== 北京比赛识别计数准确率（{summary.mode}） ========"
    )
    print(
        "公式: 单类=(1-|识别-鉴定|/鉴定)×100；"
        + (
            "平均=标样类(鉴定>0)算术平均；误报类不参与平均"
            if summary.corrected_accuracy_percent is not None
            else "总分=配置报出各类计数准确率算术平均(鉴定0识别≥1按0计)"
        )
        + (
            ""
            if summary.corrected_accuracy_percent is None
            else "；摆拍修正=MAX(0,平均-(额外非标样数/标样总数)×100)"
        )
    )
    is_baipai = summary.corrected_accuracy_percent is not None
    specimen_rows = [r for r in summary.class_rows if r[1] > 0]
    extra_rows = [r for r in summary.class_rows if r[1] <= 0 and r[2] >= 1]

    def _print_counting_rows(rows: tuple[tuple[str, int, int, float], ...]) -> None:
        print("类别           | 鉴定数 | 识别数 | 计数准确率")
        print("---------------+--------+--------+-----------")
        for cls_key, gt_n, pred_n, acc in rows:
            label = idx.get(cls_key) or cls_key
            print(
                f"{label:<14} | {gt_n:>6} | {pred_n:>6} | {acc:>8.2f}%"
            )

    if is_baipai:
        if specimen_rows:
            print("【标样类 — 参与算术平均】")
            _print_counting_rows(tuple(specimen_rows))
        if extra_rows:
            print("【额外误报类 — 不参与算术平均，计入修正惩罚】")
            _print_counting_rows(tuple(extra_rows))
        if not specimen_rows and not extra_rows:
            _print_counting_rows(summary.class_rows)
    else:
        _print_counting_rows(summary.class_rows)
    if summary.corrected_accuracy_percent is not None:
        print(
            f"标样鉴定总数={summary.total_specimen_gt}  "
            f"额外非标样识别数={summary.extra_non_specimen_pred}"
        )
    print(f"各类算术平均准确率: {summary.avg_accuracy_percent:.2f}%")
    if summary.corrected_accuracy_percent is not None:
        print(
            f"摆拍修正惩罚: {summary.penalty_percent:.2f}%  "
            f"修正后准确率: {summary.corrected_accuracy_percent:.2f}%"
        )
    else:
        print(f"生产环境总分: {summary.avg_accuracy_percent:.2f}%")


def report_mode_label(*, run_model: str) -> str:
    """``run_model`` 对应的计分模式中文名。"""
    key = str(run_model or "baipai").strip().lower()
    if key == "shengchan":
        return "生产"
    if key == "other":
        return "其他"
    return "摆拍比赛"


def _current_os_key() -> str:
    """归一化操作系统标识：darwin / linux / windows。"""
    return platform.system().strip().lower()


def _resolve_model_dir(data: dict[str, Any]) -> str | None:
    """
    解析当前操作系统对应的模型根目录，按优先级：
    1) 环境变量 ``INSECT_MODEL_DIR`` 显式指定；
    2) 配置顶层 ``model_dir``（按 os key 取，如 ``linux`` / ``darwin`` / ``windows``）。
    未配置则返回 None：保留配置中 ``model`` 写的原始路径不做改写。
    """
    env_dir = os.environ.get("INSECT_MODEL_DIR")
    if env_dir:
        return env_dir.strip()
    model_dir_map = data.get("model_dir")
    if isinstance(model_dir_map, dict):
        d = model_dir_map.get(_current_os_key())
        if isinstance(d, str) and d.strip():
            return d.strip()
    return None


def _rewrite_model_paths(node: Any, model_dir: str) -> None:
    """递归改写配置树中的 model/trt 等路径为 ``<model_dir>/<原文件名>``。"""
    rewrite_model_dir_paths(node, model_dir)


def load_insect_alg_all(path: str | Path | None = None) -> dict[str, Any]:
    launcher_path = resolve_insect_alg_all_path(path)
    effective_path = resolve_effective_insect_alg_path(launcher_path)
    if not effective_path.is_file():
        raise FileNotFoundError(f"未找到统一算法配置: {effective_path}")
    if effective_path.resolve() != launcher_path.resolve():
        run_model = read_run_model_profile(launcher_path)
        logging.info(
            "run_model=%s: %s -> %s",
            run_model,
            launcher_path.name,
            effective_path.name,
        )
    if is_insect_alg_profile_path(effective_path):
        launcher = load_insect_alg_launcher_dict()
        if not launcher:
            with open(effective_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            with open(effective_path, "r", encoding="utf-8") as f:
                profile = json.load(f)
            if not isinstance(profile, dict):
                raise ValueError(f"场景配置须为 JSON 对象: {effective_path}")
            data = compose_insect_alg_from_profile(launcher, profile)
            logging.info(
                "以 %s 为基线，已从 %s 覆盖场景差异配置",
                INSECT_ALG_LAUNCHER_JSON.name,
                effective_path.name,
            )
    else:
        with open(effective_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    if not isinstance(data, dict) or "models" not in data:
        raise ValueError(f"配置须包含顶层 models 字段: {effective_path}")
    data["run_model"] = read_run_model_profile(INSECT_ALG_LAUNCHER_JSON)
    model_dir = _resolve_model_dir(data)
    if model_dir:
        _rewrite_model_paths(data.get("models"), model_dir)
        logging.info("按操作系统(%s)切换模型根目录: %s", _current_os_key(), model_dir)
    set_inference_global_cfg(data)
    if not resolve_trt_switch(data):
        logging.info("trt_switch=false：全管线使用 YOLO .pt，不加载 TensorRT .engine")
    return data


_POSTPROCESS_ROOT_SKIP_KEYS = frozenset(
    {
        "debug",
        "report_all_switch",
        "shenchan_map_classes",
        "top1_classes",
        "top2_classes",
        "top3_classes",
        "other_classes",
        "background_classes",
        "class_tier_aliases",
    }
)


def _merge_root_cfg(
    raw: dict[str, Any],
    defaults: dict[str, Any],
    alg_config: dict[str, Any],
) -> dict[str, Any]:
    """
    合并根模型配置：defaults → 顶层 prepare/postprocess/predict_cfg → 根模型 raw（后者优先）。

    ``predict_cfg``：推理开关（``clip_profiles_enable``、``detect_seg_batch_size`` 等）。
    ``prepare``：推理前预处理（ROI 等）；``postprocess``：推理后过滤（如 dia_switch、bin_dark_ratio_min、inner_boxes_fp_threshold）。
    ``postprocess`` 中的 ``top1/top2/top3/other/background_classes``、``class_tier_aliases``（类型等价、可传递）、``shenchan_map_classes`` 为全局报出/验证配置，不并入单根推理配置。
    根模型内显式写的同名键覆盖公共段。
    """
    pipeline: dict[str, Any] = {}
    for section in ("prepare", "postprocess"):
        block = alg_config.get(section)
        if isinstance(block, dict):
            for k, v in block.items():
                if str(k).endswith("_说明"):
                    continue
                if section == "postprocess" and k in _POSTPROCESS_ROOT_SKIP_KEYS:
                    continue
                pipeline[k] = v
    predict_block = alg_config.get("predict_cfg")
    if isinstance(predict_block, dict):
        for k, v in predict_block.items():
            if str(k).endswith("_说明"):
                continue
            if k == "run_count":
                continue
            pipeline[k] = v
    out = dict(defaults)
    out.update(pipeline)
    for k, v in raw.items():
        if v is not None:
            out[k] = v
    return out


def merge_root_model_cfg(
    raw: dict[str, Any],
    alg_config: dict[str, Any],
    *,
    model_type: str = "detect",
) -> dict[str, Any]:
    """解析单根模型有效配置（含 prepare/postprocess），供脚本或测试复用。"""
    mtype = str(model_type or "detect").strip().lower()
    defaults = _SEGMENT_DEFAULTS if mtype == "segment" else _DETECT_DEFAULTS
    return _merge_root_cfg(raw, defaults, alg_config)


def resolve_root_cfg_from_alg(
    alg_config: dict[str, Any],
    root_id: str,
) -> dict[str, Any]:
    """从 ``insect_alg_all.json`` 整表解析指定根模型的有效配置。"""
    raw = (alg_config.get("models") or {}).get(root_id) or {}
    if not isinstance(raw, dict):
        return {}
    mtype = str(raw.get("model_type", "detect")).strip().lower()
    return merge_root_model_cfg(raw, alg_config, model_type=mtype)


def _coalesce(cfg: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    out = dict(defaults)
    for k, v in cfg.items():
        if v is not None:
            out[k] = v
    return out


def _is_enabled(cfg: dict[str, Any]) -> bool:
    return bool(cfg.get("enable", True))


def _out_entry_is_enabled(entry: Any) -> bool:
    """
    ``out`` 子项兼容 enable 开关：

    - null / 非 dict：保持原语义（视为 enabled，由上层逻辑决定是否跳过）
    - dict：当 ``enable=false`` 时视为关闭
    """
    if entry is None or not isinstance(entry, dict):
        return True
    return _is_enabled(entry)


def _route_table_wants_direct_output(route_table: dict[str, Any] | None) -> bool:
    """
    分类节点 ``out: {}``（空对象）或未配置子 ``out`` 且传入空表：
    不再按类名子路由，直接输出当前 row（一般为分类 top1 结果）。
    """
    return route_table is not None and not route_table


def _resolve_cls_cfg_from_route_entry(entry: dict[str, Any]) -> dict[str, Any] | None:
    """``models.cls`` 与路由项下直接写 ``cls``（简写）均支持。"""
    models = entry.get("models") or {}
    if isinstance(models, dict):
        raw = models.get("cls")
        if isinstance(raw, dict):
            return raw
    raw = entry.get("cls")
    if isinstance(raw, dict):
        return raw
    return None


def _route_entry_cls_disabled(route_entry: dict[str, Any] | None) -> bool:
    """
    路由项已配置 ``models.cls`` 且 ``enable=false``：
    上层 ``out`` 仍开启时跳过嵌套分类，直接按 ``infer_name`` 报出。
    """
    if not route_entry or not isinstance(route_entry, dict):
        return False
    cls_raw = _resolve_cls_cfg_from_route_entry(route_entry)
    if cls_raw is None:
        return False
    cls_merged = _coalesce(cls_raw, _CLS_NODE_DEFAULTS)
    return not _is_enabled(cls_merged)


def _iter_cls_route_nodes(
    cfg: dict[str, Any],
    base_path: str,
) -> list[tuple[str, dict[str, Any]]]:
    """递归收集 out/models.cls 嵌套分类节点（路径, 合并后配置）。"""
    results: list[tuple[str, dict[str, Any]]] = []
    out_table = cfg.get("out")
    if not isinstance(out_table, dict):
        return results
    for key, entry in out_table.items():
        if not isinstance(entry, dict):
            continue
        entry_path = f"{base_path}.out[{key}]"
        cls_raw = _resolve_cls_cfg_from_route_entry(entry)
        if cls_raw is not None:
            cls_merged = _coalesce(cls_raw, _CLS_NODE_DEFAULTS)
            cls_path = f"{entry_path}.models.cls"
            results.append((cls_path, cls_merged))
            results.extend(_iter_cls_route_nodes(cls_merged, cls_path))
    return results


def _describe_cls_route_node(path: str, cls_cfg: dict[str, Any]) -> str:
    """单行描述嵌套分类节点配置状态。"""
    if not _is_enabled(cls_cfg):
        return f"  - {path}: 已关闭(enable=false)"
    model = cls_cfg.get("model")
    if not model:
        return f"  - {path}: 缺少 model（将无法分类）"
    backend = cls_cfg.get("cls_backend", "auto")
    infer = resolve_inference_model_path(cls_cfg, log_label=path)
    trt = cls_cfg.get("trt")
    trt_note = f" trt={trt}" if trt else ""
    return f"  - {path}: model={model} infer={infer}{trt_note} backend={backend}"


def resolve_gray_contrast_options(cfg: dict[str, Any]) -> dict[str, Any]:
    """检测/分割根模型：灰度 CLAHE 对比度增强参数。"""
    debug_save = bool(cfg.get("gray_contrast_debug_save", False))
    # 兼容旧配置：曾用 gray_contrast_debug_dir 非空表示开启
    if not debug_save and str(cfg.get("gray_contrast_debug_dir") or "").strip():
        debug_save = True
    return {
        "gray_contrast_enhance": bool(cfg.get("gray_contrast_enhance", False)),
        "gray_clahe_clip": float(cfg.get("gray_clahe_clip", 2.0)),
        "gray_clahe_tile": int(cfg.get("gray_clahe_tile", 8)),
        "gray_contrast_debug_save": debug_save,
    }


def resolve_model_square_options(
    cfg: dict[str, Any],
    *,
    model_type: str,
) -> dict[str, bool]:
    """
    根模型 ``to_square`` 与推理补方参数映射。

    - **segment**：``to_square`` → ``pad_full_image_to_square``（整图/不切片推理前白底补方）；
      未写 ``to_square`` 时默认与 ``pad_full_image_to_square`` 一致（默认 True）。
    - **detect**：``to_square`` → ``detect_pad_square``（切片补方）及
      ``detect_pad_square_full_image``（整图检测补方）；未写时切片默认 True、整图默认 False。
    - 两类均可另写显式键覆盖；``cls_pad_square`` 默认跟随 ``to_square``（嵌套分类裁剪）。
    """
    explicit = cfg.get("to_square")
    to_sq: bool | None = None if explicit is None else bool(explicit)
    mtype = str(model_type or "").strip().lower()

    if mtype == "segment":
        pad_full = cfg.get("pad_full_image_to_square")
        if pad_full is None:
            pad_full = True if to_sq is None else to_sq
        cls_sq = cfg.get("cls_pad_square")
        if cls_sq is None:
            cls_sq = True if to_sq is None else to_sq
        return {
            "pad_full_image_to_square": bool(pad_full),
            "cls_pad_square": bool(cls_sq),
        }

    d_pad = cfg.get("detect_pad_square")
    if d_pad is None:
        d_pad = True if to_sq is None else to_sq
    d_full = cfg.get("detect_pad_square_full_image")
    if d_full is None:
        d_full = False if to_sq is None else to_sq
    cls_sq = cfg.get("cls_pad_square")
    if cls_sq is None:
        cls_sq = True if to_sq is None else to_sq
    return {
        "detect_pad_square": bool(d_pad),
        "detect_pad_square_full_image": bool(d_full),
        "cls_pad_square": bool(cls_sq),
    }


def resolve_seg_imgsz(cfg: dict[str, Any]) -> int:
    """
    分割 YOLO 推理边长（Ultralytics ``imgsz``）。

    配置键优先顺序：``seg_imgsz`` → ``infer_imgsz`` → ``imgsz``；
    ``<=0`` 或未配置表示使用 YOLO 默认（不向 predict 传 imgsz）。
    """
    for key in ("seg_imgsz", "infer_imgsz", "imgsz"):
        raw = cfg.get(key)
        if raw is None:
            continue
        try:
            n = int(raw)
        except (TypeError, ValueError):
            logging.warning("无效 %s=%r，已忽略", key, raw)
            continue
        if n > 0:
            return n
    return 0


def _primary_class_name(row: dict[str, Any]) -> str:
    for k in ("cls_name", "class_name", "seg_cls_name"):
        v = str(row.get(k, "") or "").strip()
        if v:
            return v
    return ""


def _is_other_class_name(name: str | None) -> bool:
    """``other`` 开头的类别（含 other_feishi / other_bai 等变体）。"""
    return str(name or "").strip().lower().startswith("other")


def _stash_top1_cls_before_other(row: dict[str, Any]) -> None:
    """对外输出将置为 other 前，保留分类 top1 类名与置信度（不写回 cls_name）。"""
    topk = row.get("cls_topk") or row.get("cls_top3") or []
    topk_nm = ""
    topk_conf: float | None = None
    for item in topk:
        if not isinstance(item, dict):
            continue
        nm = str(item.get("class_name", "") or "").strip()
        if not nm or _is_other_class_name(nm):
            continue
        topk_nm = nm
        try:
            tc = item.get("conf")
            if tc is not None:
                topk_conf = float(tc)
        except (TypeError, ValueError):
            topk_conf = None
        break

    cur = str(row.get("cls_name") or "").strip()
    if cur and not _is_other_class_name(cur):
        row["cls_name_top1"] = cur
        try:
            conf = float(row.get("cls_conf", 0.0) or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        if conf <= 0.0 and topk_conf is not None and topk_conf > 0.0:
            conf = topk_conf
        if conf > 0.0:
            row["cls_conf_top1"] = conf
    elif topk_nm:
        row["cls_name_top1"] = topk_nm
        if topk_conf is not None and topk_conf > 0.0:
            row["cls_conf_top1"] = topk_conf


def _parse_other_cls_conf_reason(
    reason: str | None,
) -> tuple[str, float, float] | None:
    """从 ``other_cls_conf:类名(conf<=thr)`` 解析 demote 时记录的 top1 与门限。"""
    r = str(reason or "").strip()
    m = re.match(r"^other_cls_conf:([^(]+)\(([\d.]+)<=([\d.]+)\)\s*$", r)
    if not m:
        return None
    try:
        return m.group(1).strip(), float(m.group(2)), float(m.group(3))
    except (TypeError, ValueError):
        return None


def _extract_top1_cls_conf(row: dict[str, Any]) -> float | None:
    """解析分类 top1 置信度（后置 other 时优先 demote 记录）。"""
    reason = str(row.get("cls_other_reason") or row.get("filter_reason") or "")
    parsed = _parse_other_cls_conf_reason(reason)
    if parsed is not None:
        return parsed[1]

    v = row.get("cls_conf_top1")
    if v is not None:
        try:
            return float(v)
        except (TypeError, ValueError):
            pass
    topk = row.get("cls_topk") or row.get("cls_top3") or []
    for item in topk:
        if not isinstance(item, dict):
            continue
        nm = str(item.get("class_name", "") or "").strip()
        if not nm or _is_other_class_name(nm):
            continue
        try:
            c = item.get("conf")
            if c is not None:
                return float(c)
        except (TypeError, ValueError):
            continue
    return None


def _effective_cls_conf_for_display(row: dict[str, Any]) -> float:
    """
    绘图/落盘用分类置信度。

    对外 ``other`` 时展示 demote 前模型 top1 置信度；优先 ``cls_other_reason`` 内记录。
    """
    name = str(row.get("name") or row.get("cls_name") or "").strip()
    is_other_out = _is_other_class_name(name) or str(
        row.get("cls_name") or ""
    ).strip().lower() == "other"
    if is_other_out:
        top1_conf = _extract_top1_cls_conf(row)
        if top1_conf is not None:
            return float(top1_conf)
        # other 场景禁止回退到 score/seg_conf，避免误显示 1.00
        try:
            v = row.get("cls_conf")
            if v is not None:
                return float(v)
        except (TypeError, ValueError):
            pass
        return 0.0
    try:
        v = row.get("cls_conf")
        if v is not None:
            return float(v)
    except (TypeError, ValueError):
        pass
    det = _row_detect_conf(row)
    try:
        return float(row.get("score", det) or det)
    except (TypeError, ValueError):
        return float(det)


def _is_other_filter_reason(reason: str | None) -> bool:
    """``filter_reason`` 是否为「后置 other」族（含历史笼统 ``other``）。"""
    r = str(reason or "").strip().lower()
    return r == "other" or r.startswith("other_cls")


def _format_cls_conf_other_reason(top1: str, conf: float, thr: float) -> str:
    """分类模型有 top1 但置信度未过门限，后置为 other（模型本身无 other 类）。"""
    nm = str(top1 or "").strip() or "?"
    return f"other_cls_conf:{nm}({conf:.2f}<={thr:.2f})"


def _mark_row_cls_other(
    row: dict[str, Any],
    reason: str,
    *,
    set_filter: bool = False,
) -> dict[str, Any]:
    out = dict(row)
    out["cls_other_reason"] = str(reason)
    out["cls_name"] = "other"
    if set_filter:
        out["filter"] = True
    return out


def _resolve_other_filter_reason(row: dict[str, Any]) -> str:
    """
    解析「后置 other」的 ``filter_reason``。

    分类模型（如 cls-3.6.1）无 other 类；对外 other 均为后处理产生。
    优先 ``cls_other_reason``（``_ClsRunner`` 写入）；否则按行内字段推断。
    """
    explicit = str(row.get("cls_other_reason") or "").strip()
    if explicit:
        return explicit

    top1 = _extract_top1_cls_name(row)
    conf = _extract_top1_cls_conf(row)
    if conf is None:
        try:
            conf = float(row.get("cls_conf", 0.0) or 0.0)
        except (TypeError, ValueError):
            conf = 0.0

    thr_raw = row.get("cls_conf_threshold")
    if top1 and conf > 0.0:
        if thr_raw is not None:
            try:
                return _format_cls_conf_other_reason(top1, conf, float(thr_raw))
            except (TypeError, ValueError):
                pass
        return f"other_cls_conf:{top1}({conf:.2f})"

    if bool(row.get("filter")) and conf <= 0.0:
        return "other_cls_crop"
    if conf <= 0.0:
        return "other_cls_none"
    return "other"


def _resolve_cls_filter_reason(row: dict[str, Any] | None) -> str:
    """分类节点返回 ``filter=True`` 或 ``cls_name=other`` 时的过滤原因。"""
    if row is None:
        return "cls"
    if str(row.get("cls_name") or "").strip().lower() == "other":
        return _resolve_other_filter_reason(row)
    explicit = str(row.get("cls_other_reason") or "").strip()
    if explicit:
        return explicit
    return "cls"


def _extract_top1_cls_name(row: dict[str, Any]) -> str:
    """从行内字段解析分类 top1 类名（用于对外 other 时的绘图标签）。"""
    reason = str(row.get("cls_other_reason") or row.get("filter_reason") or "")
    parsed = _parse_other_cls_conf_reason(reason)
    if parsed is not None:
        nm = parsed[0]
        if nm and not _is_other_class_name(nm):
            return nm
    for key in ("cls_name_top1", "viz_cls_name"):
        v = str(row.get(key) or "").strip()
        if v and not _is_other_class_name(v):
            return v
    topk = row.get("cls_topk") or row.get("cls_top3") or []
    for item in topk:
        if not isinstance(item, dict):
            continue
        v = str(item.get("class_name", "") or "").strip()
        if v and not _is_other_class_name(v):
            return v
    seg = str(row.get("seg_cls_name") or row.get("class_name") or "").strip()
    return seg


def _resolve_viz_cls_name(
    row: dict[str, Any],
    output_name: str,
    *,
    cn_index: dict[str, str] | None = None,
    cls_cfg: dict[str, Any] | None = None,
) -> str | None:
    """
    对外 ``name`` 为 other 族时，返回绘图用的原始预测类名（尽量中文）；无则 None。

    对外 JSON/XML 仍用 ``name``（可为 other）；``viz_name`` 仅用于 ``draw_results`` 标签。
    """
    out = str(output_name or "").strip()
    if not _is_other_class_name(out):
        return None
    top1 = _extract_top1_cls_name(row)
    if not top1:
        return None
    return _resolve_cn_display_for_class(top1, cn_index=cn_index, cls_cfg=cls_cfg)


def _row_detect_conf(row: dict[str, Any]) -> float:
    """检测/分割框置信度（不含分类 top1，避免与 cls_conf 混用）。"""
    for k in ("conf", "seg_conf"):
        try:
            v = row.get(k)
            if v is not None:
                return float(v)
        except (TypeError, ValueError):
            continue
    return 0.0


def _row_conf(row: dict[str, Any]) -> float:
    """输出展示用综合置信度（优先分类）。"""
    for k in ("cls_conf", "conf", "seg_conf"):
        try:
            v = row.get(k)
            if v is not None:
                return float(v)
        except (TypeError, ValueError):
            continue
    return 0.0


def resolve_root_detect_conf(cfg: dict[str, Any], *, default: float = 0.25) -> float:
    raw = cfg.get("detect_conf", cfg.get("conf_thresh"))
    if raw is None:
        return float(default)
    return float(raw)


def iter_out_route_entries(out: dict[str, Any] | None):
    """遍历 ``out`` 下带门限/子模型的路由项（跳过 null 与空 dict，仅用于 infer 下限等）。"""
    if not out:
        return
    for cls_key, entry in out.items():
        if entry is None or not isinstance(entry, dict):
            continue
        if not _out_entry_is_enabled(entry):
            continue
        if not entry:
            continue
        yield str(cls_key), entry


def resolve_infer_detect_conf_floor(cfg: dict[str, Any], *, default: float = 0.25) -> float:
    """
    模型推理阶段置信度下限：根 ``detect_conf`` 与各 ``out.*.detect_conf`` 的**最小值**。

    使 ``ming: 0.1`` 等低于根阈值（如 0.3）的类能在 YOLO 阶段被保留，再在路由时按各类别门限过滤。
    """
    floors = [resolve_root_detect_conf(cfg, default=default)]
    for _cls, entry in iter_out_route_entries(cfg.get("out")):
        if entry.get("detect_conf") is not None:
            floors.append(float(entry["detect_conf"]))
    return min(floors)


def resolve_entry_detect_conf(
    entry: dict[str, Any],
    root_cfg: dict[str, Any],
    *,
    default: float = 0.25,
) -> float:
    """单类路由项上的检测门限；未配置时用根 ``detect_conf``。"""
    if entry.get("detect_conf") is not None:
        return float(entry["detect_conf"])
    return resolve_root_detect_conf(root_cfg, default=default)


def _row_cls_conf(row: dict[str, Any]) -> float:
    """分类 top1 置信度（仅 ``cls_conf`` 字段）。"""
    try:
        v = row.get("cls_conf")
        if v is None:
            return 0.0
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def resolve_skip_dia_conf(cfg: dict[str, Any] | None) -> float | None:
    """``models.cls.skip_dia_conf``：分类置信度 **严格大于** 该值时跳过路由项 ``dia`` 过滤。"""
    if not cfg:
        return None
    v = cfg.get("skip_dia_conf")
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f > 0 else None


def _should_skip_dia_by_cls_conf(
    row: dict[str, Any],
    skip_dia_conf: float | None,
) -> bool:
    if skip_dia_conf is None:
        return False
    return _row_cls_conf(row) > float(skip_dia_conf)


def _passes_entry_thresholds(
    row: dict[str, Any],
    entry: dict[str, Any],
    root_cfg: dict[str, Any],
    *,
    dia_enabled: bool = True,
    skip_dia_conf: float | None = None,
) -> bool:
    """按路由项上的 detect_conf（或继承根阈值）/ dia 过滤当前实例。"""
    conf_thr = resolve_entry_detect_conf(entry, root_cfg)
    if _row_detect_conf(row) < conf_thr:
        return False
    apply_dia = dia_enabled and not _should_skip_dia_by_cls_conf(row, skip_dia_conf)
    if apply_dia:
        dia = entry.get("dia")
        if isinstance(dia, (list, tuple)) and len(dia) >= 2:
            lo, hi = float(dia[0]), float(dia[1])
            if lo > hi:
                lo, hi = hi, lo
            d = bbox_diag_px_from_row(row)
            if d < lo or d > hi:
                return False
    return True


def _threshold_filter_reason(
    row: dict[str, Any],
    entry: dict[str, Any],
    root_cfg: dict[str, Any],
    *,
    dia_enabled: bool = True,
    skip_dia_conf: float | None = None,
) -> str:
    """
    细分 ``threshold`` 过滤原因：

    - ``threshold_conf``：没过 detect_conf（检测/分割 conf；不是展示用 cls_conf）
    - ``threshold_dia``：bbox 对角线不在 dia 范围
    """
    conf_thr = resolve_entry_detect_conf(entry, root_cfg)
    if _row_detect_conf(row) < conf_thr:
        return "threshold_conf"
    apply_dia = dia_enabled and not _should_skip_dia_by_cls_conf(row, skip_dia_conf)
    if apply_dia:
        dia = entry.get("dia")
        if isinstance(dia, (list, tuple)) and len(dia) >= 2:
            lo, hi = float(dia[0]), float(dia[1])
            if lo > hi:
                lo, hi = hi, lo
            d = bbox_diag_px_from_row(row)
            if d < lo or d > hi:
                return "threshold_dia"
    return "threshold"


def resolve_in_big_conf(cfg: dict[str, Any] | None) -> float | None:
    """
    小虫检测框前景与大虫 mask **相交 / 大虫 mask** 比例的门限（``in_big_conf``）。

    相交为 0 时不计算、不过滤；比例严格大于门限时剔除误报。
    配置在 detect 根模型上；``(0, 1]`` 有效，未配置或非法时返回 None（关闭过滤）。
    """
    if not cfg:
        return None
    v = cfg.get("in_big_conf")
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if 0.0 < f <= 1.0 else None


def resolve_cls_top_n(cfg: dict[str, Any] | None) -> int:
    """
    ``cls_top_n``：分类节点保留的 top-N 候选数（按置信度降序）。

    未配置或 ``<=1`` 时视为 1，行为与改前一致（仅 top1）。
    """
    if not isinstance(cfg, dict):
        return 1
    v = cfg.get("cls_top_n")
    if v is None:
        return 1
    try:
        n = int(v)
    except (TypeError, ValueError):
        return 1
    return max(1, n)


def _cls_topk_item_conf(item: Any) -> float:
    if not isinstance(item, dict):
        return 0.0
    try:
        return float(item.get("conf", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _cls_topk_item_name(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    return str(item.get("class_name") or item.get("name") or "").strip()


def _sorted_cls_topk_entries(
    row: dict[str, Any],
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """从结果行读取分类 top-k 并按置信度降序排列。"""
    raw = row.get("cls_topk") or row.get("cls_top3") or []
    items = [
        dict(x)
        for x in raw
        if isinstance(x, dict) and _cls_topk_item_name(x)
    ]
    items.sort(key=_cls_topk_item_conf, reverse=True)
    if limit is not None and limit > 0:
        items = items[: int(limit)]
    return items


def _apply_cls_top_n_to_row(
    row: dict[str, Any],
    cls_cfg: dict[str, Any] | None,
) -> None:
    """``cls_top_n>1`` 时就地裁剪并排序 ``cls_topk``，供 XML 导出与校验标签使用。"""
    n = resolve_cls_top_n(cls_cfg)
    if n <= 1:
        return
    entries = _sorted_cls_topk_entries(row, limit=n)
    if not entries:
        return
    row["cls_top_n"] = n
    row["cls_topk"] = entries
    row["cls_top3"] = entries[: min(3, len(entries))]


def _build_cls_topn_xml_payload(
    r: dict[str, Any],
    *,
    use_cn_name: bool = False,
    cn_index: dict[str, str] | None = None,
) -> list[dict[str, Any]] | None:
    """构建 Pascal VOC ``cls_topn`` 块（``cls_top_n>1`` 且至少 2 个候选时）。"""
    n = int(r.get("cls_top_n", 1) or 1)
    if n <= 1:
        return None
    entries = _sorted_cls_topk_entries(r, limit=n)
    if len(entries) < 2:
        return None
    payload: list[dict[str, Any]] = []
    for rank, item in enumerate(entries, start=1):
        name_key = _cls_topk_item_name(item)
        if use_cn_name:
            disp = _resolve_cn_display_for_class(name_key, cn_index=cn_index)
            name = disp or name_key
        else:
            name = name_key
        payload.append(
            {
                "rank": rank,
                "name": name,
                "conf": _cls_topk_item_conf(item),
            }
        )
    return payload or None


def resolve_in_big_skip(cfg: dict[str, Any] | None) -> frozenset[str]:
    """
    ``in_big_skip``：``in_big_conf`` 判定时忽略的大虫分割类名清单（小写归一化）。

    配置在 detect 根模型；未配置或空列表时返回空集合。
    """
    if not cfg:
        return frozenset()
    raw = cfg.get("in_big_skip")
    if raw is None:
        return frozenset()
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple, set, frozenset)):
        return frozenset()
    out: set[str] = set()
    for item in raw:
        name = str(item).strip().lower()
        if name:
            out.add(name)
    return frozenset(out)


def resolve_in_big_fill(cfg: dict[str, Any] | None) -> bool:
    """
    ``in_big_fill``：对 bbox 裁切做 GrabCut/边框背景差分提取虫体并填充，再用该 mask 作分子。

    配置在 detect 根模型；默认 ``false``（Otsu 二值化，行为不变）。
    """
    if not cfg:
        return False
    return bool(cfg.get("in_big_fill", False))


def resolve_in_big_skip_det_conf(cfg: dict[str, Any] | None) -> float | None:
    """
    ``in_big_skip_det_conf``：in_big 过滤后仍可能恢复小虫的检测置信度门限。

    配置在 detect 根模型；``[0, 1]`` 有效，未配置或非法时返回 None（关闭二次恢复）。
    当所属大虫类名命中 ``in_big_filter_big`` 且小虫检测置信度 **高于** 本门限时，恢复小虫（大虫仍报出）。
    """
    if not cfg:
        return None
    v = cfg.get("in_big_skip_det_conf")
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if 0.0 <= f <= 1.0 else None


def resolve_in_big_skip_big_conf(cfg: dict[str, Any] | None) -> float | None:
    """
    ``in_big_skip_big_conf``：高于本门限且大虫在 ``in_big_filter_big`` 内时，恢复小虫并反向过滤大虫。

    配置在 detect 根模型；``[0, 1]`` 有效，未配置或非法时返回 None（关闭本档反向过滤）。
    """
    if not cfg:
        return None
    v = cfg.get("in_big_skip_big_conf")
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if 0.0 <= f <= 1.0 else None


def resolve_in_big_filter_big(cfg: dict[str, Any] | None) -> frozenset[str]:
    """
    ``in_big_filter_big``：当内部小虫数量达到 ``in_big_skip_count`` 时剔除的大虫类名清单。

    配置在 detect 根模型；未配置或空列表时返回空集合。
    """
    if not cfg:
        return frozenset()
    raw = cfg.get("in_big_filter_big")
    if raw is None:
        return frozenset()
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple, set, frozenset)):
        return frozenset()
    out: set[str] = set()
    for item in raw:
        name = str(item).strip().lower()
        if name:
            out.add(name)
    return frozenset(out)


def resolve_in_big_skip_count(cfg: dict[str, Any] | None) -> int | None:
    """``in_big_skip_count``：触发剔除 ``in_big_filter_big`` 大虫的内部小虫最少个数。"""
    if not cfg:
        return None
    v = cfg.get("in_big_skip_count")
    if v is None:
        return None
    try:
        n = int(v)
    except (TypeError, ValueError):
        return None
    return n if n >= 1 else None


def _row_class_name_for_in_big_skip(row: dict[str, Any]) -> str:
    for key in ("name", "cls_name", "class_name", "seg_cls_name", "infer_name"):
        value = str(row.get(key) or "").strip().lower()
        if value:
            return value
    return ""


def _filter_big_rows_for_in_big_skip(
    big_rows: list[dict[str, Any]],
    skip_classes: frozenset[str],
) -> list[dict[str, Any]]:
    if not skip_classes:
        return big_rows
    kept: list[dict[str, Any]] = []
    for row in big_rows:
        name = _row_class_name_for_in_big_skip(row)
        if name and name in skip_classes:
            continue
        kept.append(row)
    return kept


def _annotate_in_big_ratio(row: dict[str, Any], iou: float | None) -> dict[str, Any]:
    if iou is None:
        return row
    out = dict(row)
    out["in_big_ratio"] = round(float(iou), 6)
    return out


def _format_in_big_label_suffix(row: dict[str, Any]) -> str:
    ratio = row.get("in_big_ratio")
    if ratio is None:
        return ""
    try:
        return f" i{float(ratio):.2f}"
    except (TypeError, ValueError):
        return ""


def _mark_in_big_filtered(
    row: dict[str, Any],
    *,
    ratio: float,
    threshold: float,
) -> dict[str, Any]:
    out = dict(row)
    out["filtered"] = True
    out["filter_reason"] = "in_big"
    out["in_big_ratio"] = round(float(ratio), 6)
    out["in_big_conf"] = float(threshold)
    return out


def _row_in_big_skip_det_conf(row: dict[str, Any]) -> float:
    """in_big 二次恢复使用的检测置信度（优先 ``det_conf``）。"""
    for k in ("det_conf", "conf", "seg_conf"):
        try:
            v = row.get(k)
            if v is not None:
                return float(v)
        except (TypeError, ValueError):
            continue
    return 0.0


def _restore_in_big_filtered(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out.pop("filtered", None)
    out.pop("filter_reason", None)
    out.pop("in_big_conf", None)
    return out


def _mark_in_big_filter_big(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["filtered"] = True
    out["filter_reason"] = "in_big_filter_big"
    return out


def _in_big_big_row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    src = str(row.get("source") or "")
    name = _row_class_name_for_in_big_skip(row)
    loc = row.get("location")
    if isinstance(loc, (list, tuple)) and len(loc) >= 4:
        loc_key = tuple(int(v) for v in loc[:4])
    else:
        bbox = row_location_to_bbox(row)
        loc_key = tuple(int(v) for v in bbox) if bbox is not None else ()
    poly = row.get("polygon")
    if isinstance(poly, list) and len(poly) >= 3:
        poly_key = tuple(
            (int(p[0]), int(p[1]))
            for p in poly[: min(len(poly), 8)]
            if isinstance(p, (list, tuple)) and len(p) >= 2
        )
    else:
        poly_key = ()
    return (src, name, loc_key, poly_key)


def _is_segment_polygon_row(
    row: dict[str, Any], segment_root_ids: frozenset[str]
) -> bool:
    return (
        str(row.get("source") or "") in segment_root_ids
        and isinstance(row.get("polygon"), list)
        and len(row.get("polygon") or []) >= 3
    )


def _edge_reject_should_filter_row(
    row: dict[str, Any],
    root_cfg: dict[str, Any],
) -> bool:
    """
    与 ``PredictSize.predict`` 第 3.5 步相同的边缘联合过滤规则。

    ``predict_all`` 检测根在 ``out`` 内嵌套分类，须在**最终** ``cls_conf`` 就绪后再判定；
    故在 ``_run_detect_root`` 中对 ``PredictSize`` 传 ``edge_reject_distance=0``，在此函数统一处理。
    """
    if not bool(row.get("edge_rule_enabled", False)):
        return False
    try:
        edge_reject_distance = float(
            row.get("edge_reject_distance", root_cfg.get("edge_reject_distance", 0)) or 0
        )
    except (TypeError, ValueError):
        edge_reject_distance = 0.0
    if edge_reject_distance <= 0:
        return False
    if row.get("filter"):
        return False
    if str(row.get("cls_name") or "") == "other":
        return False
    dm = row.get("edge_min_dist")
    if dm is None:
        return False
    try:
        dm = float(dm)
    except (TypeError, ValueError):
        return False
    if dm >= edge_reject_distance:
        return False

    edge_reject_conf_threshold = row.get("edge_reject_conf_threshold")
    if edge_reject_conf_threshold is None:
        edge_reject_conf_threshold = root_cfg.get("edge_reject_conf_threshold")
    edge_reject_cls_conf_threshold = row.get("edge_reject_cls_conf_threshold")
    if edge_reject_cls_conf_threshold is None:
        edge_reject_cls_conf_threshold = root_cfg.get("edge_reject_cls_conf_threshold")

    cls_conf_thr = (
        0.0
        if edge_reject_cls_conf_threshold is None
        else float(edge_reject_cls_conf_threshold)
    )
    det_conf_thr = (
        resolve_root_detect_conf(root_cfg, default=0.3)
        if edge_reject_conf_threshold is None
        else float(edge_reject_conf_threshold)
    )
    det_conf = float(row.get("conf", 0.0) or 0.0)
    cls_conf = float(row.get("cls_conf", 0.0) or 0.0)
    return det_conf < det_conf_thr or cls_conf < cls_conf_thr


def _detect_filter_reason(row: dict[str, Any]) -> str:
    """
    细分 PredictSize.predict 内部设置 ``filter=True`` 的原因（用于替代笼统的 ``detect``）。

    说明：
    - 这里仅基于 row 上的字段做推断；阈值信息由 _run_detect_root 透传写入 row。
    - 命名尽量稳定，便于统计与检索。
    """
    # 1) 内含框误检：PredictSize._apply_inner_boxes_fp_filter
    if row.get("fp_inner_box_count") is not None:
        try:
            return f"fp_inner_boxes({int(row.get('fp_inner_box_count'))})"
        except Exception:
            return "fp_inner_boxes"

    # 2) 边距去重合并：PredictSize._apply_edge_distance_dup_merge
    if row.get("edge_dup_merged"):
        return "edge_dup_merge"

    # 3) 暗簇占比过滤：PredictSize 第 1.5 步
    dr = row.get("bin_dark_ratio", None)
    dr_min = row.get("bin_dark_ratio_min", None)
    if dr is not None and dr_min is not None:
        try:
            dr_f = float(dr)
            dr_min_f = float(dr_min)
            if dr_min_f > 0 and dr_f < dr_min_f:
                return f"bin_dark({dr_f:.2f}<{dr_min_f:.2f})"
        except (TypeError, ValueError):
            pass

    # 4) merge 后边缘距离联合置信度过滤（依赖 edge_min_dist）
    if bool(row.get("edge_rule_enabled", False)) and row.get("edge_min_dist") is not None:
        try:
            dm = float(row.get("edge_min_dist"))
        except (TypeError, ValueError):
            dm = None
        try:
            ed = float(row.get("edge_reject_distance", 0.0) or 0.0)
        except (TypeError, ValueError):
            ed = 0.0
        if dm is not None and ed > 0 and dm < ed:
            # 作图时不输出条件表达式；仅标记为 edge。
            return "edge"

    # 5) 兜底：尺寸/对角线过滤（PredictSize 第 2 步分支与 else 分支）
    if str(row.get("cls_name") or "") == "other" and float(row.get("cls_conf", 0.0) or 0.0) <= 0.0:
        # 区分：dia/对角线过滤 vs 宽高过滤（size.json）
        if row.get("dia_enabled") is False:
            # dia 已被关闭，则这里的 size 更可能来自宽高范围（或其他未记录的尺寸规则）
            return "size_wh"
        return "size"

    return "detect"


def _dia_enabled_for_width(
    img_w: int,
    dia_w: list[int] | list[float] | tuple[int, ...] | tuple[float, ...] | int | float | None,
    *,
    tolerance_ratio: float = 0.01,
) -> bool:
    """
    dia_w 为 None 时保持兼容（dia 永远生效）。
    dia_w 可配单值或列表：例如 [5472, 800]，表示仅当图片宽度接近 5472 或 800（默认允许±1%）时 dia 才生效。
    """
    if dia_w is None:
        return True
    try:
        w = float(img_w)
    except Exception:
        return True
    if w <= 0:
        return True

    if isinstance(dia_w, (int, float)):
        targets = [float(dia_w)]
    elif isinstance(dia_w, (list, tuple)):
        targets = []
        for v in dia_w:
            try:
                targets.append(float(v))
            except Exception:
                continue
    else:
        return True

    tol = max(0.0, float(tolerance_ratio))
    for t in targets:
        if t <= 0:
            continue
        if abs(w - t) / t <= tol:
            return True
    return False


def _resolve_dia_enabled(
    cfg: dict[str, Any],
    img_w: int,
    *,
    tolerance_ratio: float = 0.01,
    fallback_cfg: dict[str, Any] | None = None,
) -> bool:
    """
    是否对本层配置应用 dia（外接矩形对角线）尺寸过滤。

    - 各模型层级（detect/segment 根、``models.cls`` 等）可独立配置 ``dia_switch``。
    - ``dia_switch`` 为 false 时该层关闭，不使用尺寸过滤。
    - 否则再按 ``dia_w`` 与图片宽度决定是否生效（见 ``_dia_enabled_for_width``）；
      本层未配 ``dia_w`` 时可从 ``fallback_cfg``（通常为根模型）继承。
    """
    if not bool(cfg.get("dia_switch", True)):
        return False
    dia_w = cfg.get("dia_w")
    if dia_w is None and isinstance(fallback_cfg, dict):
        dia_w = fallback_cfg.get("dia_w")
    return _dia_enabled_for_width(
        img_w, dia_w, tolerance_ratio=tolerance_ratio
    )


def _partition_rows_by_bbox_diag(
    rows: list[dict[str, Any]],
    dia_min: float,
    dia_max: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """按外接矩形对角线拆分为保留 / 剔除两组（与 filter_rows_by_bbox_diag_range 规则一致）。"""
    lo, hi = float(dia_min), float(dia_max)
    if lo > hi:
        lo, hi = hi, lo
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for r in rows:
        d = bbox_diag_px_from_row(r)
        if lo <= d <= hi:
            kept.append(r)
        else:
            dropped.append(r)
    return kept, dropped


def _collect_class_cn_display_from_alg_node(node: Any, out: dict[str, str]) -> None:
    """递归收集 class key / infer_name → cn_name（供过滤实例绘图中文展示）。"""
    if isinstance(node, dict):
        out_table = node.get("out")
        if isinstance(out_table, dict):
            for cls_key, entry in out_table.items():
                if not isinstance(entry, dict):
                    continue
                cn = str(entry.get("cn_name") or "").strip()
                if cn:
                    route_key = str(cls_key).strip()
                    infer_key = str(entry.get("infer_name") or "").strip()
                    if route_key and route_key != "*":
                        out[route_key] = cn
                    if infer_key and infer_key != "*" and infer_key != route_key:
                        out.setdefault(infer_key, cn)
                _collect_class_cn_display_from_alg_node(entry, out)
        for key in ("models", "cls"):
            sub = node.get(key)
            if isinstance(sub, dict):
                _collect_class_cn_display_from_alg_node(sub, out)
        for k, v in node.items():
            if k in ("out", "models", "cls"):
                continue
            if isinstance(v, dict):
                _collect_class_cn_display_from_alg_node(v, out)
    elif isinstance(node, list):
        for item in node:
            _collect_class_cn_display_from_alg_node(item, out)


def build_class_cn_display_index(alg_config: dict[str, Any] | None) -> dict[str, str]:
    """从 ``insect_alg_all`` 配置树构建 class key → 中文展示名索引。"""
    index: dict[str, str] = {
        "other": "其他",
        "unknown": "未知",
        "insect": "昆虫",
        "insect_small": "小虫",
    }
    if isinstance(alg_config, dict):
        _collect_class_cn_display_from_alg_node(alg_config.get("models"), index)
    return index


def _resolve_cn_display_for_class(
    class_key: str,
    *,
    cn_index: dict[str, str] | None = None,
    cls_cfg: dict[str, Any] | None = None,
) -> str:
    """按 cls out → alg 配置 → insect_info 顺序解析中文展示名；无映射时回退原类键。"""
    key = str(class_key or "").strip()
    if not key:
        return ""
    if cls_cfg:
        out_table = cls_cfg.get("out") or {}
        _pat, entry = resolve_out_route_entry(out_table, key)
        if isinstance(entry, dict):
            cn = str(entry.get("cn_name") or "").strip()
            if cn:
                return cn
    if cn_index:
        hit = cn_index.get(key)
        if hit:
            return hit
    try:
        from script.ls_classification_ingest import cls_name_to_zh

        zh = cls_name_to_zh(key)
        if zh and zh != key:
            return zh
    except ImportError:
        pass
    return key


def _cn_display_from_cls_out(
    cls_key: str,
    cls_cfg: dict[str, Any],
    *,
    cn_index: dict[str, str] | None = None,
) -> str:
    """从分类节点 ``out`` 表解析绘图用中文名；无 cn_name 时回退全局索引/类键。"""
    return _resolve_cn_display_for_class(
        cls_key, cn_index=cn_index, cls_cfg=cls_cfg
    )


def _row_viz_display_name(
    row: dict[str, Any],
    cls_cfg: dict[str, Any] | None,
    cn_index: dict[str, str] | None = None,
) -> str:
    """分类完成后的绘图类名（优先 top1 中文名）。"""
    top1 = _extract_top1_cls_name(row)
    if top1:
        return _resolve_cn_display_for_class(top1, cn_index=cn_index, cls_cfg=cls_cfg)
    primary = _primary_class_name(row)
    if primary:
        return _resolve_cn_display_for_class(primary, cn_index=cn_index, cls_cfg=cls_cfg)
    return ""


def _apply_viz_name_if_differs_from_output(
    out: dict[str, Any],
    *,
    row: dict[str, Any],
    cls_cfg: dict[str, Any] | None,
    cn_index: dict[str, str] | None = None,
) -> None:
    """路由 infer_name/cn_name 与分类结果不一致时，绘图标签用 ``viz_name`` 展示分类名。"""
    viz = _row_viz_display_name(row, cls_cfg, cn_index)
    if not viz:
        return
    shown = str(out.get("cn_name") or out.get("name") or "").strip()
    if viz != shown:
        out["viz_name"] = viz


def _finalize_result(
    row: dict[str, Any],
    *,
    route_entry: dict[str, Any] | None,
    root_id: str,
    route_key: str | None = None,
    cn_index: dict[str, str] | None = None,
    cls_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cls_name = _primary_class_name(row)
    name = str(route_key or cls_name or "").strip() or cls_name
    infer = ""
    if route_entry:
        infer = str(route_entry.get("infer_name") or "").strip()
    if _route_entry_cls_disabled(route_entry) and infer:
        name = infer
    cn = (route_entry or {}).get("cn_name")
    poly = row.get("polygon")
    if isinstance(poly, list) and len(poly) >= 3:
        x1, y1, x2, y2 = _bbox_from_polygon(poly)
    else:
        x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
    det_conf = _row_detect_conf(row)
    cls_conf = _effective_cls_conf_for_display(row)
    out: dict[str, Any] = {
        "name": name,
        "score": _row_conf(row),
        "det_conf": float(det_conf),
        "cls_conf": float(cls_conf),
        "location": [x1, y1, x2, y2],
        "source": root_id,
        "msg": "",
    }
    if infer and infer != name:
        out["infer_name"] = infer
    if cn:
        out["cn_name"] = str(cn)
    elif cn_index or cls_cfg:
        lookup_key = _extract_top1_cls_name(row) or name
        cn_disp = _resolve_cn_display_for_class(
            lookup_key, cn_index=cn_index, cls_cfg=cls_cfg
        )
        if cn_disp and cn_disp != lookup_key:
            out["cn_name"] = cn_disp
    viz = _resolve_viz_cls_name(
        row, name, cn_index=cn_index, cls_cfg=cls_cfg
    )
    if viz:
        out["viz_name"] = viz
    poly = row.get("polygon")
    if poly:
        out["polygon"] = poly
    if "edge_min_dist" in row:
        out["edge_min_dist"] = row.get("edge_min_dist")
    if "edge_rule_enabled" in row:
        out["edge_rule_enabled"] = bool(row.get("edge_rule_enabled", False))
    if "dia_enabled" in row:
        out["dia_enabled"] = bool(row.get("dia_enabled", True))
    mr = compute_mask_bbox_fill_ratio_from_row(row)
    if mr is not None:
        out["mask_rate"] = round(float(mr), 6)
    for k in (
        "detect_id",
        "clip_profile_idx",
        "clip_tile_seq",
        "clip_profile_total",
        "clip_tile_size",
        "cls_other_reason",
        "cls_name_top1",
        "cls_conf_top1",
        "cls_conf_threshold",
        "cls_topk",
        "cls_top3",
        "cls_top_n",
    ):
        if k in row:
            out[k] = row[k]
    return out


def _finalize_filtered(
    row: dict[str, Any],
    *,
    route_entry: dict[str, Any] | None,
    root_id: str,
    reason: str,
    cls_cfg: dict[str, Any] | None = None,
    cn_index: dict[str, str] | None = None,
) -> dict[str, Any]:
    out = _finalize_result(
        row,
        route_entry=route_entry,
        root_id=root_id,
        cn_index=cn_index,
        cls_cfg=cls_cfg,
    )
    _apply_viz_name_if_differs_from_output(
        out, row=row, cls_cfg=cls_cfg, cn_index=cn_index
    )
    out["filtered"] = True
    out["filter_reason"] = str(reason)
    return out


# --------------------------------------------------------------------------- #
#  递归路由与模型缓存
# --------------------------------------------------------------------------- #


_PREFILL_CROP_EMPTY = object()


class _ClsRunner:
    """嵌套分类节点（仅 cls，无 detect/seg）。"""

    def __init__(
        self,
        cfg: dict[str, Any],
        device: str | None,
        cache: dict[str, ClsModel],
        *,
        route_label: str = "",
        prefill_cache: dict[tuple[int, str], dict[str, Any] | None] | None = None,
        root_cfg: dict[str, Any] | None = None,
        phase_add: Callable[[str, float], None] | None = None,
        cache_lock: threading.Lock | None = None,
    ):
        self.cfg = cfg
        self.device = device
        self._cache = cache
        self.route_label = (route_label or "").strip()
        self._prefill_cache = prefill_cache if prefill_cache is not None else {}
        self._root_cfg = root_cfg or {}
        self._phase_add = phase_add
        self._cache_lock = cache_lock
        self.alg = build_alg_table_from_out(cfg.get("out"))

    def _model(self) -> ClsModel:
        pad_clr = resolve_cls_pad_color(self.cfg.get("cls_pad_color"))
        key = cls_cache_key({**self.cfg, "cls_pad_color": pad_clr})
        if key in self._cache:
            return self._cache[key]
        lock_ctx = self._cache_lock if self._cache_lock is not None else nullcontext()
        with lock_ctx:
            if key in self._cache:
                return self._cache[key]
            pfx = f"[{self.route_label}] " if self.route_label else ""
            infer_path = resolve_inference_model_path(
                self.cfg,
                log_label=self.route_label or "cls",
            )
            logging.info(
                "%s加载嵌套分类模型: %s backend=%s",
                pfx,
                infer_path,
                self.cfg.get("cls_backend", "auto"),
            )
            try:
                self._cache[key] = create_classifier(
                    str(self.cfg["model"]),
                    device=self.device,
                    pad_square=bool(self.cfg.get("to_square", True)),
                    gray_binarize=bool(self.cfg.get("gray_binarize", False)),
                    pad_color_bgr=pad_clr,
                    to_gray=bool(self.cfg.get("to_gray", False)),
                    cfg=self.cfg,
                )
            except Exception:
                self._cache.pop(key, None)
                raise
        return self._cache[key]

    def _make_crop(self, image_bgr: np.ndarray, row: dict[str, Any]) -> np.ndarray | None:
        return make_cls_crop(image_bgr, row, self.cfg)

    def _finalize_cls_result(
        self,
        row: dict[str, Any],
        cls_result: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if cls_result is None:
            det = _mark_row_cls_other(dict(row), "other_cls_none")
            det["cls_conf"] = 0.0
            det["cls_topk"] = []
            det["cls_top3"] = []
            return det

        det = dict(row)
        det["cls_name"] = cls_result["class_name"]
        det["cls_conf"] = cls_result["conf"]
        _tk = cls_result.get("topk") or cls_result.get("top3") or []
        det["cls_topk"] = list(_tk)
        det["cls_top3"] = det["cls_topk"][:3]

        thr = resolve_cls_top1_threshold(
            self.alg,
            str(det.get("cls_name", "") or ""),
            self.cfg.get("cls_conf"),
        )
        if thr is not None and float(det.get("cls_conf", 0.0) or 0.0) <= thr:
            top1_nm = str(det.get("cls_name") or "").strip()
            conf = float(det.get("cls_conf", 0.0) or 0.0)
            _stash_top1_cls_before_other(det)
            det["cls_conf_threshold"] = float(thr)
            reason = _format_cls_conf_other_reason(top1_nm, conf, float(thr))
            if self.route_label:
                logging.info(
                    "[%s] 分类后置other（模型无other类）: top1=%s conf=%.3f <= thr=%.3f",
                    self.route_label,
                    top1_nm,
                    conf,
                    float(thr),
                )
            det = _mark_row_cls_other(det, reason)
            det["cls_conf"] = conf
        _apply_cls_top_n_to_row(det, self.cfg)
        return det

    def classify_row(
        self,
        image_bgr: np.ndarray,
        row: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not _is_enabled(self.cfg):
            return row

        cache_key = (id(row), self.route_label)
        if cache_key in self._prefill_cache:
            cached = self._prefill_cache[cache_key]
            if cached is _PREFILL_CROP_EMPTY:
                det = _mark_row_cls_other(dict(row), "other_cls_crop", set_filter=True)
                det["cls_conf"] = 0.0
                det["cls_topk"] = []
                det["cls_top3"] = []
                return det
            return self._finalize_cls_result(row, cached)

        model = self._model()
        gpu_session = get_gpu_crop_session()
        if gpu_session is not None and cls_job_can_defer_gpu_crop(row, self.cfg):
            padded = gpu_session.batch_crop_pad_from_rows(
                [row],
                cfg=self.cfg,
                pad_square=bool(self.cfg.get("to_square", True)),
                pad_color_bgr=resolve_cls_pad_color(self.cfg.get("cls_pad_color")),
            )
            crop = padded[0] if padded else None
        else:
            crop = self._make_crop(image_bgr, row)
        if crop is None or crop.size == 0:
            det = _mark_row_cls_other(dict(row), "other_cls_crop", set_filter=True)
            det["cls_conf"] = 0.0
            det["cls_topk"] = []
            det["cls_top3"] = []
            return det

        dev = getattr(model, "device", None)
        pad_clr = resolve_cls_pad_color(self.cfg.get("cls_pad_color"))
        _t_cls = time.perf_counter()
        cls_result = model.predict(
            crop,
            device=dev,
            pad_square=cls_infer_pad_square(self.cfg, crop),
            gray_binarize=bool(self.cfg.get("gray_binarize", False)),
            pad_color_bgr=pad_clr,
            to_gray=cls_infer_to_gray(self.cfg),
        )
        if self._phase_add is not None:
            self._phase_add("cls_batch", time.perf_counter() - _t_cls)
        return self._finalize_cls_result(row, cls_result)


class InsectPredictAll:
    """
    从 insect_alg_all.json 构建的多根推理管线。

    - 每个启用的顶层 models.<id> 独立跑图，结果带 source=<id>
    - out / models.cls 形成递归子图（见配置内注释）
    """

    def __init__(
        self,
        config: dict[str, Any] | str | Path | None = None,
        *,
        device: str | None = None,
        root_ids: list[str] | None = None,
        enable_mask_rate_filter: bool = True,
    ):
        if config is None or isinstance(config, (str, Path)):
            self.config_path = resolve_insect_alg_all_path(config)
            self.effective_config_path = resolve_effective_insect_alg_path(
                self.config_path
            )
            self.run_model = read_run_model_profile(self.config_path)
            self.config = load_insect_alg_all(self.config_path)
            self.run_model = str(self.config.get("run_model") or self.run_model).strip().lower()
        else:
            self.config_path = None
            self.effective_config_path = None
            self.config = deepcopy(config)
            self.run_model = str(self.config.get("run_model") or "baipai").strip().lower()

        set_inference_global_cfg(self.config)
        if not resolve_trt_switch(self.config):
            logging.info("trt_switch=false：全管线使用 YOLO .pt，不加载 TensorRT .engine")
        if bool(resolve_predict_cfg_value("parallel_detect_seg", self.config, default=True)):
            logging.info(
                "parallel_detect_seg=true：detect 与 segment 根模型将并发执行"
            )
        self.device = device
        self._enable_mask_rate_filter = bool(enable_mask_rate_filter)
        self._cls_cache: dict[str, ClsModel] = {}
        self._cls_cache_lock = threading.Lock()
        self._trt_cls_prefill: dict[tuple[int, str], dict[str, Any] | object] = {}
        self._detect_predictors: dict[str, PredictSize] = {}
        self._segment_predictors: dict[str, PredictSeg] = {}
        self._roots: dict[str, Any] = {}
        self._root_ids = root_ids
        self._cn_display_index = build_class_cn_display_index(self.config)
        self._segment_root_ids: frozenset[str] = frozenset()
        self._in_big_conf_by_root: dict[str, float] = {}
        self._in_big_skip_by_root: dict[str, frozenset[str]] = {}
        self._in_big_fill_by_root: dict[str, bool] = {}
        self._in_big_skip_det_conf_by_root: dict[str, float] = {}
        self._in_big_skip_big_conf_by_root: dict[str, float] = {}
        self._in_big_filter_big_by_root: dict[str, frozenset[str]] = {}
        self._in_big_skip_count_by_root: dict[str, int] = {}
        self._report_allowed = resolve_report_allowed_class_names(self.config)
        postprocess = _postprocess_block(self.config)
        self._shenchan_map_classes = parse_shenchan_map_classes(postprocess)
        self._report_tier_aliases = postprocess.get("class_tier_aliases") or {}
        self._postprocess_debug = resolve_postprocess_debug(self.config)
        self._phase_recorder = PredictPhaseRecorder(
            enabled=bool(
                resolve_predict_cfg_value(
                    "predict_phase_profile", self.config, default=False
                )
            )
        )
        self._log_report_tier_mode()
        self._build_roots()

    def _active_prefill_cache(self) -> dict[tuple[int, str], dict[str, Any] | object]:
        cache = getattr(_root_run_tls, "prefill", None)
        if cache is not None:
            return cache
        return self._trt_cls_prefill

    def _begin_root_run_tls(self) -> None:
        _root_run_tls.prefill = {}
        if self._phase_recorder.enabled:
            _root_run_tls.phase = PredictPhaseSample()
        else:
            _root_run_tls.phase = None

    def _end_root_run_tls(self) -> PredictPhaseSample | None:
        phase = getattr(_root_run_tls, "phase", None)
        _root_run_tls.prefill = None
        _root_run_tls.phase = None
        return phase

    def _phase_add(self, phase: str, seconds: float) -> None:
        local = getattr(_root_run_tls, "phase", None)
        if local is not None:
            if seconds <= 0:
                return
            attr = f"{phase}_s"
            if hasattr(local, attr):
                setattr(local, attr, getattr(local, attr) + float(seconds))
            return
        self._phase_recorder.add(phase, seconds)

    @contextmanager
    def _phase_timer(self, phase: str):
        if not self._phase_recorder.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._phase_add(phase, time.perf_counter() - t0)

    def _build_roots(self) -> None:
        models_cfg = self.config.get("models") or {}
        want = set(self._root_ids) if self._root_ids else None
        for root_id, raw in models_cfg.items():
            if want is not None and root_id not in want:
                continue
            if not isinstance(raw, dict):
                continue
            mtype = str(raw.get("model_type", "detect")).strip().lower()
            if mtype == "detect":
                defaults = _DETECT_DEFAULTS
            elif mtype == "segment":
                defaults = _SEGMENT_DEFAULTS
            else:
                logging.warning("跳过未知 model_type=%r: %s", mtype, root_id)
                continue
            merged = _merge_root_cfg(raw, defaults, self.config)
            if not _is_enabled(merged):
                logging.info("根模型已关闭(enable=false): %s", root_id)
                continue
            if not merged.get("model"):
                logging.warning("根模型缺少 model 路径: %s", root_id)
                continue
            self._roots[root_id] = merged
            if mtype == "segment":
                self._segment_root_ids = frozenset(
                    set(self._segment_root_ids) | {root_id}
                )
            in_big = resolve_in_big_conf(merged)
            if in_big is not None:
                self._in_big_conf_by_root[root_id] = in_big
                skip_classes = resolve_in_big_skip(merged)
                if skip_classes:
                    self._in_big_skip_by_root[root_id] = skip_classes
                if resolve_in_big_fill(merged):
                    self._in_big_fill_by_root[root_id] = True
                skip_det = resolve_in_big_skip_det_conf(merged)
                if skip_det is not None:
                    self._in_big_skip_det_conf_by_root[root_id] = skip_det
                skip_big = resolve_in_big_skip_big_conf(merged)
                if skip_big is not None:
                    self._in_big_skip_big_conf_by_root[root_id] = skip_big
                filter_big = resolve_in_big_filter_big(merged)
                if filter_big:
                    self._in_big_filter_big_by_root[root_id] = filter_big
                skip_count = resolve_in_big_skip_count(merged)
                if skip_count is not None:
                    self._in_big_skip_count_by_root[root_id] = skip_count
                elif filter_big:
                    self._in_big_skip_count_by_root[root_id] = 3
            nms_iou = merged.get("nms_iou")
            nms_suffix = (
                f", nms_iou={float(nms_iou):.4f}" if nms_iou is not None else ""
            )
            if mtype == "segment":
                seg_sz = resolve_seg_imgsz(merged)
                sq = resolve_model_square_options(merged, model_type="segment")
                if seg_sz > 0:
                    logging.info(
                        "已注册根模型: %s (segment), seg_imgsz=%d, to_square=%s "
                        "(pad_full_image_to_square=%s)%s",
                        root_id,
                        seg_sz,
                        merged.get("to_square"),
                        sq["pad_full_image_to_square"],
                        nms_suffix,
                    )
                else:
                    logging.info(
                        "已注册根模型: %s (segment), to_square=%s "
                        "(pad_full_image_to_square=%s)%s",
                        root_id,
                        merged.get("to_square"),
                        sq["pad_full_image_to_square"],
                        nms_suffix,
                    )
            else:
                logging.info(
                    "已注册根模型: %s (%s)%s", root_id, mtype, nms_suffix
                )
        if self._in_big_conf_by_root:
            skip_parts = [
                f"{rid}={','.join(sorted(skip))}"
                for rid, skip in sorted(self._in_big_skip_by_root.items())
            ]
            skip_msg = (
                f"；in_big_skip: {'; '.join(skip_parts)}"
                if skip_parts
                else ""
            )
            fill_parts = [
                rid for rid in sorted(self._in_big_fill_by_root)
            ]
            fill_msg = (
                f"；in_big_fill: {','.join(fill_parts)}"
                if fill_parts
                else ""
            )
            skip_det_parts = [
                f"{rid}={thr:.4f}"
                for rid, thr in sorted(self._in_big_skip_det_conf_by_root.items())
            ]
            skip_det_msg = (
                f"；in_big_skip_det_conf: {', '.join(skip_det_parts)}"
                if skip_det_parts
                else ""
            )
            skip_big_parts = [
                f"{rid}={thr:.4f}"
                for rid, thr in sorted(self._in_big_skip_big_conf_by_root.items())
            ]
            skip_big_msg = (
                f"；in_big_skip_big_conf: {', '.join(skip_big_parts)}"
                if skip_big_parts
                else ""
            )
            filter_big_parts = [
                f"{rid}={','.join(sorted(names))}"
                for rid, names in sorted(self._in_big_filter_big_by_root.items())
            ]
            filter_big_msg = (
                f"；in_big_filter_big: {'; '.join(filter_big_parts)}"
                if filter_big_parts
                else ""
            )
            logging.info(
                "in_big_conf 小虫二值化黑色占比与大虫 mask 过滤: %s%s%s%s%s%s",
                ", ".join(
                    f"{rid}={thr:.4f}"
                    for rid, thr in sorted(self._in_big_conf_by_root.items())
                ),
                skip_msg,
                fill_msg,
                skip_det_msg,
                skip_big_msg,
                filter_big_msg,
            )
        self._log_cls_pipeline_plan()

    def _log_report_tier_mode(self) -> None:
        if self._report_allowed is None:
            logging.info("报出模式: 全部报出 (report_all_switch=true)")
            return
        logging.info(
            "报出模式: top1+top2+top3+other (report_all_switch=false), 允许 %d 类",
            len(self._report_allowed),
        )
        if self._shenchan_map_classes:
            logging.info(
                "生产类名映射: %d 条 (shenchan_map_classes)",
                len(self._shenchan_map_classes),
            )

    def _apply_report_tier_filter(
        self,
        results: list[dict[str, Any]],
        *,
        filtered_acc: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        allowed = self._report_allowed
        if allowed is None:
            for row in results:
                apply_shengchan_class_map_to_row(
                    row,
                    class_map=self._shenchan_map_classes,
                    cn_index=self._cn_display_index,
                )
            return results
        kept: list[dict[str, Any]] = []
        aliases = self._report_tier_aliases
        for row in results:
            apply_shengchan_class_map_to_row(
                row,
                class_map=self._shenchan_map_classes,
                cn_index=self._cn_display_index,
            )
            name = str(row.get("name") or row.get("cls_name") or "").strip()
            if _class_matches_report_allowed(name, allowed, aliases):
                kept.append(row)
                continue
            if filtered_acc is not None:
                filtered_acc.append(
                    _finalize_filtered(
                        row,
                        route_entry=None,
                        root_id=str(row.get("source") or ""),
                        reason="report_tier",
                        cn_index=self._cn_display_index,
                    )
                )
        if filtered_acc is not None and len(kept) < len(results):
            logging.debug(
                "report_tier 过滤 %d 条（保留 %d 条）",
                len(results) - len(kept),
                len(kept),
            )
        return kept

    def _filter_small_in_big_masks(
        self,
        results: list[dict[str, Any]],
        image_bgr: np.ndarray,
        *,
        filtered_big_rows: list[dict[str, Any]] | None = None,
        source_image_stem: str | None = None,
        result_output_dir: str | Path | None = None,
        eval_metrics_root: str | Path | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """剔除二值化黑色占比超过 ``in_big_conf`` 的小虫检测框。"""
        if not self._in_big_conf_by_root or not self._segment_root_ids:
            return results, []
        if image_bgr is None or getattr(image_bgr, "size", 0) == 0:
            return results, []
        metrics_root = eval_metrics_root or result_output_dir
        debug_save = bool(
            self._postprocess_debug
            and metrics_root
            and source_image_stem
        )
        debug_dir: Path | None = None
        debug_saved = 0
        if debug_save:
            debug_dir = resolve_eval_metrics_debug_dir(metrics_root)
            debug_dir.mkdir(parents=True, exist_ok=True)
        big_rows_all = [
            r
            for r in results
            if _is_segment_polygon_row(r, self._segment_root_ids)
        ]
        if filtered_big_rows:
            seen_keys = {_in_big_big_row_key(r) for r in big_rows_all}
            for row in filtered_big_rows:
                if not _is_segment_polygon_row(row, self._segment_root_ids):
                    continue
                key = _in_big_big_row_key(row)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                big_rows_all.append(row)
        if not big_rows_all and debug_save:
            logging.warning(
                "in_big debug: 本图无带 polygon 的大虫分割实例，仍将输出稻飞虱二值化裁切"
            )
        if not big_rows_all and not debug_save:
            return results, []
        kept: list[dict[str, Any]] = []
        filtered: list[dict[str, Any]] = []
        for row in results:
            src = str(row.get("source", ""))
            thr = self._in_big_conf_by_root.get(src)
            if thr is None:
                kept.append(row)
                continue
            skip_classes = self._in_big_skip_by_root.get(src, frozenset())
            fill_holes = bool(self._in_big_fill_by_root.get(src, False))
            big_rows = _filter_big_rows_for_in_big_skip(big_rows_all, skip_classes)
            if not big_rows and not debug_save:
                kept.append(row)
                continue
            ratio, parts = collect_in_big_debug_parts_for_row(
                row, big_rows, image_bgr, fill_holes=fill_holes
            )
            if debug_save and debug_dir is not None and parts is not None:
                crop_bgr, black_local, big_mask_bool, union_bool, big_name = parts
                panel = build_in_big_debug_panel(
                    crop_bgr,
                    black_local,
                    big_mask_bool,
                    denom_bool=union_bool,
                    ratio=ratio,
                    threshold=thr,
                    big_class_name=big_name,
                    fill_holes_applied=fill_holes,
                )
                if panel is not None:
                    loc = row.get("location") or row_location_to_bbox(row)
                    tag = "unknown"
                    if isinstance(loc, (list, tuple)) and len(loc) >= 4:
                        tag = (
                            f"{int(loc[0])}-{int(loc[1])}-"
                            f"{int(loc[2])}-{int(loc[3])}"
                        )
                    cls_name = str(
                        row.get("name") or row.get("cls_name") or "dfs"
                    ).strip()
                    out_path = save_in_big_debug_panel(
                        panel,
                        debug_dir,
                        image_stem=str(source_image_stem),
                        instance_tag=f"{cls_name}_{src}_{tag}",
                        ratio=ratio,
                    )
                    logging.info("in_big debug 已保存: %s", out_path)
                    debug_saved += 1
            if not big_rows:
                kept.append(_annotate_in_big_ratio(row, ratio))
                continue
            if ratio is not None and ratio > thr:
                filtered.append(
                    _mark_in_big_filtered(row, ratio=ratio, threshold=thr)
                )
            else:
                kept.append(_annotate_in_big_ratio(row, ratio))
        if debug_save and debug_saved:
            logging.info(
                "in_big debug 本图共保存 %d 张 -> %s",
                debug_saved,
                debug_dir,
            )
        if filtered:
            logging.info(
                "in_big_conf 过滤小虫 %d 条（segment 参考实例 %d）",
                len(filtered),
                len(big_rows_all),
            )
        return kept, filtered

    def _resolve_in_big_skip_override(
        self,
        kept: list[dict[str, Any]],
        in_big_filtered: list[dict[str, Any]],
        filtered_acc: list[dict[str, Any]] | None,
        image_bgr: np.ndarray,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]] | None]:
        """
        in_big 过滤后的二次判决：

        1. 小虫检测置信度 > ``in_big_skip_big_conf`` 且所属大虫在 ``in_big_filter_big`` 内
           → 恢复小虫并反向过滤该大虫（单条即可，无需 ``in_big_skip_count``）；
        2. 小虫检测置信度 > ``in_big_skip_det_conf`` 且所属大虫在 ``in_big_filter_big`` 内
           → 恢复小虫（大虫仍报出）；
        3. 小虫检测置信度 > ``in_big_skip_det_conf`` 且所属大虫已不在报出列表 → 仍输出小虫；
        4. 大虫未过滤且在 ``in_big_filter_big`` 中，内部小虫数 ≥ ``in_big_skip_count``
           → 输出满足置信度门限的小虫，并过滤该大虫。
        """
        if not in_big_filtered:
            return kept, in_big_filtered, filtered_acc
        if not (
            self._in_big_skip_det_conf_by_root or self._in_big_skip_big_conf_by_root
        ):
            return kept, in_big_filtered, filtered_acc
        if image_bgr is None or getattr(image_bgr, "size", 0) == 0:
            return kept, in_big_filtered, filtered_acc

        big_rows_all: list[dict[str, Any]] = []
        filtered_big_keys: set[tuple[Any, ...]] = set()
        for row in kept:
            if _is_segment_polygon_row(row, self._segment_root_ids):
                big_rows_all.append(row)
        for row in filtered_acc or []:
            if _is_segment_polygon_row(row, self._segment_root_ids):
                big_rows_all.append(row)
                filtered_big_keys.add(_in_big_big_row_key(row))

        if not big_rows_all:
            return kept, in_big_filtered, filtered_acc

        def _match_small_to_big(
            small_row: dict[str, Any],
        ) -> tuple[dict[str, Any] | None, float | None, str]:
            src = str(small_row.get("source") or "")
            skip_classes = self._in_big_skip_by_root.get(src, frozenset())
            fill_holes = bool(self._in_big_fill_by_root.get(src, False))
            big_rows = _filter_big_rows_for_in_big_skip(big_rows_all, skip_classes)
            big_row, ratio = find_best_in_big_big_row(
                small_row, big_rows, image_bgr, fill_holes=fill_holes
            )
            return big_row, ratio, src

        def _passes_skip_det(small_row: dict[str, Any], src: str) -> bool:
            det_thr = self._in_big_skip_det_conf_by_root.get(src)
            if det_thr is None:
                return False
            return _row_in_big_skip_det_conf(small_row) > det_thr

        def _passes_skip_big_conf(small_row: dict[str, Any], src: str) -> bool:
            big_thr = self._in_big_skip_big_conf_by_root.get(src)
            if big_thr is None:
                return False
            return _row_in_big_skip_det_conf(small_row) > big_thr

        def _big_in_filter_list(big_row: dict[str, Any] | None, src: str) -> bool:
            if big_row is None:
                return False
            filter_big = self._in_big_filter_big_by_root.get(src, frozenset())
            if not filter_big:
                return False
            return _row_class_name_for_in_big_skip(big_row) in filter_big

        def _inside_big(ratio: float | None, src: str) -> bool:
            thr = self._in_big_conf_by_root.get(src)
            return thr is not None and ratio is not None and ratio > thr

        def _counts_toward_in_big_skip(
            small_row: dict[str, Any], ratio: float | None, src: str
        ) -> bool:
            """统计大虫内部小虫：已输出 + 判定 in_big 的并集。"""
            ratio = _resolve_in_big_ratio(small_row, ratio)
            if _inside_big(ratio, src):
                return True
            if str(small_row.get("filter_reason") or "") == "in_big":
                return True
            return ratio is not None and ratio > 0.0

        def _resolve_in_big_ratio(
            small_row: dict[str, Any], ratio: float | None
        ) -> float | None:
            if str(small_row.get("filter_reason") or "") == "in_big":
                try:
                    stored = float(small_row.get("in_big_ratio"))
                except (TypeError, ValueError):
                    stored = None
                if stored is not None and (ratio is None or stored > ratio):
                    ratio = stored
            return ratio

        kept_big_by_key = {
            _in_big_big_row_key(row): row
            for row in kept
            if _is_segment_polygon_row(row, self._segment_root_ids)
        }

        remaining: list[dict[str, Any]] = []
        restored: list[dict[str, Any]] = []
        restored_keys: set[tuple[Any, ...]] = set()
        big_keys_to_filter: set[tuple[Any, ...]] = set()

        def _try_restore_small(small: dict[str, Any]) -> None:
            small_key = (
                str(small.get("source") or ""),
                tuple(small.get("location") or ()),
                str(small.get("name") or ""),
            )
            if small_key in restored_keys:
                return
            restored_keys.add(small_key)
            restored.append(_restore_in_big_filtered(small))
            if small in remaining:
                remaining.remove(small)

        for small in in_big_filtered:
            big_row, ratio, src = _match_small_to_big(small)
            ratio = _resolve_in_big_ratio(small, ratio)
            if big_row is None or not _inside_big(ratio, src):
                remaining.append(small)
                continue
            big_key = _in_big_big_row_key(big_row)
            in_filter_big = _big_in_filter_list(big_row, src)

            if in_filter_big and _passes_skip_big_conf(small, src):
                _try_restore_small(small)
                if big_key in kept_big_by_key:
                    big_keys_to_filter.add(big_key)
                continue

            if in_filter_big and _passes_skip_det(small, src):
                _try_restore_small(small)
                continue

            if not _passes_skip_det(small, src):
                remaining.append(small)
                continue
            # 大虫已不在报出列表（前置过滤或 report_tier）→ 仍输出小虫
            if big_key not in kept_big_by_key or big_key in filtered_big_keys:
                _try_restore_small(small)
            else:
                remaining.append(small)

        big_to_smalls: dict[tuple[Any, ...], list[tuple[dict[str, Any], str]]] = {}
        small_scan: list[dict[str, Any]] = [
            r
            for r in kept
            if str(r.get("source") or "") in self._in_big_conf_by_root
        ]
        small_scan.extend(remaining)
        small_scan.extend(in_big_filtered)
        seen_small_keys: set[tuple[Any, ...]] = set()
        for small in small_scan:
            small_key = (
                str(small.get("source") or ""),
                tuple(small.get("location") or ()),
                str(small.get("name") or ""),
            )
            if small_key in seen_small_keys:
                continue
            seen_small_keys.add(small_key)
            big_row, ratio, src = _match_small_to_big(small)
            ratio = _resolve_in_big_ratio(small, ratio)
            if big_row is None or not _counts_toward_in_big_skip(small, ratio, src):
                continue
            big_key = _in_big_big_row_key(big_row)
            big_to_smalls.setdefault(big_key, []).append((small, src))

        for big_key, small_list in big_to_smalls.items():
            big_row = kept_big_by_key.get(big_key)
            if big_row is None:
                continue
            big_name = _row_class_name_for_in_big_skip(big_row)
            detect_roots = {src for _, src in small_list}
            triggered = False
            for det_root in detect_roots:
                filter_big = self._in_big_filter_big_by_root.get(det_root, frozenset())
                if not filter_big or big_name not in filter_big:
                    continue
                need = self._in_big_skip_count_by_root.get(det_root, 3)
                if len(small_list) >= need:
                    triggered = True
                    break
            if not triggered:
                continue
            if big_key in big_keys_to_filter:
                continue
            big_keys_to_filter.add(big_key)
            for small, src in small_list:
                if str(small.get("filter_reason") or "") != "in_big":
                    continue
                if not _passes_skip_det(small, src):
                    continue
                _try_restore_small(small)

        if big_keys_to_filter:
            new_kept: list[dict[str, Any]] = []
            for row in kept:
                if (
                    _is_segment_polygon_row(row, self._segment_root_ids)
                    and _in_big_big_row_key(row) in big_keys_to_filter
                ):
                    if filtered_acc is not None:
                        filtered_acc.append(_mark_in_big_filter_big(row))
                    continue
                new_kept.append(row)
            kept = new_kept

        if restored:
            kept.extend(restored)
            logging.info(
                "in_big_skip 二次恢复小虫 %d 条（仍过滤 %d 条）",
                len(restored),
                len(remaining),
            )
        if big_keys_to_filter:
            logging.info(
                "in_big_filter_big 过滤大虫 %d 条",
                len(big_keys_to_filter),
            )
        return kept, remaining, filtered_acc

    def _log_cls_pipeline_plan(self) -> None:
        """启动时汇总分类职责：根模型不内嵌 cls，细分类走 out/models.cls。"""
        if not self._roots:
            return
        logging.info(
            "predict_all 分类管线: 根模型 PredictSize/PredictSeg 不内嵌 cls_model_path；"
            "细分类由 out→models.cls 嵌套路由(_ClsRunner)承担"
        )
        for root_id, cfg in self._roots.items():
            mtype = str(cfg.get("model_type", "detect")).strip().lower()
            root_model = cfg.get("model")
            infer_model = resolve_inference_model_path(cfg, log_label=root_id)
            logging.info(
                "  根模型 %s (%s): detect/seg pt=%s infer=%s → 本步不传 cls_model_path",
                root_id,
                mtype,
                root_model,
                infer_model,
            )
            cls_nodes = _iter_cls_route_nodes(cfg, root_id)
            if not cls_nodes:
                logging.warning(
                    "  根模型 %s: out 下未配置 models.cls，推理结果将保留检测/分割粗类",
                    root_id,
                )
                continue
            for path, cls_cfg in cls_nodes:
                logging.info("%s", _describe_cls_route_node(path, cls_cfg))

    def _collect_trt_cls_jobs_for_rows(
        self,
        rows: list[dict[str, Any]],
        image_bgr: np.ndarray,
        route_table: dict[str, Any] | None,
        *,
        route_path: str,
        root_cfg: dict[str, Any],
        dia_enabled: bool,
        skip_dia_conf: float | None = None,
        orig_row: dict[str, Any] | None = None,
    ) -> list[ClsBatchJob]:
        """收集本层路由上需 YOLO 批量分类的 crop（与 ``_walk_route`` 首层 cls 判定一致）。"""
        if not route_table or not rows:
            return []

        jobs: list[ClsBatchJob] = []
        token_row = orig_row if orig_row is not None else None
        for row in rows:
            orig = token_row if token_row is not None else row
            if str(row.get("cls_name", "") or "") == "other":
                continue
            if _route_table_wants_direct_output(route_table):
                continue
            cls_key = _primary_class_name(row)
            _pat, entry = resolve_out_route_entry(
                route_table,
                cls_key,
                dia_px=bbox_diag_px_from_row(row),
                entry_enabled=_out_entry_is_enabled,
            )
            if entry is None or not _out_entry_is_enabled(entry):
                continue
            if isinstance(entry, dict) and not entry:
                continue
            if not isinstance(entry, dict):
                continue

            cls_raw = _resolve_cls_cfg_from_route_entry(entry)
            if cls_raw is None:
                continue
            cls_merged = _coalesce(cls_raw, _CLS_NODE_DEFAULTS)
            if not _is_enabled(cls_merged) or not cls_merged.get("model"):
                continue
            if not cfg_uses_yolo_cls_batch(cls_merged):
                continue

            cls_route_label = f"{route_path}.out[{cls_key}].models.cls"
            cache_key = (id(orig), cls_route_label)
            prefill = self._active_prefill_cache()
            if cache_key in prefill:
                continue

            effective_skip_dia = resolve_skip_dia_conf(cls_merged) or skip_dia_conf
            cls_first = bool(effective_skip_dia is not None)
            if cls_first:
                need_cls = True
            else:
                need_cls = _passes_entry_thresholds(
                    row,
                    entry,
                    root_cfg,
                    dia_enabled=dia_enabled,
                    skip_dia_conf=effective_skip_dia,
                )
            if not need_cls:
                continue

            gpu_session = get_gpu_crop_session()
            if gpu_session is not None and cls_job_can_defer_gpu_crop(row, cls_merged):
                crop = None
            else:
                crop = make_cls_crop(image_bgr, row, cls_merged)
                if crop is None or crop.size == 0:
                    prefill[cache_key] = _PREFILL_CROP_EMPTY
                    continue

            jobs.append(
                ClsBatchJob(
                    row=row,
                    crop=crop,
                    cfg=cls_merged,
                    route_label=cls_route_label,
                    group_key=cls_batch_group_key(cls_merged),
                    batch_size=resolve_cls_batch_size(
                        cls_merged,
                        root_cfg=root_cfg,
                        global_cfg=self.config,
                    ),
                    row_token=id(orig),
                )
            )
        return jobs

    def _prefill_trt_cls_for_rows(
        self,
        rows: list[dict[str, Any]],
        image_bgr: np.ndarray,
        route_table: dict[str, Any] | None,
        *,
        root_id: str,
        root_cfg: dict[str, Any],
        dia_enabled: bool,
        skip_dia_conf: float | None = None,
        max_layers: int = 8,
        img_w: int = 0,
    ) -> None:
        """
        多轮预填 YOLO 嵌套分类：每轮对本层待分类实例批量推理，再沿 out 进入下一层。

        YOLO11 classify（``.pt`` / ``.engine``）走 ``predict_batch``；ConvNeXt 仍在 ``classify_row`` 单条推理。
        """
        self._active_prefill_cache().clear()
        if not route_table or not rows:
            return

        prefill = self._active_prefill_cache()
        with self._phase_timer("cls_batch"):
            pending: list[
                tuple[dict[str, Any], dict[str, Any], str, dict[str, Any], bool]
            ] = [
                (row, route_table, root_id, row, dia_enabled) for row in rows
            ]
            for _layer in range(max(1, int(max_layers))):
                jobs: list[ClsBatchJob] = []
                for sim_row, rt, rpath, orig_row, layer_dia_enabled in pending:
                    jobs.extend(
                        self._collect_trt_cls_jobs_for_rows(
                            [sim_row],
                            image_bgr,
                            rt,
                            route_path=rpath,
                            root_cfg=root_cfg,
                            dia_enabled=layer_dia_enabled,
                            skip_dia_conf=skip_dia_conf,
                            orig_row=orig_row,
                        )
                    )
                if not jobs:
                    break
                prefill.update(
                    run_cls_job_batches(
                        jobs,
                        cls_cache=self._cls_cache,
                        device=self.device,
                        cache_lock=self._cls_cache_lock,
                    )
                )

                next_pending: list[
                    tuple[dict[str, Any], dict[str, Any], str, dict[str, Any], bool]
                ] = []
                for sim_row, rt, rpath, orig_row, _layer_dia in pending:
                    cls_key = _primary_class_name(sim_row)
                    _pat, entry = resolve_out_route_entry(
                        rt,
                        cls_key,
                        dia_px=bbox_diag_px_from_row(sim_row),
                        entry_enabled=_out_entry_is_enabled,
                    )
                    if not isinstance(entry, dict) or not entry:
                        continue
                    cls_raw = _resolve_cls_cfg_from_route_entry(entry)
                    if cls_raw is None:
                        continue
                    cls_merged = _coalesce(cls_raw, _CLS_NODE_DEFAULTS)
                    if not cfg_uses_yolo_cls_batch(cls_merged):
                        continue
                    cls_route_label = f"{rpath}.out[{cls_key}].models.cls"
                    cache_key = (id(orig_row), cls_route_label)
                    if cache_key not in prefill:
                        continue
                    raw = prefill[cache_key]
                    if raw is _PREFILL_CROP_EMPTY or not isinstance(raw, dict):
                        continue
                    row2 = dict(sim_row)
                    row2["cls_name"] = raw["class_name"]
                    row2["cls_conf"] = raw["conf"]
                    _tk = raw.get("topk") or raw.get("top3") or []
                    row2["cls_topk"] = list(_tk)
                    row2["cls_top3"] = row2["cls_topk"][:3]
                    _apply_cls_top_n_to_row(row2, cls_merged)
                    child_out = cls_merged.get("out")
                    if not isinstance(child_out, dict) or not child_out:
                        continue
                    cls_dia_enabled = _resolve_dia_enabled(
                        cls_merged, img_w, fallback_cfg=root_cfg
                    )
                    next_pending.append(
                        (row2, child_out, cls_route_label, orig_row, cls_dia_enabled)
                    )
                if not next_pending:
                    break
                pending = next_pending

    def _make_cls_runner(
        self,
        cls_merged: dict[str, Any],
        *,
        route_label: str,
        root_cfg: dict[str, Any] | None = None,
    ) -> _ClsRunner:
        phase_add = self._phase_add if self._phase_recorder.enabled else None
        return _ClsRunner(
            cls_merged,
            self.device,
            self._cls_cache,
            route_label=route_label,
            prefill_cache=self._active_prefill_cache(),
            root_cfg=root_cfg,
            phase_add=phase_add,
            cache_lock=self._cls_cache_lock,
        )

    def _create_detect_predictor(self, root_id: str, cfg: dict[str, Any]) -> PredictSize:
        out_table = cfg.get("out") or {}
        alg = build_alg_table_from_out(
            out_table, entry_enabled=_out_entry_is_enabled
        )
        cls_list = [str(k) for k in out_table.keys()] if out_table else ["insect"]
        # 注意：dia_w 的开关依赖“当前图片宽度”，而 predictor 是跨图片缓存的；
        # 因此这里不在构造 predictor 时固定 diag_filter_range，而是在 _run_detect_root 中按图片宽度动态传入。
        diag_range = None
        size_path = cfg.get("size_config_path")
        sq = resolve_model_square_options(cfg, model_type="detect")
        gray_enh = resolve_gray_contrast_options(cfg)
        infer_conf = resolve_infer_detect_conf_floor(cfg, default=0.3)
        root_conf = resolve_root_detect_conf(cfg, default=0.3)
        if infer_conf < root_conf:
            logging.info(
                "检测推理 conf_thresh=%.4f（根 detect_conf=%.4f，out 类另有更低门限）",
                infer_conf,
                root_conf,
            )
        log_prefix = f"predict_all/{root_id}/PredictSize"
        detect_path = resolve_inference_model_path(cfg, log_label=f"{root_id}/detect")
        predictor = PredictSize(
            detect_model_path=detect_path,
            size_config_path=str(size_path) if size_path else None,
            cls_list=cls_list,
            cls_model_path=None,
            cls_deferred=True,
            log_prefix=log_prefix,
            offset_rate=float(cfg.get("offset_rate", 1.2)),
            conf_thresh=infer_conf,
            conf_merge=float(cfg.get("conf_merge", 0.1)),
            conf_merge_draw=float(cfg.get("conf_merge_draw", 0.01)),
            iou_threshold=float(cfg.get("iou_threshold", 0.3)),
            ior_threshold=float(cfg.get("ior_threshold", 0.4)),
            device=self.device,
            augment=bool(cfg.get("augment", False)),
            half=bool(cfg.get("half", False)),
            cls_pad_square=sq["cls_pad_square"],
            cls_gray_binarize=bool(cfg.get("gray_binarize", False)),
            cls_to_gray=bool(cfg.get("to_gray", False)),
            inner_boxes_fp_threshold=int(cfg.get("inner_boxes_fp_threshold", 8)),
            bin_dark_ratio_min=float(cfg.get("bin_dark_ratio_min", 0.2)),
            diag_filter_range=diag_range,
            nms_iou=cfg.get("nms_iou"),
            max_det=cfg.get("max_det"),
            nms_agnostic=cfg.get("nms_agnostic"),
            insect_alg=alg or None,
            **gray_enh,
        )
        return predictor

    def _create_segment_predictor(self, root_id: str, cfg: dict[str, Any]) -> PredictSeg:
        out_table = cfg.get("out") or {}
        alg = build_alg_table_from_out(
            out_table, entry_enabled=_out_entry_is_enabled
        )
        cls_keys = [str(k) for k in out_table.keys()] if out_table else None
        seg_imgsz = resolve_seg_imgsz(cfg)
        sq = resolve_model_square_options(cfg, model_type="segment")
        gray_enh = resolve_gray_contrast_options(cfg)
        infer_conf = resolve_infer_detect_conf_floor(cfg, default=0.25)
        root_conf = resolve_root_detect_conf(cfg, default=0.25)
        if infer_conf < root_conf:
            logging.info(
                "分割推理 conf_thresh=%.4f（根 detect_conf=%.4f，out 类另有更低门限）",
                infer_conf,
                root_conf,
            )
        log_prefix = f"predict_all/{root_id}/PredictSeg"
        seg_path = resolve_inference_model_path(cfg, log_label=f"{root_id}/segment")
        predictor = PredictSeg(
            seg_model_path=seg_path,
            cls_model_path=None,
            cls_deferred=True,
            log_prefix=log_prefix,
            cls_list=cls_keys,
            conf_thresh=infer_conf,
            conf_merge=float(cfg.get("conf_merge", 0.1)),
            conf_merge_draw=float(cfg.get("conf_merge_draw", 0.01)),
            ior_threshold=float(cfg.get("ior_threshold", 0.5)),
            device=self.device,
            augment=bool(cfg.get("augment", False)),
            cls_pad_square=sq["cls_pad_square"],
            cls_gray_binarize=bool(cfg.get("gray_binarize", False)),
            cls_to_gray=bool(cfg.get("to_gray", False)),
            seg_imgsz=seg_imgsz,
            nms_iou=cfg.get("nms_iou"),
            max_det=cfg.get("max_det"),
            nms_agnostic=cfg.get("nms_agnostic"),
            retina_masks=bool(cfg.get("retina_masks", False)),
            poly_merge=bool(cfg.get("poly_merge", False)),
            poly_merge_edge_px=float(cfg.get("poly_merge_edge_px", 5.0)),
            poly_merge_edge_ratio=float(cfg.get("poly_merge_edge_ratio", 0.5)),
            poly_merge_contain_ratio=float(cfg.get("poly_merge_contain_ratio", 0.7)),
            poly_merge_cross_class=bool(cfg.get("poly_merge_cross_class", True)),
            poly_merge_max_points=int(cfg.get("poly_merge_max_points", 80)),
            roi_disk_edge_hug_ratio=float(cfg.get("roi_disk_edge_hug_ratio", 0.55)),
            roi_disk_edge_outside_min=float(
                cfg.get("roi_disk_edge_outside_min", 0.02)
            ),
            filter_rows_by_roi_boundary=bool(
                cfg.get("filter_rows_by_roi_boundary", True)
            ),
            crop_pad_ratio=float(cfg.get("crop_pad_ratio", 0.05)),
            cls_crop_from_bbox=bool(cfg.get("from_bbox", False)),
            cls_crop_background=cfg.get("cls_crop_background"),
            cls_pad_color=cfg.get("cls_pad_color"),
            insect_alg=alg or None,
            **gray_enh,
        )
        if seg_imgsz > 0:
            logging.info(
                "分割模型 seg_imgsz=%d（写入 ModelSegmenter.imgsz）: %s",
                seg_imgsz,
                cfg.get("model"),
            )
        if bool(cfg.get("retina_masks", False)):
            logging.info(
                "分割模型已开启 retina_masks（高分辨率掩码）: %s", cfg.get("model")
            )
        if bool(cfg.get("poly_merge", False)):
            logging.info(
                "分割多边形相似合并已开启 (edge_px=%.1f, edge_ratio=%.2f, "
                "contain_ratio=%.2f, cross_class=%s): %s",
                float(cfg.get("poly_merge_edge_px", 5.0)),
                float(cfg.get("poly_merge_edge_ratio", 0.5)),
                float(cfg.get("poly_merge_contain_ratio", 0.7)),
                bool(cfg.get("poly_merge_cross_class", True)),
                cfg.get("model"),
            )
        return predictor

    def _emit_detect_route_result(
        self,
        row: dict[str, Any],
        *,
        route_entry: dict[str, Any] | None,
        root_id: str,
        filtered_out: list[dict[str, Any]] | None,
        route_key: str | None = None,
    ) -> list[dict[str, Any]]:
        """检测根路由叶节点输出：嵌套分类完成后再做 mask_rate / 边缘联合过滤。"""
        if (
            self._enable_mask_rate_filter
            and route_entry
            and not mask_rate_passes(row, route_entry)
        ):
            if filtered_out is not None:
                filtered_out.append(
                    _finalize_filtered(
                        row,
                        route_entry=route_entry,
                        root_id=root_id,
                        reason="mask_rate",
                        cn_index=self._cn_display_index,
                    )
                )
            return []
        root_cfg = self._roots.get(root_id, {})
        if _edge_reject_should_filter_row(row, root_cfg):
            if filtered_out is not None:
                filtered_out.append(
                    _finalize_filtered(
                        row,
                        route_entry=route_entry,
                        root_id=root_id,
                        reason="edge",
                        cn_index=self._cn_display_index,
                    )
                )
            return []
        return [
            _finalize_result(
                row,
                route_entry=route_entry,
                route_key=route_key,
                root_id=root_id,
                cn_index=self._cn_display_index,
            )
        ]

    def _walk_route(
        self,
        row: dict[str, Any],
        image_bgr: np.ndarray,
        route_table: dict[str, Any] | None,
        *,
        root_id: str,
        route_path: str = "",
        dia_enabled: bool = True,
        skip_dia_conf: float | None = None,
        filtered_out: list[dict[str, Any]] | None = None,
        img_w: int = 0,
        root_cfg_for_dia: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """按 out 表递归：models.cls → 再进入该 cls 的 out。"""
        if not route_path:
            route_path = root_id
        cls_key = _primary_class_name(row)
        if route_table is not None:
            if _route_table_wants_direct_output(route_table):
                return self._emit_detect_route_result(
                    row,
                    route_entry=None,
                    root_id=root_id,
                    filtered_out=filtered_out,
                )
            if cls_key == "other":
                if filtered_out is not None:
                    filtered_out.append(
                        _finalize_filtered(
                            row,
                            route_entry=None,
                            root_id=root_id,
                            reason=_resolve_other_filter_reason(row),
                            cn_index=self._cn_display_index,
                        )
                    )
                return []
            _pat, entry = resolve_out_route_entry(
                route_table,
                cls_key,
                dia_px=bbox_diag_px_from_row(row),
                entry_enabled=_out_entry_is_enabled,
            )
            matched_route_key = str(_pat or cls_key or "").strip() or cls_key
            if entry is None:
                if filtered_out is not None:
                    filtered_out.append(
                        _finalize_filtered(
                            row,
                            route_entry=None,
                            root_id=root_id,
                            reason="no_route",
                            cn_index=self._cn_display_index,
                        )
                    )
                return []
            if not _out_entry_is_enabled(entry):
                if filtered_out is not None:
                    filtered_out.append(
                        _finalize_filtered(
                            row,
                            route_entry=entry if isinstance(entry, dict) else None,
                            root_id=root_id,
                            reason="disabled",
                            cn_index=self._cn_display_index,
                        )
                    )
                return []
            # null / {}：叶节点，无嵌套 models；直接输出当前实例（分类名取自 row）
            if entry is None or (isinstance(entry, dict) and not entry):
                return self._emit_detect_route_result(
                    row,
                    route_entry=entry,
                    route_key=matched_route_key,
                    root_id=root_id,
                    filtered_out=filtered_out,
                )
            if not isinstance(entry, dict):
                return []
            cls_raw = _resolve_cls_cfg_from_route_entry(entry)
            cls_merged: dict[str, Any] | None = None
            cls_route_label = ""
            if cls_raw is not None:
                cls_merged = _coalesce(cls_raw, _CLS_NODE_DEFAULTS)
                cls_route_label = f"{route_path}.out[{cls_key}].models.cls"

            root_cfg = self._roots.get(root_id, {})
            root_cfg_for_dia = root_cfg_for_dia or root_cfg
            effective_skip_dia = resolve_skip_dia_conf(cls_merged) or skip_dia_conf
            cls_first = bool(
                cls_merged is not None
                and _is_enabled(cls_merged)
                and cls_merged.get("model")
                and effective_skip_dia is not None
            )
            row_work = row
            if cls_first:
                runner = self._make_cls_runner(
                    cls_merged,
                    route_label=cls_route_label,
                    root_cfg=root_cfg,
                )
                row_work = runner.classify_row(image_bgr, row)
                if row_work is None or row_work.get("filter"):
                    if filtered_out is not None:
                        filtered_out.append(
                            _finalize_filtered(
                                row_work if row_work is not None else row,
                                route_entry=entry,
                                root_id=root_id,
                                reason=_resolve_cls_filter_reason(row_work),
                                cls_cfg=cls_merged,
                                cn_index=self._cn_display_index,
                            )
                        )
                    return []

            if not _passes_entry_thresholds(
                row_work,
                entry,
                root_cfg,
                dia_enabled=dia_enabled,
                skip_dia_conf=effective_skip_dia,
            ):
                if filtered_out is not None:
                    row_for_label = row_work
                    # 父级 detect_conf/dia 在分类前拦截时，仍补跑分类供绘图展示最终类名
                    if (
                        cls_merged
                        and _is_enabled(cls_merged)
                        and not cls_first
                    ):
                        runner = self._make_cls_runner(
                            cls_merged,
                            route_label=cls_route_label,
                            root_cfg=root_cfg,
                        )
                        row2 = runner.classify_row(image_bgr, row)
                        if row2 is not None:
                            row_for_label = row2
                    filtered_out.append(
                        _finalize_filtered(
                            row_for_label,
                            route_entry=entry,
                            root_id=root_id,
                            reason=_threshold_filter_reason(
                                row_for_label,
                                entry,
                                root_cfg,
                                dia_enabled=dia_enabled,
                                skip_dia_conf=effective_skip_dia,
                            ),
                            cls_cfg=cls_merged,
                            cn_index=self._cn_display_index,
                        )
                    )
                return []

            if cls_merged is not None and _is_enabled(cls_merged):
                if not cls_merged.get("model"):
                    logging.warning(
                        "[%s] 路由项已配置 models.cls 但缺少 model，跳过分类",
                        cls_route_label or route_path,
                    )
                    return self._emit_detect_route_result(
                        row_work,
                        route_entry=entry,
                        route_key=matched_route_key,
                        root_id=root_id,
                        filtered_out=filtered_out,
                    )
                if cls_first:
                    row2 = row_work
                else:
                    runner = self._make_cls_runner(
                        cls_merged,
                        route_label=cls_route_label,
                        root_cfg=root_cfg,
                    )
                    row2 = runner.classify_row(image_bgr, row)
                if row2 is None or row2.get("filter"):
                    if filtered_out is not None:
                        filtered_out.append(
                            _finalize_filtered(
                                row2 if row2 is not None else row,
                                route_entry=entry,
                                root_id=root_id,
                                reason=_resolve_cls_filter_reason(row2),
                                cls_cfg=cls_merged,
                                cn_index=self._cn_display_index,
                            )
                        )
                    return []
                child_out = cls_merged.get("out")
                if child_out is None or (
                    isinstance(child_out, dict) and not child_out
                ):
                    return self._emit_detect_route_result(
                        row2,
                        route_entry=entry,
                        route_key=matched_route_key,
                        root_id=root_id,
                        filtered_out=filtered_out,
                    )
                return self._walk_route(
                    row2,
                    image_bgr,
                    child_out,
                    root_id=root_id,
                    route_path=cls_route_label,
                    dia_enabled=_resolve_dia_enabled(
                        cls_merged, img_w, fallback_cfg=root_cfg_for_dia
                    ),
                    skip_dia_conf=effective_skip_dia,
                    filtered_out=filtered_out,
                    img_w=img_w,
                    root_cfg_for_dia=root_cfg_for_dia,
                )
            return self._emit_detect_route_result(
                row_work,
                route_entry=entry,
                route_key=matched_route_key,
                root_id=root_id,
                filtered_out=filtered_out,
            )

        return self._emit_detect_route_result(
            row,
            route_entry=None,
            root_id=root_id,
            filtered_out=filtered_out,
        )

    def _get_detect_predictor(self, root_id: str, cfg: dict[str, Any]) -> PredictSize:
        if root_id not in self._detect_predictors:
            self._detect_predictors[root_id] = self._create_detect_predictor(root_id, cfg)
        return self._detect_predictors[root_id]

    def _get_segment_predictor(self, root_id: str, cfg: dict[str, Any]) -> PredictSeg:
        if root_id not in self._segment_predictors:
            self._segment_predictors[root_id] = self._create_segment_predictor(root_id, cfg)
        return self._segment_predictors[root_id]

    def _run_detect_root(
        self,
        image_bgr: np.ndarray,
        root_id: str,
        cfg: dict[str, Any],
        *,
        filtered_out: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        predictor = self._get_detect_predictor(root_id, cfg)
        sq = resolve_model_square_options(cfg, model_type="detect")
        collect_filtered = filtered_out is not None
        _img_w = int(image_bgr.shape[1]) if getattr(image_bgr, "shape", None) is not None else 0
        _dia_enabled = _resolve_dia_enabled(cfg, _img_w)
        _edge_rule_enabled = bool(int(cfg.get("edge_reject_distance", 5)) > 0)
        _edge_reject_distance = int(cfg.get("edge_reject_distance", 5))
        # 这些阈值若未在 cfg 显式配置，predictor 内部仍可能有默认值；
        # 这里写入“实际生效值”到 row，避免 filter_reason 推断失真。
        _edge_reject_conf_threshold = cfg.get("edge_reject_conf_threshold")
        _edge_reject_cls_conf_threshold = cfg.get("edge_reject_cls_conf_threshold")
        _bin_dark_ratio_min_cfg = cfg.get("bin_dark_ratio_min")
        _bin_dark_ratio_min_eff = (
            float(_bin_dark_ratio_min_cfg)
            if _bin_dark_ratio_min_cfg is not None
            else float(getattr(predictor, "bin_dark_ratio_min", 0.0) or 0.0)
        )
        # 边缘联合过滤在 _walk_route 嵌套分类之后执行（使用最终 cls_conf），
        # PredictSize 内仅保留 edge_min_dist 等字段，不在此步按检测 conf 误滤。
        _clip_profiles = resolve_clip_profiles_from_cfg(cfg, global_cfg=self.config)
        _clip_batch = resolve_clip_batch_size_from_cfg(cfg, global_cfg=self.config)
        _logged = getattr(self, "_detect_batch_log_roots", None)
        if _logged is None:
            _logged = set()
            self._detect_batch_log_roots = _logged
        if root_id not in _logged:
            _logged.add(root_id)
            logging.info(
                "[detect/%s] clip_batch_size=%s profiles=%s clip_profiles_enable=%s",
                root_id,
                _clip_batch,
                [
                    f"{p.clip_size}/{p.overlap_size}@{p.clip_start}"
                    for p in _clip_profiles
                ],
                resolve_clip_profiles_enable(cfg, global_cfg=self.config),
            )
        _predict_clip: dict[str, Any] = (
            {"clip_profiles": _clip_profiles, "clip_batch_size": _clip_batch}
            if len(_clip_profiles) > 1
            else {
                "clip_size": _clip_profiles[0].clip_size,
                "overlap_size": _clip_profiles[0].overlap_size,
                "clip_start": _clip_profiles[0].clip_start,
                "clip_batch_size": _clip_batch,
            }
        )
        with self._phase_timer("detect"):
            rows = predictor.predict(
                image_bgr,
                **_predict_clip,
                edge_reject_distance=0,
                edge_reject_conf_threshold=None,
                edge_reject_cls_conf_threshold=None,
                cls_pad_square=sq["cls_pad_square"],
                cls_gray_binarize=bool(cfg.get("gray_binarize", False)),
                detect_pad_square=sq["detect_pad_square"],
                detect_pad_square_full_image=sq["detect_pad_square_full_image"],
                detect_nms_iou=cfg.get("nms_iou"),
                detect_max_det=cfg.get("max_det"),
                detect_nms_agnostic=cfg.get("nms_agnostic"),
                diag_filter_range=cfg.get("dia") if _dia_enabled else None,
                inner_boxes_fp_threshold=cfg.get("inner_boxes_fp_threshold"),
                bin_dark_ratio_min=_bin_dark_ratio_min_eff,
                edge_dup_diag_ratio=cfg.get("edge_dup_diag_ratio"),
                return_full_final=collect_filtered,
            )
            for r in rows:
                if isinstance(r, dict):
                    r["edge_rule_enabled"] = _edge_rule_enabled
                    r["dia_enabled"] = _dia_enabled
                    r["edge_reject_distance"] = _edge_reject_distance
                    r["edge_reject_conf_threshold"] = _edge_reject_conf_threshold
                    r["edge_reject_cls_conf_threshold"] = _edge_reject_cls_conf_threshold
                    r["bin_dark_ratio_min"] = _bin_dark_ratio_min_eff

            if collect_filtered:
                passed: list[dict[str, Any]] = []
                for row in rows:
                    if row.get("filter"):
                        filtered_out.append(
                            _finalize_filtered(
                                row,
                                route_entry=None,
                                root_id=root_id,
                                reason=_detect_filter_reason(row),
                                cn_index=self._cn_display_index,
                            )
                        )
                    else:
                        passed.append(row)
                rows = passed

        out_route = cfg.get("out")
        self._prefill_trt_cls_for_rows(
            rows,
            image_bgr,
            out_route if isinstance(out_route, dict) else None,
            root_id=root_id,
            root_cfg=cfg,
            dia_enabled=_dia_enabled,
            skip_dia_conf=resolve_skip_dia_conf(cfg),
            img_w=_img_w,
        )
        results: list[dict[str, Any]] = []
        with self._phase_timer("detect"):
            for row in rows:
                if str(row.get("cls_name", "") or "") == "other":
                    if filtered_out is not None:
                        filtered_out.append(
                            _finalize_filtered(
                                row,
                                route_entry=None,
                                root_id=root_id,
                                reason=_resolve_other_filter_reason(row),
                                cn_index=self._cn_display_index,
                            )
                        )
                    continue
                results.extend(
                    self._walk_route(
                        row,
                        image_bgr,
                        out_route,
                        root_id=root_id,
                        dia_enabled=_dia_enabled,
                        filtered_out=filtered_out,
                        img_w=_img_w,
                        root_cfg_for_dia=cfg,
                    )
                )
        return results

    def _run_segment_root(
        self,
        image_bgr: np.ndarray,
        root_id: str,
        cfg: dict[str, Any],
        *,
        filtered_out: list[dict[str, Any]] | None = None,
        source_image_stem: str | None = None,
        result_output_dir: str | Path | None = None,
        roi_circle: tuple[int, int, int] | None = None,
    ) -> list[dict[str, Any]]:
        predictor = self._get_segment_predictor(root_id, cfg)
        seg_imgsz = resolve_seg_imgsz(cfg)
        sq = resolve_model_square_options(cfg, model_type="segment")
        collect_filtered = filtered_out is not None
        gray_enh = resolve_gray_contrast_options(cfg)
        debug_dir: str | None = None
        if gray_enh.get("gray_contrast_debug_save") and result_output_dir:
            debug_dir = str(result_output_dir)
        # dia_switch 关闭时不做尺寸过滤；开启时仅在图片宽度命中 dia_w 时生效（与 _run_detect_root 一致）。
        _img_w = int(image_bgr.shape[1]) if getattr(image_bgr, "shape", None) is not None else 0
        _dia_enabled = _resolve_dia_enabled(cfg, _img_w)
        _clip_profiles = resolve_clip_profiles_from_cfg(cfg, global_cfg=self.config)
        _clip_batch = resolve_clip_batch_size_from_cfg(cfg, global_cfg=self.config)
        _predict_clip: dict[str, Any] = (
            {"clip_profiles": _clip_profiles, "clip_batch_size": _clip_batch}
            if len(_clip_profiles) > 1
            else {
                "clip_size": _clip_profiles[0].clip_size,
                "overlap_size": _clip_profiles[0].overlap_size,
                "clip_start": _clip_profiles[0].clip_start,
                "clip_batch_size": _clip_batch,
            }
        )
        with self._phase_timer("seg"):
            rows = predictor.predict(
                image_bgr,
                **_predict_clip,
                padding=bool(cfg.get("padding", True)),
                pad_full_image_to_square=sq["pad_full_image_to_square"],
                min_size=int(cfg.get("min_instance_size", 3)),
                max_size=cfg.get("max_instance_size"),
                imgsz=seg_imgsz if seg_imgsz > 0 else None,
                nms_iou=cfg.get("nms_iou"),
                max_det=cfg.get("max_det"),
                nms_agnostic=cfg.get("nms_agnostic"),
                retina_masks=bool(cfg.get("retina_masks", False)),
                cls_pad_square=sq["cls_pad_square"],
                cls_crop_from_bbox=bool(cfg.get("from_bbox", False)),
                cls_crop_background=cfg.get("cls_crop_background"),
                cls_pad_color=cfg.get("cls_pad_color"),
                return_all_rows=collect_filtered,
                debug_image_stem=source_image_stem,
                gray_contrast_debug_dir=debug_dir,
                roi_circle=roi_circle,
            )
            for r in rows:
                if isinstance(r, dict):
                    r["dia_enabled"] = _dia_enabled

            if collect_filtered:
                passed: list[dict[str, Any]] = []
                for row in rows:
                    if row.get("filter"):
                        filtered_out.append(
                            _finalize_filtered(
                                row,
                                route_entry=None,
                                root_id=root_id,
                                reason=str(row.get("filter_reason") or "seg"),
                                cn_index=self._cn_display_index,
                            )
                        )
                    else:
                        passed.append(row)
                rows = passed

            dia_root = cfg.get("dia")
            if _dia_enabled and isinstance(dia_root, (list, tuple)) and len(dia_root) >= 2:
                if filtered_out is not None:
                    rows, dia_drop = _partition_rows_by_bbox_diag(
                        rows, float(dia_root[0]), float(dia_root[1])
                    )
                    for row in dia_drop:
                        filtered_out.append(
                            _finalize_filtered(
                                row,
                                route_entry=None,
                                root_id=root_id,
                                reason="dia",
                                cn_index=self._cn_display_index,
                            )
                        )
                else:
                    rows, _ = filter_rows_by_bbox_diag_range(
                        rows, float(dia_root[0]), float(dia_root[1])
                    )

        out_route = cfg.get("out")
        self._prefill_trt_cls_for_rows(
            rows,
            image_bgr,
            out_route if isinstance(out_route, dict) else None,
            root_id=root_id,
            root_cfg=cfg,
            dia_enabled=_dia_enabled,
            skip_dia_conf=resolve_skip_dia_conf(cfg),
            img_w=_img_w,
        )
        results: list[dict[str, Any]] = []
        with self._phase_timer("seg"):
            for row in rows:
                seg_nm = str(row.get("class_name", row.get("seg_cls_name", "")) or "")
                if not row.get("cls_name"):
                    row = dict(row)
                    row["cls_name"] = seg_nm
                if str(row.get("cls_name", "") or "") == "other":
                    if filtered_out is not None:
                        filtered_out.append(
                            _finalize_filtered(
                                row,
                                route_entry=None,
                                root_id=root_id,
                                reason=_resolve_other_filter_reason(row),
                                cn_index=self._cn_display_index,
                            )
                        )
                    continue
                results.extend(
                    self._walk_route(
                        row,
                        image_bgr,
                        out_route,
                        root_id=root_id,
                        dia_enabled=_dia_enabled,
                        filtered_out=filtered_out,
                        img_w=_img_w,
                        root_cfg_for_dia=cfg,
                    )
                )
        return results

    def _accumulate_root_phase(self, phase: PredictPhaseSample | None) -> None:
        if phase is None or self._phase_recorder._current is None:
            return
        cur = self._phase_recorder._current
        for attr, _ in _PHASE_LABELS:
            setattr(cur, attr, getattr(cur, attr) + getattr(phase, attr))

    def _run_one_root(
        self,
        root_id: str,
        cfg: dict[str, Any],
        mtype: str,
        root_image: np.ndarray,
        *,
        roi_circle: tuple[int, int, int] | None = None,
        collect_filtered: bool = False,
        source_image_stem: str | None = None,
        result_output_dir: str | Path | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], PredictPhaseSample | None]:
        local_filtered: list[dict[str, Any]] | None = [] if collect_filtered else None
        use_gpu_crop = resolve_use_gpu_crop(self.config)
        with gpu_crop_session_scope(
            root_image, device=self.device, enabled=use_gpu_crop
        ):
            self._begin_root_run_tls()
            try:
                if mtype == "detect":
                    part = self._run_detect_root(
                        root_image, root_id, cfg, filtered_out=local_filtered
                    )
                elif mtype == "segment":
                    part = self._run_segment_root(
                        root_image,
                        root_id,
                        cfg,
                        filtered_out=local_filtered,
                        source_image_stem=source_image_stem,
                        result_output_dir=result_output_dir,
                        roi_circle=roi_circle,
                    )
                else:
                    part = []
            finally:
                phase = self._end_root_run_tls()
        return part, local_filtered or [], phase

    def _run_root_group(
        self,
        tasks: list[tuple[str, dict[str, Any], str, np.ndarray, tuple[int, int, int] | None]],
        *,
        collect_filtered: bool,
        source_image_stem: str | None,
        result_output_dir: str | Path | None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], PredictPhaseSample | None]:
        all_parts: list[dict[str, Any]] = []
        all_filtered: list[dict[str, Any]] = []
        phases: list[PredictPhaseSample] = []
        for root_id, cfg, mtype, root_image, roi_circle in tasks:
            part, filt, phase = self._run_one_root(
                root_id,
                cfg,
                mtype,
                root_image,
                roi_circle=roi_circle,
                collect_filtered=collect_filtered,
                source_image_stem=source_image_stem,
                result_output_dir=result_output_dir,
            )
            all_parts.extend(part)
            all_filtered.extend(filt)
            if phase is not None:
                phases.append(phase)
        merged = (
            merge_predict_phase_samples(phases, overlap=len(phases) > 1)
            if phases
            else None
        )
        return all_parts, all_filtered, merged

    def predict(
        self,
        image_bgr: np.ndarray,
        *,
        root_ids: list[str] | None = None,
        collect_filtered: bool = False,
        source_image_stem: str | None = None,
        result_output_dir: str | Path | None = None,
        eval_metrics_root: str | Path | None = None,
        phase_profile: bool | None = None,
    ) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        对单张 BGR 图执行所有（或指定）根模型推理。

        返回统一列表：name, score, location, source[, cn_name, polygon]。
        ``collect_filtered=True`` 时返回 ``(results, filtered)``，filtered 为被过滤实例。
        ``phase_profile=True`` 或配置 ``predict_phase_profile`` 时采集分阶段耗时。
        """
        if image_bgr is None or getattr(image_bgr, "size", 0) == 0:
            return ([], []) if collect_filtered else []

        prev_phase_enabled = self._phase_recorder.enabled
        if phase_profile is not None:
            self._phase_recorder.enabled = bool(phase_profile)
        recording = self._phase_recorder.enabled
        if recording:
            self._phase_recorder.begin_image()

        try:
            want = set(root_ids) if root_ids else None
            all_results: list[dict[str, Any]] = []
            need_in_big_override = bool(
                self._in_big_skip_det_conf_by_root
                or self._in_big_skip_big_conf_by_root
            )
            root_collect_filtered = collect_filtered or need_in_big_override
            filtered_acc: list[dict[str, Any]] | None = (
                [] if root_collect_filtered else None
            )
            root_tasks: list[
                tuple[str, dict[str, Any], str, np.ndarray, tuple[int, int, int] | None]
            ] = []
            for root_id, cfg in self._roots.items():
                if want is not None and root_id not in want:
                    continue
                root_image = image_bgr
                roi_circle: tuple[int, int, int] | None = None
                if bool(cfg.get("roi_switch", False)):
                    with self._phase_timer("post"):
                        roi_res = apply_roi_preprocess(image_bgr, cfg)
                    root_image = roi_res.image
                    roi_circle = roi_circle_from_apply(roi_res)
                    if roi_res.applied:
                        logging.debug(
                            "根模型 %s ROI 预处理: type=%s center=%s radius=%s score=%.3f",
                            root_id,
                            roi_res.roi_type,
                            roi_res.center,
                            roi_res.radius,
                            roi_res.score,
                        )
                mtype = str(cfg.get("model_type", "detect")).strip().lower()
                if mtype not in ("detect", "segment"):
                    continue
                root_tasks.append((root_id, cfg, mtype, root_image, roi_circle))

            parallel_detect_seg = bool(
                resolve_predict_cfg_value(
                    "parallel_detect_seg", self.config, default=True
                )
            )
            detect_tasks = [t for t in root_tasks if t[2] == "detect"]
            seg_tasks = [t for t in root_tasks if t[2] == "segment"]
            can_parallel = (
                parallel_detect_seg and bool(detect_tasks) and bool(seg_tasks)
            )
            group_kw = {
                "collect_filtered": root_collect_filtered,
                "source_image_stem": source_image_stem,
                "result_output_dir": result_output_dir,
            }
            if can_parallel:
                with ThreadPoolExecutor(max_workers=2) as ex:
                    f_detect = ex.submit(self._run_root_group, detect_tasks, **group_kw)
                    f_seg = ex.submit(self._run_root_group, seg_tasks, **group_kw)
                    d_parts, d_filt, d_phase = f_detect.result()
                    s_parts, s_filt, s_phase = f_seg.result()
                all_results.extend(d_parts)
                all_results.extend(s_parts)
                if filtered_acc is not None:
                    filtered_acc.extend(d_filt)
                    filtered_acc.extend(s_filt)
                group_phases = [p for p in (d_phase, s_phase) if p is not None]
                if group_phases:
                    merged = merge_predict_phase_samples(
                        group_phases,
                        overlap=True,
                        roots_parallel=True,
                    )
                    self._accumulate_root_phase(merged)
                    if self._phase_recorder._current is not None:
                        self._phase_recorder._current.roots_parallel = True
            else:
                for root_id, cfg, mtype, root_image, roi_circle in root_tasks:
                    part, filt, phase = self._run_one_root(
                        root_id,
                        cfg,
                        mtype,
                        root_image,
                        roi_circle=roi_circle,
                        collect_filtered=root_collect_filtered,
                        source_image_stem=source_image_stem,
                        result_output_dir=result_output_dir,
                    )
                    all_results.extend(part)
                    if filtered_acc is not None:
                        filtered_acc.extend(filt)
                    self._accumulate_root_phase(phase)
            with self._phase_timer("post"):
                all_results, in_big_filtered = self._filter_small_in_big_masks(
                    all_results,
                    image_bgr,
                    filtered_big_rows=filtered_acc,
                    source_image_stem=source_image_stem,
                    result_output_dir=result_output_dir,
                    eval_metrics_root=eval_metrics_root,
                )
                all_results = self._apply_report_tier_filter(
                    all_results, filtered_acc=filtered_acc
                )
                all_results, in_big_filtered, filtered_acc = (
                    self._resolve_in_big_skip_override(
                        all_results,
                        in_big_filtered,
                        filtered_acc,
                        image_bgr,
                    )
                )
                if collect_filtered and filtered_acc is not None and in_big_filtered:
                    filtered_acc.extend(in_big_filtered)
            if collect_filtered:
                return all_results, filtered_acc or []
            return all_results
        finally:
            if recording:
                sample = self._phase_recorder.end_image()
                if sample is not None and logging.getLogger().isEnabledFor(logging.DEBUG):
                    logging.debug(
                        "predict 分阶段耗时:%s",
                        format_predict_phase_line(sample),
                    )
            self._phase_recorder.enabled = prev_phase_enabled

    def predict_path(self, image_path: str | Path, **kwargs: Any) -> list[dict[str, Any]]:
        img = load_image_bgr_from_ref(str(image_path))
        if img is None:
            logging.warning("无法读取图片: %s", image_path)
            return []
        stem = kwargs.pop("source_image_stem", None) or Path(image_path).stem
        return self.predict(img, source_image_stem=stem, **kwargs)

    def predict_ref(
        self, image_ref: str, **kwargs: Any
    ) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """从 URL 或本地路径读图后执行 ``predict``（供多进程 worker 使用）。"""
        ref = str(image_ref or "").strip()
        if not ref:
            if kwargs.get("collect_filtered"):
                return [], []
            return []
        img = load_image_bgr_from_ref(ref)
        if img is None or getattr(img, "size", 0) == 0:
            raise ValueError(f"图片不存在或解码失败 ref={ref!r}")
        stem = kwargs.pop("source_image_stem", None)
        if stem is None and not ref.startswith(("http://", "https://")):
            stem = Path(ref).stem
        return self.predict(img, source_image_stem=stem, **kwargs)

    def warmup_models(self) -> dict[str, Any]:
        """
        启动时预加载 detect/segment 与嵌套 cls 权重（避免首次 predict 才占 GPU）。

        返回摘要：``{load_s, detect, segment, cls}``。
        """
        t0 = time.perf_counter()
        n_detect = 0
        n_seg = 0
        n_cls = 0
        if not self._roots:
            logging.info("[warmup] 无启用的根模型，跳过权重预加载")
            return {"load_s": 0.0, "detect": 0, "segment": 0, "cls": 0}
        for root_id, cfg in self._roots.items():
            mtype = str(cfg.get("model_type", "detect")).strip().lower()
            if mtype == "segment":
                self._get_segment_predictor(root_id, cfg)
                n_seg += 1
            else:
                self._get_detect_predictor(root_id, cfg)
                n_detect += 1
            for path, cls_cfg in _iter_cls_route_nodes(cfg, root_id):
                if not _is_enabled(cls_cfg) or not cls_cfg.get("model"):
                    continue
                pad_clr = resolve_cls_pad_color(cls_cfg.get("cls_pad_color"))
                key = cls_cache_key({**cls_cfg, "cls_pad_color": pad_clr})
                if key in self._cls_cache:
                    continue
                with self._cls_cache_lock:
                    if key in self._cls_cache:
                        continue
                    infer_path = resolve_inference_model_path(
                        cls_cfg, log_label=path
                    )
                    logging.info(
                        "[warmup] 加载嵌套分类 %s infer=%s backend=%s",
                        path,
                        infer_path,
                        cls_cfg.get("cls_backend", "auto"),
                    )
                    self._cls_cache[key] = create_classifier(
                        str(cls_cfg["model"]),
                        device=self.device,
                        pad_square=bool(cls_cfg.get("to_square", True)),
                        gray_binarize=bool(cls_cfg.get("gray_binarize", False)),
                        pad_color_bgr=pad_clr,
                        to_gray=bool(cls_cfg.get("to_gray", False)),
                        cfg=cls_cfg,
                    )
                    n_cls += 1
        load_s = time.perf_counter() - t0
        summary = {
            "load_s": load_s,
            "detect": n_detect,
            "segment": n_seg,
            "cls": n_cls,
            "roots": list(self._roots.keys()),
        }
        logging.info(
            "[warmup] 模型权重预加载完成 %.2fs detect=%d segment=%d cls=%d 根模型=%s",
            load_s,
            n_detect,
            n_seg,
            n_cls,
            summary["roots"],
        )
        self._warmup_gpu_forward()
        self._log_cuda_memory("warmup")
        return summary

    def _log_cuda_memory(self, tag: str) -> None:
        try:
            import torch

            if not torch.cuda.is_available():
                logging.warning("[%s] torch.cuda.is_available()=False（权重可能在 CPU）", tag)
                return
            idx = 0
            if self.device and ":" in str(self.device):
                idx = int(str(self.device).rsplit(":", 1)[-1])
            alloc = torch.cuda.memory_allocated(idx) / (1024**2)
            reserved = torch.cuda.memory_reserved(idx) / (1024**2)
            logging.info(
                "[%s] CUDA:%d allocated=%.0fMiB reserved=%.0fMiB",
                tag,
                idx,
                alloc,
                reserved,
            )
        except Exception:
            logging.debug("[%s] CUDA 内存探测跳过", tag, exc_info=True)

    def _warmup_gpu_forward(self) -> None:
        """
        Ultralytics/YOLO 常在首次 ``predict`` 才把权重迁到 GPU；
        启动阶段跑一次 dummy 前向，使 ``nvidia-smi`` 可见显存占用。
        """
        if not self._roots:
            return
        prev_phase = self._phase_recorder.enabled
        self._phase_recorder.enabled = False
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        logging.info("[warmup] dummy 前向推理以激活 GPU …")
        t0 = time.perf_counter()
        try:
            self.predict(dummy, collect_filtered=False)
            logging.info("[warmup] dummy 前向完成 %.2fs", time.perf_counter() - t0)
        except Exception:
            logging.exception(
                "[warmup] dummy 前向失败（首次真实推理仍会尝试加载）"
            )
        finally:
            self._phase_recorder.enabled = prev_phase

    def release(self) -> None:
        for p in self._detect_predictors.values():
            p.release()
        self._detect_predictors.clear()
        for p in self._segment_predictors.values():
            p.release()
        self._segment_predictors.clear()
        self._cls_cache.clear()
        self._trt_cls_prefill.clear()
        clear_yolo_cache()
        logging.info("InsectPredictAll 已释放全部模型缓存")


def load_image_bgr_from_ref(url_or_path: str) -> np.ndarray | None:
    """
    从远程 URL 或本地路径加载 BGR 图像。

    ``http(s)://`` 走下载解码；其余字符串视为服务端本地磁盘路径。
    失败返回 ``None``（本地路径）或抛出 ``ValueError``（远程 URL 解码失败）。
    """
    from urllib.request import Request as UrlRequest
    from urllib.request import urlopen

    ref = str(url_or_path or "").strip()
    if not ref:
        return None
    if ref.startswith(("http://", "https://")):
        req = UrlRequest(ref, headers={"User-Agent": "insect-predict/1.0"})
        with urlopen(req, timeout=60) as resp:
            data = resp.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None or getattr(img, "size", 0) == 0:
            raise ValueError(f"图片解码失败 url={ref!r}")
        return img
    img = cv2.imread(ref, cv2.IMREAD_COLOR)
    if img is None or getattr(img, "size", 0) == 0:
        return None
    return img


def create_pipeline(
    config_path: str | Path | None = None,
    *,
    device: str | None = None,
    root_ids: list[str] | None = None,
    enable_mask_rate_filter: bool = True,
) -> InsectPredictAll:
    """工厂：加载 insect_alg_all.json 并构建管线。"""
    return InsectPredictAll(
        config_path,
        device=device,
        root_ids=root_ids,
        enable_mask_rate_filter=enable_mask_rate_filter,
    )


def predict(
    image_bgr: np.ndarray,
    *,
    config_path: str | Path | None = None,
    device: str | None = None,
    root_ids: list[str] | None = None,
    pipeline: InsectPredictAll | None = None,
) -> list[dict[str, Any]]:
    """
    便捷函数：单次推理。传入已构建的 pipeline 可复用模型加载。
    """
    if pipeline is None:
        pipe = create_pipeline(config_path, device=device, root_ids=root_ids)
        try:
            return pipe.predict(image_bgr, root_ids=root_ids)
        finally:
            pipe.release()
    return pipeline.predict(image_bgr, root_ids=root_ids)


# --------------------------------------------------------------------------- #
#  VOC XML / 标注校验（Demo 开关，默认关闭以保持原行为）
# --------------------------------------------------------------------------- #


def _eval_row_pred_class_name(
    r: dict[str, Any],
    *,
    use_infer_name: bool = False,
    use_cn_name: bool = False,
    cn_index: dict[str, str] | None = None,
) -> str:
    """校验/比赛计数用预测类名：后置 other 不计入原类识别数。"""
    if _is_filtered_other_row(r):
        return "other"
    if _is_other_class_name(str(r.get("cls_name") or "")):
        return "other"
    if use_cn_name:
        return _label_display_name(r, cn_index=cn_index)
    if use_infer_name:
        return str(r.get("infer_name") or r.get("name") or "unknown")
    return str(r.get("name") or r.get("cls_name") or "unknown")


def _result_row_to_pred_dict(
    r: dict[str, Any],
    *,
    use_infer_name: bool = False,
    use_cn_name: bool = False,
    cn_index: dict[str, str] | None = None,
) -> dict[str, Any]:
    """单条 predict/filtered 结果转为几何匹配用的 pred dict。"""
    loc = r.get("location") or [0, 0, 0, 0]
    x1, y1, x2, y2 = (int(loc[0]), int(loc[1]), int(loc[2]), int(loc[3]))
    det = float(r.get("det_conf", r.get("score", 0.0)) or 0.0)
    cls_c = float(r.get("cls_conf", r.get("score", det)) or det)
    cls_name = _eval_row_pred_class_name(
        r,
        use_infer_name=use_infer_name,
        use_cn_name=use_cn_name,
        cn_index=cn_index,
    )
    out = {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "cls_name": cls_name,
        "conf": det,
        "cls_conf": cls_c,
    }
    for k in (
        "name",
        "cls_name_top1",
        "cls_conf_top1",
        "cls_other_reason",
        "filter_reason",
        "cls_topk",
        "cls_top3",
        "seg_cls_name",
        "class_name",
        "viz_cls_name",
    ):
        if k in r:
            out[k] = r[k]
    cls_topn = _build_cls_topn_xml_payload(
        r,
        use_cn_name=use_cn_name,
        cn_index=cn_index,
    )
    if cls_topn:
        out["cls_topn"] = cls_topn
    return out


def _results_to_pred_rows(
    results: list[dict[str, Any]],
    *,
    use_infer_name: bool = False,
    use_cn_name: bool = False,
    cn_index: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """将 predict 统一结果转为 write_pascal_voc_xml / 几何匹配用的行 dict。"""
    return [
        _result_row_to_pred_dict(
            r,
            use_infer_name=use_infer_name,
            use_cn_name=use_cn_name,
            cn_index=cn_index,
        )
        for r in results
    ]


def _is_filtered_other_row(r: dict[str, Any]) -> bool:
    """被路由/阈值置为 other 并进入 filtered 列表的实例。"""
    if _is_other_filter_reason(str(r.get("filter_reason") or "")):
        return True
    return _is_other_class_name(str(r.get("name") or r.get("cls_name") or ""))


def _build_eval_pred_pool(
    results: list[dict[str, Any]],
    filtered: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], list[bool], list[int]]:
    """
    构建验证用 pred 池：保留 results，并追加 filtered 中 ``other`` 实例仅参与几何匹配。

    返回 (pred_rows, match_only_other_flags, filtered_other_indices)。
    ``match_only_other=True`` 的 pred 若未匹配 GT 不计 FP；匹配后按类型错统计。
    """
    preds = [_result_row_to_pred_dict(r) for r in results]
    match_only_other = [False] * len(preds)
    filtered_other_indices: list[int] = []
    for fi, r in enumerate(filtered or []):
        if not _is_filtered_other_row(r):
            continue
        preds.append(_result_row_to_pred_dict(r))
        match_only_other.append(True)
        filtered_other_indices.append(fi)
    return preds, match_only_other, filtered_other_indices


def save_pascal_voc_xml_for_results(
    image_bgr: np.ndarray,
    results: list[dict[str, Any]],
    *,
    out_dir: str | Path,
    rel_path: Path,
    use_cn_name: bool = False,
    cn_index: dict[str, str] | None = None,
    include_dual_conf: bool = False,
) -> None:
    """将 predict 结果写成与输入图同目录的 Pascal VOC bbox xml。"""
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    h_img, w_img = image_bgr.shape[:2]
    depth = 3 if image_bgr.ndim >= 3 else 1
    xml_path = out_p / f"{Path(rel_path.name).stem}.xml"
    write_pascal_voc_xml(
        str(xml_path),
        folder_name=out_p.name or "",
        image_filename=rel_path.name,
        width=w_img,
        height=h_img,
        depth=depth,
        results=_results_to_pred_rows(
            results,
            use_infer_name=not use_cn_name,
            use_cn_name=use_cn_name,
            cn_index=cn_index,
        ),
        include_dual_conf=include_dual_conf,
    )


LABELME_VERSION = "5.0.1"


def _normalize_labelme_points(
    polygon: Any,
    img_w: int,
    img_h: int,
) -> list[list[float]]:
    """将 polygon（像素坐标）裁剪到图像内，供 LabelMe ``points`` 使用。"""
    if img_w <= 0 or img_h <= 0:
        return []
    pts: list[list[float]] = []
    for p in polygon or []:
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            continue
        try:
            px = float(p[0])
            py = float(p[1])
        except (TypeError, ValueError):
            continue
        pts.append(
            [
                max(0.0, min(float(img_w), px)),
                max(0.0, min(float(img_h), py)),
            ]
        )
    return pts if len(pts) >= 3 else []


def _result_to_labelme_shape(
    r: dict[str, Any],
    img_w: int,
    img_h: int,
    *,
    cn_index: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    """单条 predict 结果 → LabelMe shape（polygon）；无 mask 时退化为 bbox 四角。"""
    loc = r.get("location") or [0, 0, 0, 0]
    x1, y1, x2, y2 = int(loc[0]), int(loc[1]), int(loc[2]), int(loc[3])
    points = _normalize_labelme_points(r.get("polygon"), img_w, img_h)
    if len(points) < 3:
        points = _normalize_labelme_points(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            img_w,
            img_h,
        )
    if len(points) < 3:
        return None
    label = _label_display_name(r, cn_index=cn_index)
    det_conf = float(r.get("det_conf", r.get("score", 0.0)) or 0.0)
    cls_conf = float(r.get("cls_conf", r.get("score", det_conf)) or det_conf)
    return {
        "label": label,
        "points": points,
        "group_id": None,
        "description": f"det={det_conf:.3f} cls={cls_conf:.3f}",
        "shape_type": "polygon",
        "flags": {},
    }


def save_labelme_json_for_results(
    image_bgr: np.ndarray,
    results: list[dict[str, Any]],
    *,
    out_dir: str | Path,
    rel_path: Path,
    source_image_path: Path | str | None = None,
    copy_image: bool = True,
    cn_index: dict[str, str] | None = None,
) -> Path:
    """
    将 predict 结果写成 LabelMe 可打开的 JSON（与图同目录、同名 stem）。

    - 分割实例写 ``shape_type=polygon``；检测-only 无 polygon 时退化为外接框四角。
    - ``copy_image=True`` 时将原图复制到 ``out_dir``，``imagePath`` 为文件名，
      便于 ``labelme <out_dir>`` 批量打开校对。
    """
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    h_img, w_img = image_bgr.shape[:2]
    image_filename = Path(rel_path.name).name
    shapes: list[dict[str, Any]] = []
    for r in results:
        shape = _result_to_labelme_shape(r, w_img, h_img, cn_index=cn_index)
        if shape is not None:
            shapes.append(shape)
    if copy_image and source_image_path is not None:
        src = Path(source_image_path)
        if src.is_file():
            dest_image = out_p / image_filename
            if dest_image.resolve() != src.resolve():
                shutil.copy2(src, dest_image)
    doc = {
        "version": LABELME_VERSION,
        "flags": {},
        "shapes": shapes,
        "imagePath": image_filename,
        "imageData": None,
        "imageHeight": int(h_img),
        "imageWidth": int(w_img),
    }
    json_path = out_p / f"{Path(rel_path.name).stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return json_path


def _predict_all_has_validation_gt(
    img_path: Path,
    rel: Path,
    *,
    label_alias_map: dict[str, str] | None,
) -> bool:
    """是否存在可用于校验的 GT：同目录 VOC xml，或文件名可解析的中文/拼音类别。"""
    if img_path.with_suffix(".xml").is_file():
        return True
    return infer_gt_label_from_filename_stem(rel.stem, label_alias_map) is not None


def _load_predict_all_ground_truth(
    img_path: Path,
    rel: Path,
    img_bgr: np.ndarray | None,
    *,
    label_alias_map: dict[str, str] | None,
) -> tuple[list[dict[str, Any]] | None, str]:
    """
    加载校验用 GT：优先同目录 VOC xml；无 xml 时从文件名 stem 推断图级类别标签。

    文件名 GT 口径：不代表整图都是该虫，而是图内报出框均应判为该种类（见
    ``_accumulate_predict_all_filename_gt_validation``）。

    返回 ``(gts, source)``，``source`` 为 ``"xml"`` | ``"filename"`` | ``""``。
    """
    src_xml = img_path.with_suffix(".xml")
    if src_xml.is_file():
        try:
            gts = parse_pascal_voc_objects(str(src_xml))
            if gts:
                return gts, "xml"
        except Exception as e:
            logging.warning("读取标注 xml 失败 %s: %s", src_xml, e)

    pseudo = build_filename_pseudo_gt_objects(
        rel.stem, label_alias_map=label_alias_map
    )
    if pseudo:
        return pseudo, "filename"
    return None, ""


def _inc_stat_by_cls(
    stat_by_cls: dict[str, dict[str, int]], cls_name: str, key: str, n: int = 1
) -> None:
    cls_name = str(cls_name or "")
    if cls_name not in stat_by_cls:
        stat_by_cls[cls_name] = {"gt": 0, "pred": 0, "tp": 0, "fp": 0, "fn": 0, "cls_err": 0}
    stat_by_cls[cls_name][key] = int(stat_by_cls[cls_name].get(key, 0)) + int(n)


def _accumulate_predict_all_filename_gt_validation(
    stat_total: dict[str, int],
    stat_by_cls: dict[str, dict[str, int]],
    results: list[dict[str, Any]],
    gt_label: str,
    *,
    class_merge: dict[str, list[str]] | None,
    cls_merge_type_index: Any = None,
    label_alias_map: dict[str, str] | None = None,
    fuzzy_only_wildcard: bool = False,
    tier_equivalence: ClassTierEquivalence | None = None,
) -> tuple[list[tuple[int, int, float]], set[int], list[dict[str, Any]]]:
    """
    文件名 GT 校验：图内每个**报出框**（``results``）应按文件名类别分类，不做几何匹配。

    - 每个非 ``other`` 报出框计 1 次标注（gt）；类对 → tp，类错 → cls_err + fp（记在预测类上）；
    - 无任何非 ``other`` 报出框时计 1 次漏检（fn），表示该图未检出可评估目标。
    """
    preds = _results_to_pred_rows(results)
    merge = class_merge
    gt_name = str(gt_label or "").strip()
    if not gt_name or is_metric_ignored_other(gt_name, merge):
        return [], set(), preds

    gt_norm = normalize_class_name(gt_name, merge, label_alias_map=label_alias_map)
    pred_eval_idx = [
        i
        for i, r in enumerate(preds)
        if not is_metric_ignored_other(str(r.get("cls_name", "") or ""), merge)
    ]

    if not pred_eval_idx:
        stat_total["fn"] += 1
        _inc_stat_by_cls(stat_by_cls, gt_norm, "gt", 1)
        _inc_stat_by_cls(stat_by_cls, gt_norm, "fn", 1)
        return [], set(), preds

    for i in pred_eval_idx:
        _inc_stat_by_cls(stat_by_cls, gt_norm, "gt", 1)
        pred_cls = str(preds[i].get("cls_name", "") or "")
        pred_norm = normalize_class_name(
            pred_cls, merge, label_alias_map=label_alias_map
        )
        _inc_stat_by_cls(stat_by_cls, pred_norm, "pred", 1)
        if is_class_match(
            pred_cls,
            gt_name,
            merge,
            cls_merge_type_index,
            label_alias_map=label_alias_map,
            fuzzy_only_wildcard=fuzzy_only_wildcard,
            tier_equivalence=tier_equivalence,
        ):
            stat_total["tp"] += 1
            _inc_stat_by_cls(stat_by_cls, gt_norm, "tp", 1)
        else:
            stat_total["cls_err"] += 1
            stat_total["fp"] += 1
            _inc_stat_by_cls(stat_by_cls, gt_norm, "cls_err", 1)
            _inc_stat_by_cls(stat_by_cls, pred_norm, "fp", 1)

    return [], set(), preds


def _annotate_filename_gt_eval_tags(
    results_for_draw: list[dict[str, Any]],
    gt_name: str,
    *,
    class_merge: dict[str, list[str]] | None,
    cls_merge_type_index: Any = None,
    label_alias_map: dict[str, str] | None = None,
    fuzzy_only_wildcard: bool = False,
    tier_equivalence: ClassTierEquivalence | None = None,
) -> list[dict[str, Any]]:
    """文件名 GT：按类名给每个报出框打 tp / cls_err / fp（other）标签，无几何漏报框。"""
    gt_name = str(gt_name or "").strip()
    for r in results_for_draw:
        pred_cls = str(r.get("name", "") or "")
        if is_metric_ignored_other(pred_cls, class_merge):
            r["eval_tag"] = "fp"
            continue
        if is_class_match(
            pred_cls,
            gt_name,
            class_merge,
            cls_merge_type_index,
            label_alias_map=label_alias_map,
            fuzzy_only_wildcard=fuzzy_only_wildcard,
            tier_equivalence=tier_equivalence,
        ):
            r["eval_tag"] = "tp"
        else:
            r["eval_tag"] = "cls_err"
            r["eval_gt_name"] = gt_name
    return []


def _accumulate_predict_all_validation(
    stat_total: dict[str, int],
    stat_by_cls: dict[str, dict[str, int]],
    results: list[dict[str, Any]],
    gts: list[dict],
    *,
    class_merge: dict[str, list[str]] | None,
    geom_metric: str = "iou",
    geom_threshold: float = 0.5,
    cls_merge_type_index: Any = None,
    label_alias_map: dict[str, str] | None = None,
    filtered: list[dict[str, Any]] | None = None,
    fuzzy_only_wildcard: bool = False,
    tier_equivalence: ClassTierEquivalence | None = None,
) -> tuple[list[tuple[int, int, float]], set[int], list[dict[str, Any]]]:
    """
    与 predict_size_validate_lib 评估口径一致：bbox IoU/IoR 匹配 + 类别合并/cls_merge 父子等价。
    被过滤为 ``other`` 的实例仍参与与 GT 的几何匹配：匹配成功计类型错，避免 GT 误计漏报。
    匹配优先级：① 报出框（``results``）优于 filtered-other 仅几何匹配框；② 同类优于异类；③ 几何分高者优先。
    返回 (matches, matched_p, pred_rows) 供混淆矩阵与类型混淆切图使用。
    """
    preds, match_only_other, _filtered_other_indices = _build_eval_pred_pool(
        results, filtered
    )
    merge = class_merge

    def _pred_skip(r: dict, *, for_match_only_other: bool = False) -> bool:
        if for_match_only_other:
            return False
        return is_metric_ignored_other(str(r.get("cls_name", "") or ""), merge)

    def _gt_skip(g: dict) -> bool:
        return is_metric_ignored_other(str(g.get("name", "") or ""), merge)

    pred_eval_idx = [
        i
        for i, r in enumerate(preds)
        if match_only_other[i] or not _pred_skip(r)
    ]
    gt_eval_idx = [j for j, g in enumerate(gts) if not _gt_skip(g)]

    matches: list[tuple[int, int, float]] = []
    matched_p: set[int] = set()

    if not gts:
        fp_idx = [i for i in pred_eval_idx if not match_only_other[i]]
        stat_total["fp"] += len(fp_idx)
        for i in pred_eval_idx:
            if match_only_other[i]:
                continue
            pn = normalize_class_name(
                preds[i].get("cls_name", ""), merge, label_alias_map=label_alias_map
            )
            _inc_stat_by_cls(stat_by_cls, pn, "pred", 1)
        for i in fp_idx:
            pn = normalize_class_name(
                preds[i].get("cls_name", ""), merge, label_alias_map=label_alias_map
            )
            _inc_stat_by_cls(stat_by_cls, pn, "fp", 1)
        return matches, matched_p, preds

    if not gt_eval_idx:
        fp_idx = [i for i in pred_eval_idx if not match_only_other[i]]
        stat_total["fp"] += len(fp_idx)
        for i in pred_eval_idx:
            if match_only_other[i]:
                continue
            pn = normalize_class_name(
                preds[i].get("cls_name", ""), merge, label_alias_map=label_alias_map
            )
            _inc_stat_by_cls(stat_by_cls, pn, "pred", 1)
        for i in fp_idx:
            pn = normalize_class_name(
                preds[i].get("cls_name", ""), merge, label_alias_map=label_alias_map
            )
            _inc_stat_by_cls(stat_by_cls, pn, "fp", 1)
        return matches, matched_p, preds

    preds_eval = [preds[i] for i in pred_eval_idx]
    gts_eval = [gts[j] for j in gt_eval_idx]
    pred_tier_eval = [
        1 if match_only_other[pred_eval_idx[pi]] else 0 for pi in range(len(pred_eval_idx))
    ]
    prefer_class_match: list[list[bool]] = []
    for pi, p in enumerate(preds_eval):
        row: list[bool] = []
        pred_cls = str(p.get("cls_name", "") or "")
        for gj, g in enumerate(gts_eval):
            row.append(
                is_class_match(
                    pred_cls,
                    str(g.get("name", "") or ""),
                    merge,
                    cls_merge_type_index,
                    label_alias_map=label_alias_map,
                    fuzzy_only_wildcard=fuzzy_only_wildcard,
                    tier_equivalence=tier_equivalence,
                )
            )
        prefer_class_match.append(row)
    m = str(geom_metric or "iou").lower().strip()
    thr = float(geom_threshold)
    _match_kw = {
        "pred_tier": pred_tier_eval,
        "prefer_class_match": prefer_class_match,
    }
    if m == "ior":
        matches_ev, matched_p_ev, matched_g_ev = match_pred_gt_ior(
            preds_eval, gts_eval, thr, **_match_kw
        )
    else:
        matches_ev, matched_p_ev, matched_g_ev = match_pred_gt(
            preds_eval, gts_eval, thr, metric="iou", **_match_kw
        )

    matches = [(pred_eval_idx[pi], gt_eval_idx[gj], sc) for pi, gj, sc in matches_ev]
    matched_p = {pred_eval_idx[i] for i in matched_p_ev}
    matched_g = {gt_eval_idx[j] for j in matched_g_ev}
    pred_to_gt = {i: j for i, j, _ in matches}

    stat_total["geom_pairs"] += len(matches)

    for j in gt_eval_idx:
        _inc_stat_by_cls(
            stat_by_cls,
            normalize_class_name(
                gts[j].get("name", ""), merge, label_alias_map=label_alias_map
            ),
            "gt",
            1,
        )
    for i in pred_eval_idx:
        if match_only_other[i]:
            continue
        _inc_stat_by_cls(
            stat_by_cls,
            normalize_class_name(
                preds[i].get("cls_name", ""), merge, label_alias_map=label_alias_map
            ),
            "pred",
            1,
        )

    for i in range(len(preds)):
        if match_only_other[i]:
            if i not in matched_p:
                continue
        elif _pred_skip(preds[i]):
            continue
        if i not in matched_p:
            stat_total["fp"] += 1
            _inc_stat_by_cls(
                stat_by_cls,
                normalize_class_name(
                    preds[i].get("cls_name", ""), merge, label_alias_map=label_alias_map
                ),
                "fp",
                1,
            )
        else:
            j = pred_to_gt[i]
            pred_cls = preds[i].get("cls_name", "")
            gt_name = gts[j].get("name", "")
            gt_norm = normalize_class_name(
                gt_name, merge, label_alias_map=label_alias_map
            )
            if is_class_match(
                pred_cls,
                gt_name,
                merge,
                cls_merge_type_index,
                label_alias_map=label_alias_map,
                fuzzy_only_wildcard=fuzzy_only_wildcard,
                tier_equivalence=tier_equivalence,
            ):
                stat_total["tp"] += 1
                _inc_stat_by_cls(stat_by_cls, gt_norm, "tp", 1)
            else:
                stat_total["cls_err"] += 1
                stat_total["fp"] += 1
                _inc_stat_by_cls(stat_by_cls, gt_norm, "cls_err", 1)
                _inc_stat_by_cls(
                    stat_by_cls,
                    normalize_class_name(
                        pred_cls, merge, label_alias_map=label_alias_map
                    ),
                    "fp",
                    1,
                )

    stat_total["fn"] += len(gt_eval_idx) - len(matched_g_ev)
    for j in gt_eval_idx:
        if j not in matched_g:
            _inc_stat_by_cls(
                stat_by_cls,
                normalize_class_name(
                    gts[j].get("name", ""), merge, label_alias_map=label_alias_map
                ),
                "fn",
                1,
            )

    return matches, matched_p, preds


def _annotate_validation_eval_tags(
    results_for_draw: list[dict[str, Any]],
    filtered_for_draw: list[dict[str, Any]] | None,
    *,
    gts: list[dict],
    matches: list[tuple[int, int, float]] | None,
    matched_p: set[int] | None,
    n_result_preds: int,
    filtered_other_indices: list[int],
    class_merge: dict[str, list[str]] | None,
    cls_merge_type_index: Any = None,
    label_alias_map: dict[str, str] | None = None,
    fuzzy_only_wildcard: bool = False,
    tier_equivalence: ClassTierEquivalence | None = None,
) -> list[dict[str, Any]] | None:
    """为 results / filtered-other 写入 eval_tag，并返回未匹配的漏报 GT 框列表。"""
    pred_to_gt = {int(pi): int(gj) for pi, gj, _sc in (matches or [])}
    mset = set(matched_p or set())
    matched_g = {int(gj) for _pi, gj, _sc in (matches or [])}

    for pi in range(n_result_preds):
        if pi not in mset:
            results_for_draw[pi]["eval_tag"] = "fp"
            continue
        gj = pred_to_gt.get(pi)
        if gj is None or gj < 0 or gj >= len(gts):
            results_for_draw[pi]["eval_tag"] = "fp"
            continue
        pred_cls = str(results_for_draw[pi].get("name", "") or "")
        gt_name = str(gts[gj].get("name", "") or "")
        if is_class_match(
            pred_cls,
            gt_name,
            class_merge,
            cls_merge_type_index,
            label_alias_map=label_alias_map,
            fuzzy_only_wildcard=fuzzy_only_wildcard,
            tier_equivalence=tier_equivalence,
        ):
            results_for_draw[pi]["eval_tag"] = "tp"
        else:
            results_for_draw[pi]["eval_tag"] = "cls_err"
            results_for_draw[pi]["eval_gt_name"] = gt_name

    if filtered_for_draw is not None:
        for oi, fi in enumerate(filtered_other_indices):
            pi = n_result_preds + oi
            if pi not in mset:
                continue
            gj = pred_to_gt.get(pi)
            if gj is None or gj < 0 or gj >= len(gts):
                continue
            gt_name = str(gts[gj].get("name", "") or "")
            filtered_for_draw[fi]["eval_tag"] = "cls_err"
            filtered_for_draw[fi]["eval_gt_name"] = gt_name

    eval_boxes: list[dict[str, Any]] = []
    for gj in range(len(gts)):
        if gj in matched_g:
            continue
        g = gts[gj]
        eval_boxes.append(
            {
                "eval_tag": "fn",
                "name": str(g.get("name", "") or ""),
                "location": [
                    int(g.get("x1", 0)),
                    int(g.get("y1", 0)),
                    int(g.get("x2", 0)),
                    int(g.get("y2", 0)),
                ],
            }
        )
    return eval_boxes


def build_eval_class_merge(
    base: dict[str, list[str]] | None = None,
    *,
    insect_wildcard: bool = True,
) -> dict[str, list[str]] | None:
    """评估用类别合并表（默认叠加 insect: ['*']，口径同 predict_seg_validate）。"""
    if base is None and not insect_wildcard:
        return None
    out: dict[str, list[str]] = dict(base or {})
    if insect_wildcard:
        aliases = [str(a).strip() for a in (out.get("insect") or []) if str(a).strip()]
        if "*" not in aliases:
            aliases.append("*")
        out["insect"] = aliases
    return out or None


# --------------------------------------------------------------------------- #
#  Demo（绘图/保存等仅在此配置，不写进 config/insect_alg_all.json）
# --------------------------------------------------------------------------- #

# 来源配色（BGR）：尽量高对比、彼此区分，便于在密集虫情图上快速辨识。
_SOURCE_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "detect_daofeishi": (80, 200, 60),   # 稻飞虱：清新绿
    "detect_big": (255, 170, 40),        # 大虫分割：明亮天蓝
}
# 未命中来源配色表时的默认正常框：醒目橙红
_DEFAULT_BORDER_BGR = (40, 120, 245)
# 非 TOP1/TOP2 关注清单（推理类或标注类均不在内）的实例：灰黑框弱化展示
_OTHER_BORDER_BGR = (120, 120, 120)
_OTHER_LABEL_FG_BGR = (205, 205, 205)
# 被过滤实例：更暗的灰，弱化存在感
_FILTERED_BORDER_BGR = (80, 80, 80)
_LABEL_BG_BGR = (25, 25, 25)
_LABEL_FG_BGR = (255, 255, 255)
_FILTERED_LABEL_FG_BGR = (210, 210, 210)

# 评估模式（VOC xml 对照）配色：TP/类型错/多报/漏报
_EVAL_TP_BGR = (60, 200, 60)        # green
_EVAL_CLS_ERR_BGR = (60, 60, 230)   # red-ish
_EVAL_FP_BGR = (40, 170, 255)       # orange-ish
_EVAL_FN_BGR = (255, 140, 40)       # blue-ish (GT missed)


def build_draw_focus_set(
    *,
    alg_config: dict[str, Any] | None = None,
    root_ids: list[str] | None = None,
    validation_focus: ValidationFocusConfig | None = None,
    merge: dict[str, list[str]] | None = None,
    label_alias_map: dict[str, str] | None = None,
) -> frozenset[str]:
    """绘图/报出类关注集合（``report_classes``，含 normalize 后的 canonical 名）。"""
    if validation_focus is None:
        validation_focus = resolve_validation_focus_config(
            alg_config, root_ids=root_ids
        )
    return build_eval_focus_set(
        validation_focus.report_classes,
        merge=merge,
        label_alias_map=label_alias_map,
    )


def _class_name_in_draw_focus(
    raw: str | None,
    focus: frozenset[str],
    *,
    merge: dict[str, list[str]] | None = None,
    label_alias_map: dict[str, str] | None = None,
) -> bool:
    if not focus:
        return True
    s = str(raw or "").strip()
    if not s:
        return False
    if s in focus:
        return True
    canon = normalize_class_name(s, merge, label_alias_map=label_alias_map)
    return bool(canon) and canon in focus


def _result_in_draw_focus(
    r: dict[str, Any],
    focus: frozenset[str],
    *,
    merge: dict[str, list[str]] | None = None,
    label_alias_map: dict[str, str] | None = None,
) -> bool:
    """推理检出类型或标注类型任一在关注清单内即视为关注实例。"""
    pred = str(r.get("name") or "").strip()
    gt = str(r.get("eval_gt_name") or "").strip()
    return _class_name_in_draw_focus(
        pred, focus, merge=merge, label_alias_map=label_alias_map
    ) or _class_name_in_draw_focus(
        gt, focus, merge=merge, label_alias_map=label_alias_map
    )
def _resolve_cjk_font(font_size: int):
    # 字体查找统一走 predict_size_validate_lib.resolve_cjk_font_path
    # （环境变量 INSECT_CJK_FONT → 跨平台候选路径 → fc-match 兜底，解析结果缓存）。
    if ImageFont is None:
        return None
    fp = resolve_cjk_font_path()
    if fp:
        try:
            return ImageFont.truetype(fp, int(font_size))
        except Exception:
            logging.warning("加载中文字体失败: %s", fp, exc_info=True)
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def _measure_label_text(text: str, font_size: int) -> tuple[int, int]:
    """返回 (宽, 高) 像素。"""
    if Image is not None and ImageDraw is not None:
        font = _resolve_cjk_font(font_size)
        if font is not None:
            dummy = Image.new("RGB", (4, 4))
            draw = ImageDraw.Draw(dummy)
            bbox = draw.textbbox((0, 0), text, font=font)
            return int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = max(0.5, float(font_size) / 28.0)
    (tw, th), bl = cv2.getTextSize(text, font, fs, 2)
    return int(tw), int(th + bl)


def _label_display_name(
    r: dict[str, Any], *, cn_index: dict[str, str] | None = None
) -> str:
    """绘图/XML 中文类名：优先 ``insect_alg_all`` 构建的 ``cn_index``，再 ``viz_name`` / 行内 ``cn_name``。"""
    if cn_index:
        for key in (
            str(r.get("infer_name") or "").strip(),
            str(r.get("name") or "").strip(),
        ):
            if key:
                hit = cn_index.get(key)
                if hit:
                    return hit
    viz = str(r.get("viz_name") or "").strip()
    if viz:
        if cn_index:
            hit = cn_index.get(viz)
            if hit:
                return hit
            resolved = _resolve_cn_display_for_class(viz, cn_index=cn_index)
            if resolved and resolved != viz:
                return resolved
        return viz
    cn = str(r.get("cn_name") or "").strip()
    if cn:
        return cn
    name = str(r.get("name") or "unknown").strip()
    if cn_index:
        return _resolve_cn_display_for_class(name, cn_index=cn_index)
    return name


def _format_result_label(
    r: dict[str, Any], *, show_source: bool, cn_index: dict[str, str] | None = None
) -> str:
    display = _label_display_name(r, cn_index=cn_index)
    det_conf = float(r.get("det_conf", r.get("conf", 0.0)) or 0.0)
    cls_conf = _effective_cls_conf_for_display(r)

    dist_part = ""
    if bool(r.get("edge_rule_enabled", False)):
        dm = r.get("edge_min_dist", None)
        if dm is not None:
            try:
                dist_part = f" d:{int(round(float(dm)))}"
            except (TypeError, ValueError):
                dist_part = ""
    in_big_part = _format_in_big_label_suffix(r)
    if not bool(r.get("dia_enabled", True)) and dist_part:
        # dia_w 未命中时，edge_min_dist 仍可展示；这里不做额外处理，仅占位便于未来扩展
        pass
    if show_source:
        src = str(r.get("source") or "").strip()
        if src:
            return (
                f"{display} {det_conf:.2f}/{cls_conf:.2f}{dist_part}{in_big_part}"
                f"{format_clip_slice_label_suffix(r)} [{src}]"
            )
    return (
        f"{display} {det_conf:.2f}/{cls_conf:.2f}{dist_part}{in_big_part}"
        f"{format_clip_slice_label_suffix(r)}"
    )


def _result_bbox_diag_px(r: dict[str, Any]) -> float | None:
    """从 predict/filtered 统一结果行读取外接框对角线（像素）。"""
    try:
        loc = r.get("location")
        if isinstance(loc, (list, tuple)) and len(loc) >= 4:
            x1, y1, x2, y2 = (int(loc[0]), int(loc[1]), int(loc[2]), int(loc[3]))
            return bbox_diag_px(x1, y1, x2, y2)
        return bbox_diag_px_from_row(r)
    except (TypeError, ValueError, KeyError):
        return None


def _is_dia_threshold_filter_reason(reason: str, *, dia_enabled: bool = True) -> bool:
    r = str(reason or "").strip().lower()
    if r in ("threshold_dia", "dia"):
        return True
    return r == "size" and dia_enabled


def _humanize_other_filter_reason(
    reason: str,
    *,
    cn_index: dict[str, str] | None = None,
    row: dict[str, Any] | None = None,
) -> str:
    """将 ``other_cls_*`` 过滤原因转为可读中文（用于绘图标签）。"""
    r = str(reason or "").strip()
    if not r:
        return ""
    if r == "other_cls_crop":
        return "分类裁剪为空"
    if r == "other_cls_none":
        return "分类器无结果"
    if r.startswith("other_cls_conf:"):
        parsed = _parse_other_cls_conf_reason(r)
        if parsed is not None:
            nm, conf_t1, thr = parsed
            disp = _resolve_cn_display_for_class(nm, cn_index=cn_index)
            return f"分类低置信→其他 top1={disp}({conf_t1:.2f}<={thr:.2f})"
        top1_nm = str((row or {}).get("cls_name_top1") or "").strip()
        conf_t1 = _extract_top1_cls_conf(row or {})
        thr = (row or {}).get("cls_conf_threshold")
        if top1_nm and not _is_other_class_name(top1_nm) and conf_t1 is not None:
            disp = _resolve_cn_display_for_class(top1_nm, cn_index=cn_index)
            thr_part = ""
            if thr is not None:
                try:
                    thr_part = f"<={float(thr):.2f}"
                except (TypeError, ValueError):
                    pass
            return f"分类低置信→其他 top1={disp}({conf_t1:.2f}{thr_part})"
        body = r[len("other_cls_conf:") :]
        if "(" in body and body.endswith(")"):
            nm, expr = body.split("(", 1)
            nm = nm.strip()
            disp = _resolve_cn_display_for_class(nm, cn_index=cn_index)
            return f"分类低置信→其他 top1={disp}({expr[:-1]})"
        return f"分类低置信→其他 {body}"
    if r == "other":
        top1 = _extract_top1_cls_name(row or {})
        if top1:
            disp = _resolve_cn_display_for_class(top1, cn_index=cn_index)
            return f"后置其他 top1={disp}"
        return "后置其他(未记录原因)"
    if r == "report_tier":
        nm = str(
            (row or {}).get("name")
            or (row or {}).get("cls_name_top1")
            or (row or {}).get("cls_name")
            or ""
        ).strip()
        if nm:
            disp = _resolve_cn_display_for_class(nm, cn_index=cn_index)
            return f"非报出层级({disp})"
        return "非报出层级"
    if r == "roi_edge":
        mode = str((row or {}).get("roi_edge_mode") or "").strip()
        edge_frac = (row or {}).get("roi_edge_frac")
        if mode == "edge_hug" and edge_frac is not None:
            return f"圆盘边缘贴边误报(edge={float(edge_frac):.2f})"
        if edge_frac is not None:
            return f"圆盘边缘跨界误报(edge={float(edge_frac):.2f})"
        return "圆盘边缘误报"
    return r


def _format_filter_reason_for_label(
    r: dict[str, Any],
    *,
    cn_index: dict[str, str] | None = None,
) -> str:
    """绘图用过滤原因；dia 阈值过滤时在原因中的 dia 后附上实测对角线像素。"""
    reason = str(
        r.get("filter_reason") or r.get("cls_other_reason") or ""
    ).strip()
    if not reason:
        return ""
    if reason == "report_tier" or _is_other_filter_reason(reason):
        return _humanize_other_filter_reason(reason, cn_index=cn_index, row=r)
    if reason == "roi_edge":
        return _humanize_other_filter_reason("roi_edge", cn_index=cn_index, row=r)
    if not _is_dia_threshold_filter_reason(
        reason, dia_enabled=bool(r.get("dia_enabled", True))
    ):
        return reason
    d = _result_bbox_diag_px(r)
    if d is None:
        return reason
    d_str = str(int(round(d)))
    r_low = reason.lower()
    if r_low == "threshold_dia":
        return f"threshold_dia:{d_str}"
    if r_low == "dia":
        return f"dia:{d_str}"
    if r_low == "size":
        return f"size dia:{d_str}"
    return reason


def _append_filter_reason_to_label(
    label: str,
    r: dict[str, Any],
    *,
    cn_index: dict[str, str] | None = None,
) -> str:
    """在标签末尾附加过滤原因（filtered 或对外 other 实例）。"""
    if not (r.get("filtered") or _is_other_class_name(str(r.get("name") or ""))):
        return label
    reason = _format_filter_reason_for_label(r, cn_index=cn_index)
    if reason:
        return f"{label} 因:{reason}"
    return label


def _format_filtered_label(
    r: dict[str, Any], *, show_source: bool, cn_index: dict[str, str] | None = None
) -> str:
    base = _format_result_label(r, show_source=show_source, cn_index=cn_index)
    reason = _format_filter_reason_for_label(r, cn_index=cn_index)
    if reason:
        return f"{base} 过滤:{reason}"
    return f"{base} [过滤]"


def _format_cls_topn_gt_match_suffix(
    r: dict[str, Any],
    gt_name: str,
    *,
    cn_index: dict[str, str] | None = None,
    class_merge: dict[str, list[str]] | None = None,
    cls_merge_type_index: Any = None,
    label_alias_map: dict[str, str] | None = None,
    fuzzy_only_wildcard: bool = False,
    tier_equivalence: ClassTierEquivalence | None = None,
) -> str:
    """
    类型错时，若 top2..topN 中含与 GT 匹配的候选，追加到标签供排查。

    ``cls_top_n<=1`` 或无匹配候选时返回空串。
    """
    if int(r.get("cls_top_n", 1) or 1) <= 1:
        return ""
    entries = _sorted_cls_topk_entries(r)
    if len(entries) < 2:
        return ""
    parts: list[str] = []
    for rank, item in enumerate(entries[1:], start=2):
        cls_nm = _cls_topk_item_name(item)
        if not cls_nm:
            continue
        if not is_class_match(
            cls_nm,
            gt_name,
            class_merge,
            cls_merge_type_index,
            label_alias_map=label_alias_map,
            fuzzy_only_wildcard=fuzzy_only_wildcard,
            tier_equivalence=tier_equivalence,
        ):
            continue
        conf = _cls_topk_item_conf(item)
        disp = _resolve_cn_display_for_class(cls_nm, cn_index=cn_index)
        parts.append(f"top{rank}:{disp}({conf:.2f})")
    if not parts:
        return ""
    return " " + " ".join(parts)


def _format_eval_tag_label(
    r: dict[str, Any],
    eval_tag: str,
    *,
    show_source: bool,
    cn_index: dict[str, str] | None = None,
    class_merge: dict[str, list[str]] | None = None,
    cls_merge_type_index: Any = None,
    label_alias_map: dict[str, str] | None = None,
    fuzzy_only_wildcard: bool = False,
    tier_equivalence: ClassTierEquivalence | None = None,
) -> str:
    """评估模式标签；类型错时在预测类名后附加 GT 正确类名。"""
    base = _format_result_label(r, show_source=show_source, cn_index=cn_index)
    tag = str(eval_tag or "").strip().upper()
    t = str(eval_tag or "").strip().lower()
    if t in ("cls_err", "class_error", "type_error"):
        gt_raw = str(r.get("eval_gt_name") or "").strip()
        if gt_raw:
            gt_disp = _resolve_cn_display_for_class(gt_raw, cn_index=cn_index)
            pred_disp = _label_display_name(r, cn_index=cn_index)
            rest = base[len(pred_disp) :].strip() if pred_disp else base
            topn_suffix = _format_cls_topn_gt_match_suffix(
                r,
                gt_raw,
                cn_index=cn_index,
                class_merge=class_merge,
                cls_merge_type_index=cls_merge_type_index,
                label_alias_map=label_alias_map,
                fuzzy_only_wildcard=fuzzy_only_wildcard,
                tier_equivalence=tier_equivalence,
            )
            if rest:
                out = f"{tag} {pred_disp} GT:{gt_disp} {rest}{topn_suffix}".strip()
            else:
                out = f"{tag} {pred_disp} GT:{gt_disp}{topn_suffix}".strip()
            return _append_filter_reason_to_label(out, r, cn_index=cn_index)
    out = f"{tag} {base}".strip()
    return _append_filter_reason_to_label(out, r, cn_index=cn_index)


def _label_box_metrics(text: str, font_size: int) -> tuple[int, int, int]:
    """返回标签背景框 (宽, 高, 内边距)。"""
    pad = max(6, font_size // 5)
    tw, th = _measure_label_text(text, font_size)
    return tw + pad * 2, th + pad * 2, pad


def _rects_overlap(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    *,
    margin: int = 3,
) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    m = int(margin)
    return not (ax2 + m <= bx1 or bx2 + m <= ax1 or ay2 + m <= by1 or by2 + m <= ay1)


def _clamp_label_origin(
    lx: int, ly: int, box_w: int, box_h: int, img_w: int, img_h: int
) -> tuple[int, int]:
    """将标签左上角坐标约束在图像内，保证整块标签可见。"""
    max_lx = max(0, int(img_w) - int(box_w))
    max_ly = max(0, int(img_h) - int(box_h))
    return max(0, min(int(lx), max_lx)), max(0, min(int(ly), max_ly))


class _LabelPlacer:
    """调试图标签布局：记录已占用区域，避免重叠并尽量保持在画面内。"""

    def __init__(self, img_w: int, img_h: int, *, gap: int = 4):
        self.img_w = int(img_w)
        self.img_h = int(img_h)
        self.gap = max(2, int(gap))
        self._occupied: list[tuple[int, int, int, int]] = []

    def _fits(self, lx: int, ly: int, box_w: int, box_h: int) -> bool:
        rect = (lx, ly, lx + box_w, ly + box_h)
        return not any(_rects_overlap(rect, occ) for occ in self._occupied)

    def place(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        box_w: int,
        box_h: int,
    ) -> tuple[int, int]:
        bx1, by1, bx2, by2 = int(x1), int(y1), int(x2), int(y2)
        if bx2 < bx1:
            bx1, bx2 = bx2, bx1
        if by2 < by1:
            by1, by2 = by2, by1
        g = self.gap

        candidates: list[tuple[int, int]] = [
            (bx1, by1 - box_h - g),
            (bx1, by2 + g),
            (bx2 - box_w, by1 - box_h - g),
            (bx2 - box_w, by2 + g),
            (bx1 + 2, by1 + 2),
            (bx1 + 2, by2 - box_h - 2),
            ((bx1 + bx2) // 2 - box_w // 2, by1 - box_h - g),
            ((bx1 + bx2) // 2 - box_w // 2, by2 + g),
            (bx2 - box_w - 2, by1 + 2),
        ]

        for lx, ly in candidates:
            lx, ly = _clamp_label_origin(lx, ly, box_w, box_h, self.img_w, self.img_h)
            if self._fits(lx, ly, box_w, box_h):
                self._occupied.append((lx, ly, lx + box_w, ly + box_h))
                return lx, ly

        # 兜底：从默认位置起逐步下移/右移，直到不重叠或耗尽步数
        lx, ly = _clamp_label_origin(bx1, by1 - box_h - g, box_w, box_h, self.img_w, self.img_h)
        for step in range(48):
            if self._fits(lx, ly, box_w, box_h):
                self._occupied.append((lx, ly, lx + box_w, ly + box_h))
                return lx, ly
            ly += box_h + g
            if ly + box_h > self.img_h:
                ly = max(0, by2 + g)
                lx += max(8, box_w // 3)
            lx, ly = _clamp_label_origin(lx, ly, box_w, box_h, self.img_w, self.img_h)
            if step >= 24 and step % 4 == 0:
                lx, ly = _clamp_label_origin(
                    bx1 + (step // 4) * 12, by1 + (step // 4) * (box_h + g),
                    box_w,
                    box_h,
                    self.img_w,
                    self.img_h,
                )

        self._occupied.append((lx, ly, lx + box_w, ly + box_h))
        return lx, ly


def _draw_label_above_box(
    img_bgr: np.ndarray,
    text: str,
    x1: int,
    y1: int,
    *,
    font_size: int,
    border_bgr: tuple[int, int, int],
    img_w: int,
    img_h: int,
    text_bgr: tuple[int, int, int] | None = None,
    x2: int | None = None,
    y2: int | None = None,
    placer: _LabelPlacer | None = None,
) -> np.ndarray:
    box_w, box_h, pad = _label_box_metrics(text, font_size)
    bx2 = int(x2 if x2 is not None else x1)
    by2 = int(y2 if y2 is not None else y1)
    if placer is not None:
        lx, ly = placer.place(int(x1), int(y1), bx2, by2, box_w, box_h)
    else:
        lx, ly = _clamp_label_origin(int(x1), int(y1) - box_h - 2, box_w, box_h, img_w, img_h)
        if ly + box_h > int(y1):
            lx, ly = _clamp_label_origin(int(x1), int(y1) + 2, box_w, box_h, img_w, img_h)
    lx2 = lx + box_w
    ly2 = ly + box_h
    cv2.rectangle(img_bgr, (lx, ly), (lx2, ly2), _LABEL_BG_BGR, thickness=-1)
    cv2.rectangle(img_bgr, (lx, ly), (lx2, ly2), border_bgr, thickness=2)
    text_x = lx + pad
    text_y = ly + pad
    return _draw_cn_text(
        img_bgr,
        text,
        (text_x, text_y),
        font_size=font_size,
        color_bgr=text_bgr if text_bgr is not None else _LABEL_FG_BGR,
    )


ALLOWED_OUTPUT_IMAGE_SCALES: tuple[float, ...] = (0.5, 0.25, 0.125)


def resolve_output_image_scale(scale: float | None) -> float | None:
    """
    解析可视化结果图保存缩放比例。

    - ``None`` / 未配置：不缩放（保持原图尺寸）
    - 允许 ``0.5``、``0.25``、``0.125``（分别为 1/2、1/4、1/8）
    """
    if scale is None:
        return None
    s = float(scale)
    if abs(s - 1.0) < 1e-9:
        return None
    for allowed in ALLOWED_OUTPUT_IMAGE_SCALES:
        if abs(s - allowed) < 1e-9:
            return allowed
    allowed_txt = ", ".join(str(x) for x in ALLOWED_OUTPUT_IMAGE_SCALES)
    raise ValueError(
        f"output_image_scale 仅支持 {allowed_txt}，或不配置保持原尺寸；收到: {scale!r}"
    )


def resize_bgr_for_output(image: np.ndarray, scale: float) -> np.ndarray:
    """按线性比例缩小 BGR 图（仅用于落盘前；绘制仍在原图上完成）。"""
    if scale is None or scale >= 1.0 - 1e-9:
        return image
    h, w = image.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    if new_w == w and new_h == h:
        return image
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image, (new_w, new_h), interpolation=interp)


def _draw_result_polygon(
    img_draw: np.ndarray,
    polygon: Any,
    color: tuple[int, int, int],
    *,
    line_thk: int,
) -> np.ndarray:
    """在图上绘制分割多边形轮廓（仅边框，不填充 mask；与正常/过滤实例共用）。"""
    pts = np.asarray(polygon, dtype=np.int32)
    if len(pts) < 3:
        return img_draw
    cv2.polylines(img_draw, [pts], True, color, line_thk, lineType=cv2.LINE_AA)
    return img_draw


def _select_filtered_for_draw(
    filtered: list[dict[str, Any]] | None,
    *,
    draw_filter_sources: frozenset[str],
    draw_validation_focus: bool,
    report_focus: frozenset[str] | None = None,
    class_merge: dict[str, list[str]] | None = None,
    label_alias_map: dict[str, str] | None = None,
) -> list[dict[str, Any]] | None:
    """
    选择要绘制的 filtered 实例。

    - ``models.{root}.draw_filter=true``：绘制该根模型来源的全部 filtered（深灰 + 长标签）；
    - 否则且 ``draw_validation_focus=True``：仅绘制关注类
      （预测类名或 eval_gt_name 任一在关注清单内）。
    """
    if not filtered:
        return None
    picked: list[dict[str, Any]] = []
    for r in filtered:
        src = str(r.get("source") or "").strip()
        if src in draw_filter_sources:
            picked.append(r)
            continue
        if not draw_validation_focus:
            continue
        if _result_in_draw_focus(
            r,
            report_focus or build_draw_focus_set(
                merge=class_merge, label_alias_map=label_alias_map
            ),
            merge=class_merge,
            label_alias_map=label_alias_map,
        ):
            picked.append(r)
    return picked or None


def _is_eval_tp_tag(eval_tag: str) -> bool:
    t = str(eval_tag or "").strip().lower()
    return t in ("tp", "true_positive")


def _should_draw_result_label(
    eval_tag: str,
    *,
    draw_label: bool,
    draw_label_errors_only: bool,
) -> bool:
    """``draw_label_errors_only=True`` 时 TP 仍画框但不画标签，便于定位 cls_err/fp/fn。"""
    if not draw_label:
        return False
    if not draw_label_errors_only:
        return True
    if not str(eval_tag or "").strip():
        return True
    return not _is_eval_tp_tag(eval_tag)


def _format_draw_result_label(
    r: dict[str, Any],
    *,
    eval_tag: str,
    show_source: bool,
    cn_index: dict[str, str] | None = None,
    draw_label_errors_only: bool = False,
    class_merge: dict[str, list[str]] | None = None,
    cls_merge_type_index: Any = None,
    label_alias_map: dict[str, str] | None = None,
    fuzzy_only_wildcard: bool = False,
    tier_equivalence: ClassTierEquivalence | None = None,
) -> str:
    """
    校验可视化标签文案。

    ``draw_label_errors_only=False`` 时 TP 仅绘中文类名；cls_err/fp/fn 等仍用完整评估标签。
    无 ``eval_tag`` 时回退推理标签（含置信度等）。
    """
    tag = str(eval_tag or "").strip()
    if tag and not draw_label_errors_only and _is_eval_tp_tag(tag):
        return _label_display_name(r, cn_index=cn_index)
    if tag:
        return _format_eval_tag_label(
            r,
            tag,
            show_source=show_source,
            cn_index=cn_index,
            class_merge=class_merge,
            cls_merge_type_index=cls_merge_type_index,
            label_alias_map=label_alias_map,
            fuzzy_only_wildcard=fuzzy_only_wildcard,
            tier_equivalence=tier_equivalence,
        )
    return _format_result_label(
        r, show_source=show_source, cn_index=cn_index
    )


def _skip_non_focus_eval_clutter_draw(
    r: dict[str, Any],
    *,
    in_focus: bool,
    draw_eval_fp_non_focus: bool,
) -> bool:
    """
    draw_eval_fp_non_focus=False 时，仅绘制 TOP1+TOP2 关注类相关框
    （``in_focus`` 由 ``_result_in_draw_focus`` 判定：预测类或 eval_gt_name 在关注清单内）。

    不再因任意 ``eval_gt_name`` / ``cls_err`` 即视为「有标注关联」而绘制
    （否则 GT 为粗类「昆虫」的非关注物种类型错会被误画出来）。
    """
    if draw_eval_fp_non_focus:
        return False
    return not in_focus


def draw_results(
    image: np.ndarray,
    results: list[dict[str, Any]],
    output_path: str | None = None,
    *,
    draw_bbox: bool = True,
    draw_polygon: bool = False,
    draw_label: bool = True,
    label_font_size: int | None = None,
    show_source_in_label: bool = False,
    eval_boxes: list[dict[str, Any]] | None = None,
    filtered_results: list[dict[str, Any]] | None = None,
    draw_filter: bool = False,
    cn_index: dict[str, str] | None = None,
    draw_focus: frozenset[str] | None = None,
    class_merge: dict[str, list[str]] | None = None,
    label_alias_map: dict[str, str] | None = None,
    output_image_scale: float | None = None,
    draw_eval_fp_non_focus: bool = True,
    draw_label_errors_only: bool = False,
    cls_merge_type_index: Any = None,
    fuzzy_only_wildcard: bool = False,
    tier_equivalence: ClassTierEquivalence | None = None,
) -> np.ndarray:
    """
    将 predict 统一结果绘制到图上。

    标签使用 PIL + 系统中文字体（OpenCV putText 无法显示中文）。
    ``draw_filter=True`` 时先绘制被过滤实例（深灰配色 + 完整过滤标签；``draw_polygon=True`` 时与保留实例同样绘制多边形轮廓）。
    调用方通过 ``filtered_results`` 传入待绘制的 filtered 子集（可为 TOP1+TOP2 关注类子集）。
    ``draw_eval_fp_non_focus=False`` 时，仅绘制 TOP1+TOP2 关注类相关框（预测类或 ``eval_gt_name`` 在关注清单内）；
    非关注类的 FP、FN、类型错（如 GT 为粗类「昆虫」）不绘制。
    ``draw_label=False`` 时不绘制框旁文字标签，仍保留 bbox / polygon 轮廓。
    ``draw_label_errors_only=False`` 时校验 TP 标签仅绘中文类名；错误框仍保留完整评估标签。
    ``draw_label_errors_only=True`` 时校验 TP 仍画框但不画标签，仅 cls_err/fp/fn 等保留文字标签。
    ``output_image_scale`` 仅在 ``output_path`` 落盘前缩小整图（绘制仍在原分辨率完成）。
    """
    save_scale = resolve_output_image_scale(output_image_scale)
    img_draw = image.copy()
    h_img, w_img = img_draw.shape[:2]
    params = _auto_draw_params(w_img, h_img)
    rect_thk = int(params["rect_thk"])
    filter_thk = max(1, rect_thk - 1)
    font_px = int(label_font_size or params["cap_font_size"])
    font_px = max(20, min(56, font_px))
    label_placer = _LabelPlacer(w_img, h_img)
    label_match_kw = {
        "class_merge": class_merge,
        "cls_merge_type_index": cls_merge_type_index,
        "label_alias_map": label_alias_map,
        "fuzzy_only_wildcard": fuzzy_only_wildcard,
        "tier_equivalence": tier_equivalence,
    }
    focus_set = (
        draw_focus
        if draw_focus is not None
        else build_draw_focus_set(merge=class_merge, label_alias_map=label_alias_map)
    )

    def _eval_color(tag: str) -> tuple[int, int, int] | None:
        t = str(tag or "").strip().lower()
        if t in ("tp", "true_positive"):
            return _EVAL_TP_BGR
        if t in ("cls_err", "class_error", "type_error"):
            return _EVAL_CLS_ERR_BGR
        if t in ("fp", "false_positive"):
            return _EVAL_FP_BGR
        if t in ("fn", "false_negative"):
            return _EVAL_FN_BGR
        return None

    # 评估专用：先画漏报 FN 的 GT 框（不依赖 source 配色）
    if eval_boxes and draw_bbox:
        for b in eval_boxes:
            loc = b.get("location") or b.get("bbox") or [0, 0, 0, 0]
            x1, y1, x2, y2 = [int(v) for v in loc]
            eval_color = _eval_color(str(b.get("eval_tag") or ""))
            if eval_color is None:
                continue
            gt_name = str(b.get("name") or "")
            in_focus = _class_name_in_draw_focus(
                gt_name,
                focus_set,
                merge=class_merge,
                label_alias_map=label_alias_map,
            )
            if _skip_non_focus_eval_clutter_draw(
                b,
                in_focus=in_focus,
                draw_eval_fp_non_focus=draw_eval_fp_non_focus,
            ):
                continue
            if in_focus:
                color = eval_color
                text_bgr = None
            else:
                color = _OTHER_BORDER_BGR
                text_bgr = _OTHER_LABEL_FG_BGR
            label = (
                f"{str(b.get('eval_tag')).upper()}:{gt_name}"
                if gt_name
                else str(b.get("eval_tag")).upper()
            )
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, rect_thk, lineType=cv2.LINE_AA)
            if _should_draw_result_label(
                str(b.get("eval_tag") or ""),
                draw_label=draw_label,
                draw_label_errors_only=draw_label_errors_only,
            ):
                img_draw = _draw_label_above_box(
                    img_draw,
                    label,
                    x1,
                    y1,
                    font_size=font_px,
                    border_bgr=color,
                    img_w=w_img,
                    img_h=h_img,
                    text_bgr=text_bgr,
                    x2=x2,
                    y2=y2,
                    placer=label_placer,
                )

    if draw_filter and filtered_results:
        for r in filtered_results:
            x1, y1, x2, y2 = [int(v) for v in r["location"]]
            eval_tag = str(r.get("eval_tag") or "").strip()
            eval_color = _eval_color(eval_tag) if eval_tag else None
            in_focus = _result_in_draw_focus(
                r,
                focus_set,
                merge=class_merge,
                label_alias_map=label_alias_map,
            )
            if _skip_non_focus_eval_clutter_draw(
                r,
                in_focus=in_focus,
                draw_eval_fp_non_focus=draw_eval_fp_non_focus,
            ):
                continue
            if eval_color is not None and in_focus:
                color = eval_color
                text_bgr = None
                label = _format_draw_result_label(
                    r,
                    eval_tag=eval_tag,
                    show_source=show_source_in_label,
                    cn_index=cn_index,
                    draw_label_errors_only=draw_label_errors_only,
                    **label_match_kw,
                )
            else:
                color = _FILTERED_BORDER_BGR
                text_bgr = _FILTERED_LABEL_FG_BGR
                label = _format_filtered_label(
                    r, show_source=show_source_in_label, cn_index=cn_index
                )
            if draw_polygon and r.get("polygon"):
                img_draw = _draw_result_polygon(
                    img_draw,
                    r["polygon"],
                    color,
                    line_thk=filter_thk,
                )
            if draw_bbox:
                cv2.rectangle(
                    img_draw, (x1, y1), (x2, y2), color, filter_thk, lineType=cv2.LINE_AA
                )
                if _should_draw_result_label(
                    eval_tag,
                    draw_label=draw_label,
                    draw_label_errors_only=draw_label_errors_only,
                ):
                    img_draw = _draw_label_above_box(
                        img_draw,
                        label,
                        x1,
                        y1,
                        font_size=font_px,
                        border_bgr=color,
                        img_w=w_img,
                        img_h=h_img,
                        text_bgr=text_bgr if text_bgr is not None else _FILTERED_LABEL_FG_BGR,
                        x2=x2,
                        y2=y2,
                        placer=label_placer,
                    )

    # 非关注实例置底层先画，关注实例后画避免被遮挡；稳定排序保持组内原有相对顺序。
    ordered_results = sorted(
        results,
        key=lambda rr: (
            0
            if not _result_in_draw_focus(
                rr,
                focus_set,
                merge=class_merge,
                label_alias_map=label_alias_map,
            )
            else 1
        ),
    )
    for r in ordered_results:
        x1, y1, x2, y2 = [int(v) for v in r["location"]]
        source = str(r.get("source", "") or "")
        in_focus = _result_in_draw_focus(
            r,
            focus_set,
            merge=class_merge,
            label_alias_map=label_alias_map,
        )
        eval_tag = str(r.get("eval_tag") or "").strip()
        if _skip_non_focus_eval_clutter_draw(
            r,
            in_focus=in_focus,
            draw_eval_fp_non_focus=draw_eval_fp_non_focus,
        ):
            continue
        eval_color = _eval_color(eval_tag) if eval_tag else None
        if eval_color is not None and in_focus:
            color = eval_color
            text_bgr = None
            label = _format_draw_result_label(
                r,
                eval_tag=eval_tag,
                show_source=show_source_in_label,
                cn_index=cn_index,
                draw_label_errors_only=draw_label_errors_only,
                **label_match_kw,
            )
        elif not in_focus:
            color = _OTHER_BORDER_BGR
            text_bgr = _OTHER_LABEL_FG_BGR
            if eval_tag:
                label = _format_draw_result_label(
                    r,
                    eval_tag=eval_tag,
                    show_source=show_source_in_label,
                    cn_index=cn_index,
                    draw_label_errors_only=draw_label_errors_only,
                    **label_match_kw,
                )
            else:
                label = _format_result_label(
                    r, show_source=show_source_in_label, cn_index=cn_index
                )
        else:
            color = _SOURCE_COLORS_BGR.get(source, _DEFAULT_BORDER_BGR)
            text_bgr = None
            label = _format_result_label(
                r, show_source=show_source_in_label, cn_index=cn_index
            )

        if draw_polygon and r.get("polygon"):
            img_draw = _draw_result_polygon(
                img_draw,
                r["polygon"],
                color,
                line_thk=rect_thk,
            )

        if draw_bbox:
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, rect_thk, lineType=cv2.LINE_AA)
            if _should_draw_result_label(
                eval_tag,
                draw_label=draw_label,
                draw_label_errors_only=draw_label_errors_only,
            ):
                img_draw = _draw_label_above_box(
                    img_draw,
                    label,
                    x1,
                    y1,
                    font_size=font_px,
                    border_bgr=color,
                    img_w=w_img,
                    img_h=h_img,
                    text_bgr=text_bgr,
                    x2=x2,
                    y2=y2,
                    placer=label_placer,
                )

    if output_path:
        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        img_to_save = (
            resize_bgr_for_output(img_draw, save_scale)
            if save_scale is not None
            else img_draw
        )
        cv2.imwrite(output_path, img_to_save)
    return img_draw


# --------------------------------------------------------------------------- #
#  调测：将统一结果的分类框图上报 LS 标注系统（默认关闭，复用 ls_seg_classification_ingest）
#  - 关闭时下列函数不会被调用，主流程与改前完全一致。
#  - 开启时仅做「统一结果 → ingest row」适配，再交给既有上报实现。
# --------------------------------------------------------------------------- #


def _result_to_ls_ingest_row(r: dict[str, Any]) -> dict[str, Any]:
    """
    将 ``predict`` 统一结果转为 ``ls_seg_classification_ingest`` 可消费的 row。

    字段映射见 02-dr《统一管线分类框图上报LS调测开关》；检测来源无 polygon 时，
    以外接框四角兜底，满足上报路径对 ``len(polygon) >= 3`` 的最小要求（裁剪仍按 bbox）。
    """
    loc = r.get("location") or [0, 0, 0, 0]
    x1, y1, x2, y2 = int(loc[0]), int(loc[1]), int(loc[2]), int(loc[3])
    det_conf = float(r.get("det_conf", r.get("score", 0.0)) or 0.0)
    cls_conf_raw = r.get("cls_conf", None)
    if cls_conf_raw is None:
        cls_conf = float(r.get("score", det_conf) or det_conf)
    else:
        try:
            cls_conf = float(cls_conf_raw)
        except (TypeError, ValueError):
            cls_conf = float(r.get("score", det_conf) or det_conf)
    name = str(r.get("name") or "unknown")
    row: dict[str, Any] = {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "cls_name": name,
        "cls_conf": cls_conf,
        "seg_cls_name": name,
        "seg_conf": det_conf,
    }
    if r.get("filtered") or r.get("filter"):
        row["filter"] = True
    poly = r.get("polygon")
    if poly and len(poly) >= 3:
        row["polygon"] = poly
    else:
        row["polygon"] = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    return row


def build_ls_ingestor(
    *,
    ls_ingest_url: str,
    choice_from_name: str = "choice",
    choice_to_name: str = "image",
    ingest_batch_size: int = 200,
    jpeg_quality: int = 95,
):
    """构造 LS 图像分类上报器（复用 ls_seg_classification_ingest，按需导入）。"""
    from script.ls_seg_classification_ingest import (
        LsSegClassificationIngestor,
        resolve_ls_ingest_url,
    )

    return LsSegClassificationIngestor(
        ingest_url=resolve_ls_ingest_url(ingest_url=ls_ingest_url),
        choice_from_name=choice_from_name,
        choice_to_name=choice_to_name,
        ingest_batch_size=max(1, int(ingest_batch_size)),
        jpeg_quality=max(50, min(100, int(jpeg_quality))),
    )


def ingest_results_to_ls(
    ingestor: Any,
    results: list[dict[str, Any]],
    image_bgr: np.ndarray,
    *,
    source_image_name: str,
    skip_filtered: bool = True,
    skip_other: bool = True,
    ingest_crop_from_bbox: bool = True,
    pad_ratio: float = 0.05,
    pad_square: bool = False,
    post: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    将 ``predict_all`` 统一结果以「图像分类」任务上报 LS（复用 ingest_predict_seg_results）。

    :return: ``(tasks, responses)``；``post=False`` 时 responses 为空列表。
    """
    from script.ls_seg_classification_ingest import ingest_predict_seg_results

    rows = [_result_to_ls_ingest_row(r) for r in results]
    return ingest_predict_seg_results(
        ingestor,
        rows,
        image_bgr,
        source_image_name=source_image_name,
        skip_filtered=skip_filtered,
        skip_other=skip_other,
        ingest_crop_from_bbox=ingest_crop_from_bbox,
        pad_ratio=pad_ratio,
        pad_square=pad_square,
        post=post,
    )


# --------------------------------------------------------------------------- #
#  调测：将分类框图按类别分文件夹输出到本地目录（LS_INGEST_URL 填本地路径时启用）
#  - LS_INGEST_URL 以 http(s):// 开头 → 标注系统上报（行为不变，走上面的 ingest）。
#  - 否则视为本地目录 → 复用同一套裁剪逻辑，把分类框图落盘到 <目录>/<类别>/ 下。
# --------------------------------------------------------------------------- #


def _ls_target_is_local_dir(target: str) -> bool:
    """
    判定 ``LS_INGEST_URL`` 取值是「标注系统 URL」还是「本地输出目录」。

    - ``http://`` / ``https://``（大小写不敏感）开头 → 标注系统 URL（返回 False）。
    - 空字符串 → False（仍按 URL 模式，由 resolve_ls_ingest_url 报缺失，保持现状）。
    - 其它 → 本地目录（返回 True）。
    """
    t = str(target or "").strip()
    if not t:
        return False
    low = t.lower()
    return not (low.startswith("http://") or low.startswith("https://"))


def _pad_bgra_to_square_transparent(bgra: np.ndarray) -> np.ndarray:
    """将 BGRA 裁剪居中补成正方形，补边为透明（alpha=0）。"""
    h, w = bgra.shape[:2]
    if h == w:
        return bgra
    side = max(h, w)
    top = (side - h) // 2
    bottom = side - h - top
    left = (side - w) // 2
    right = side - w - left
    return cv2.copyMakeBorder(
        bgra, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0)
    )


def _crop_instance_bgra_from_polygon(
    image_bgr: np.ndarray,
    polygon: list[list[int]],
    *,
    pad_ratio: float = 0.05,
    pad_square: bool = False,
) -> np.ndarray | None:
    """
    按分割多边形裁剪实例并生成 BGRA 裁剪：多边形内为原始像素、多边形外 alpha=0（透明）。

    几何（外接框 + ``pad_ratio`` 外扩）与 ``crop_instance_bgr_from_polygon`` 保持一致，
    区别仅在于不填底色，而是把掩码写入 alpha 通道供 PNG 透明输出。
    """
    if image_bgr is None or image_bgr.size == 0:
        return None
    poly = list(polygon or [])
    if len(poly) < 3:
        return None
    pts = np.asarray([[int(p[0]), int(p[1])] for p in poly], dtype=np.int32)
    h_img, w_img = image_bgr.shape[:2]
    x1, y1 = int(pts[:, 0].min()), int(pts[:, 1].min())
    x2, y2 = int(pts[:, 0].max()), int(pts[:, 1].max())
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    pad = int(round(max(bw, bh) * float(pad_ratio)))
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w_img, x2 + pad)
    y2 = min(h_img, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image_bgr[y1:y2, x1:x2].copy()
    if crop.ndim == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    local = pts - np.array([x1, y1], dtype=np.int32)
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [local], 255)
    bgra = np.dstack([crop[:, :, :3], mask])
    if pad_square:
        bgra = _pad_bgra_to_square_transparent(bgra)
    return bgra


def export_results_to_local_dir(
    results: list[dict[str, Any]],
    image_bgr: np.ndarray,
    *,
    out_root: str | Path,
    source_image_name: str,
    skip_filtered: bool = True,
    skip_other: bool = True,
    crop_from_bbox: bool = True,
    pad_ratio: float = 0.05,
    pad_square: bool = False,
    jpeg_quality: int = 95,
    seg_transparent_png: bool = True,
) -> int:
    """
    将统一结果的分类框图按**类别分文件夹**输出到本地目录（替代 LS 上报）。

    裁剪方式：
    - ``crop_from_bbox=True``：取外接框矩形（复用 ``crop_instance_for_ingest``），写 ``.jpg``。
    - ``crop_from_bbox=False``（分割图）：按分割多边形裁剪。
      - ``seg_transparent_png=True``（默认）：多边形外透明，写**带 alpha 通道的 ``.png``**。
      - ``seg_transparent_png=False``：多边形外白底，写 ``.jpg``（旧行为）。

    目录结构：``<out_root>/<类别名>/<图名>_<序号>_<类别名>.{png|jpg}``；
    ``skip_filtered`` / ``skip_other`` 与上报语义一致。

    :return: 实际写盘的裁剪图数量。
    """
    from script.ls_classification_ingest import (
        _safe_slug,
        is_other_cls_name,
        normalize_cls_name_for_ingest,
    )
    from script.ls_seg_classification_ingest import crop_instance_for_ingest

    out_p = Path(out_root)
    stem = Path(source_image_name).stem
    quality = max(50, min(100, int(jpeg_quality)))
    use_alpha_png = (not crop_from_bbox) and seg_transparent_png
    saved = 0
    for idx, r in enumerate(results):
        row = _result_to_ls_ingest_row(r)
        if skip_filtered and row.get("filter"):
            continue
        cls_name = normalize_cls_name_for_ingest(
            str(row.get("cls_name") or "unknown")
        )
        if skip_other and is_other_cls_name(cls_name):
            continue
        try:
            if use_alpha_png:
                crop = _crop_instance_bgra_from_polygon(
                    image_bgr,
                    list(row.get("polygon") or []),
                    pad_ratio=pad_ratio,
                    pad_square=pad_square,
                )
            else:
                crop = crop_instance_for_ingest(
                    image_bgr,
                    row,
                    ingest_crop_from_bbox=crop_from_bbox,
                    pad_ratio=pad_ratio,
                    pad_square=pad_square,
                )
        except Exception as exc:
            logging.warning(
                "本地导出裁剪失败 %s #%s: %s", source_image_name, idx, exc
            )
            continue
        if crop is None or getattr(crop, "size", 0) == 0:
            logging.warning("本地导出裁剪为空 %s #%s", source_image_name, idx)
            continue
        cls_dir = out_p / _safe_slug(cls_name)
        cls_dir.mkdir(parents=True, exist_ok=True)
        ext = "png" if use_alpha_png else "jpg"
        fname = f"{_safe_slug(stem)}_{idx}_{_safe_slug(cls_name)}.{ext}"
        out_file = cls_dir / fname
        if use_alpha_png:
            ok = cv2.imwrite(str(out_file), crop)
        else:
            ok = cv2.imwrite(
                str(out_file), crop, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            )
        if ok:
            saved += 1
        else:
            logging.warning("本地导出写盘失败: %s", out_file)
    return saved


def _rmtree_onerror(func, path: str, exc_info) -> None:
    """shutil.rmtree 回调：忽略外置盘/网络盘上已消失的 AppleDouble(._*) 文件。"""
    exc = exc_info[1]
    if isinstance(exc, FileNotFoundError):
        return
    if isinstance(exc, PermissionError):
        try:
            os.chmod(path, 0o700)
            func(path)
        except OSError:
            logging.warning("清理时无法删除: %s", path)
        return
    raise exc


def _clear_run_output_dir(out_dir: str) -> None:
    """若输出目录已存在则整目录删除，避免本次结果与上次可视化、评估产物混合。"""
    p = Path(out_dir).expanduser().resolve()
    if not p.exists():
        return
    if not p.is_dir():
        logging.warning("输出路径已存在且不是目录，跳过清理: %s", p)
        return
    try:
        shutil.rmtree(p, onerror=_rmtree_onerror)
    except OSError as e:
        logging.warning("清理输出目录未完全成功，将继续执行: %s (%s)", p, e)
        return
    logging.info("已清理上次输出目录: %s", p)


_demo_print_lock = threading.Lock()


def _demo_safe_print(*args: Any, **kwargs: Any) -> None:
    with _demo_print_lock:
        print(*args, **kwargs)


def _is_nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _should_draw_result_image(
    save_path: Path,
    *,
    draw_picture_again: bool,
) -> bool:
    """
    是否执行结果图绘制落盘。

    - ``draw_picture_again=True``：续跑时若预测 xml 已存在仍重绘（校验开启时会对照 GT xml 标注框）。
    - ``draw_picture_again=False``：仅当结果图文件不存在或为空时才绘制。
    """
    if draw_picture_again:
        return True
    return not _is_nonempty_file(save_path)


def _eval_processed_stamp_path(
    output_dir: str, eval_subdir: str, rel: Path
) -> Path:
    safe = rel.as_posix().replace("/", "__")
    return Path(output_dir) / eval_subdir / "all" / "processed" / f"{safe}.stamp"


def _write_eval_processed_stamp(
    output_dir: str, eval_subdir: str, rel: Path
) -> None:
    stamp = _eval_processed_stamp_path(output_dir, eval_subdir, rel)
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(f"{rel.as_posix()}\n", encoding="utf-8")


def predict_all_image_complete_for_skip(
    *,
    output_dir: str,
    eval_subdir: str,
    rel: Path,
    out_dir: Path,
    has_validation_gt: bool,
    enable_validation: bool,
) -> bool:
    """
    增量续跑：预测 xml 必存在即跳过推理；有 GT 时另需 eval_metrics 完成戳（旧口径）。

    若预测 xml 已含 ``det_conf``/``cls_conf``，视为可仅跳过推理、按当前算法配置重新校验并重绘结果图，
    不再要求 eval 完成戳（复跑时会重算校验统计）。
    调用方须保证 OUTPUT_XML 已开启且会写出 pred xml。
    """
    pred_xml = out_dir / f"{rel.stem}.xml"
    if not _is_nonempty_file(pred_xml):
        return False
    if enable_validation and has_validation_gt:
        if voc_xml_has_dual_confidence(pred_xml):
            return True
        stamp = _eval_processed_stamp_path(output_dir, eval_subdir, rel)
        if not _is_nonempty_file(stamp):
            return False
    return True


def _build_cn_to_class_key_index(cn_index: dict[str, str]) -> dict[str, str]:
    """中文展示名 → 配置类 key（与 ``build_class_cn_display_index`` 互逆）。"""
    rev: dict[str, str] = {}
    for key, cn in cn_index.items():
        cn_s = str(cn or "").strip()
        key_s = str(key or "").strip()
        if cn_s and key_s and cn_s not in rev:
            rev[cn_s] = key_s
    return rev


def _collect_cls_global_conf_fallbacks(config: dict[str, Any]) -> list[float]:
    """收集各 ``cls`` 节点上的公共 ``cls_conf``（续跑无 per-class 配置时的回退）。"""
    fallbacks: list[float] = []

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            model = node.get("model")
            cls_conf = node.get("cls_conf")
            if model and cls_conf is not None:
                try:
                    fallbacks.append(float(cls_conf))
                except (TypeError, ValueError):
                    pass
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(config.get("models"))
    return fallbacks


def _get_resume_global_cls_conf(pipeline: InsectPredictAll) -> float | None:
    """续跑分类门限全局回退：与各 cls 节点 ``cfg.cls_conf`` 一致，取最小值（最宽松）。"""
    if hasattr(pipeline, "_resume_global_cls_conf_cache"):
        return pipeline._resume_global_cls_conf_cache
    fallbacks = _collect_cls_global_conf_fallbacks(pipeline.config)
    val: float | None = min(fallbacks) if fallbacks else None
    pipeline._resume_global_cls_conf_cache = val
    return val


def _get_cn_to_class_key_index(pipeline: InsectPredictAll) -> dict[str, str]:
    cached = getattr(pipeline, "_cn_to_class_key_index_cache", None)
    if cached is not None:
        return cached
    rev = _build_cn_to_class_key_index(pipeline._cn_display_index)
    pipeline._cn_to_class_key_index_cache = rev
    return rev


def _resolve_resume_xml_class_key(
    name: str,
    *,
    cn_to_key: dict[str, str],
    label_alias_map: dict[str, str] | None = None,
) -> str:
    """续跑 xml 类名 → 配置类 key（cn_index 反向表 + 评估 alias 表）。"""
    raw = str(name or "").strip()
    if not raw:
        return raw
    hit = cn_to_key.get(raw)
    if hit:
        return hit
    if label_alias_map:
        resolved = resolve_eval_label(raw, label_alias_map)
        if resolved:
            return resolved
    return raw


def _normalize_resume_xml_row_class_keys(
    row: dict[str, Any],
    *,
    cn_to_key: dict[str, str],
    label_alias_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    续跑 xml 行类名规范化：中文展示名 / 评估别名 → 配置类 key，供门限/report_tier 与线上一致。
    """
    out = dict(row)
    name = str(out.get("name") or out.get("cls_name") or "").strip()
    if not name:
        return out
    key = _resolve_resume_xml_class_key(
        name, cn_to_key=cn_to_key, label_alias_map=label_alias_map
    )
    if key and key != name:
        out["name"] = key
        out["cls_name"] = key
        out.setdefault("cn_name", name)
        out.setdefault("infer_name", key)
    return out


def _collect_cls_out_dicts_from_config(config: dict[str, Any]) -> list[dict[str, Any]]:
    """收集各分类 ``cls`` 节点的 ``out`` 表（续跑 ``cls_conf`` 与线上一致）。"""
    outs: list[dict[str, Any]] = []

    def _walk(node: Any) -> None:
        if not isinstance(node, dict):
            return
        if node.get("model") and "cls_conf" in node:
            out = node.get("out")
            if isinstance(out, dict) and out:
                outs.append(out)
        for value in node.values():
            if isinstance(value, dict):
                _walk(value)
            elif isinstance(value, list):
                for item in value:
                    _walk(item)

    _walk(config.get("models"))
    return outs


def _walk_models_for_out_dicts(node: Any, outs: list[dict[str, Any]]) -> None:
    """递归收集 ``models`` 子树中全部 ``out`` 表（供续跑后置过滤合并门限）。"""
    if not isinstance(node, dict):
        return
    out = node.get("out")
    if isinstance(out, dict) and out:
        outs.append(out)
    for key, value in node.items():
        if key == "out":
            continue
        if isinstance(value, dict):
            _walk_models_for_out_dicts(value, outs)
        elif isinstance(value, list):
            for item in value:
                _walk_models_for_out_dicts(item, outs)


def _collect_all_out_dicts_from_config(config: dict[str, Any]) -> list[dict[str, Any]]:
    models = config.get("models")
    outs: list[dict[str, Any]] = []
    if isinstance(models, dict):
        _walk_models_for_out_dicts(models, outs)
    return outs


def _build_merged_postprocess_alg_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    合并分类 ``cls`` 节点 ``out`` 路由门限（首条匹配与线上一致）。

    不使用 segment/detect 的 ``out``（含 ``*`` 通配 dia），避免盖掉 per-class ``cls_conf``。
    """
    cls_outs = _collect_cls_out_dicts_from_config(config)
    if not cls_outs:
        cls_outs = _collect_all_out_dicts_from_config(config)
    patterns: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for out in cls_outs:
        table = build_alg_table_from_out(out)
        ordered = table.get(_ROUTE_PATTERNS_KEY) if table else None
        if not isinstance(ordered, list):
            continue
        for item in ordered:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            pat, entry = str(item[0]), item[1]
            if pat in seen or not isinstance(entry, dict):
                continue
            patterns.append((pat, entry))
            seen.add(pat)
    return {_ROUTE_PATTERNS_KEY: patterns} if patterns else {}


def _get_merged_postprocess_alg(pipeline: InsectPredictAll) -> dict[str, Any]:
    cached = getattr(pipeline, "_merged_postprocess_alg_cache", None)
    if cached is not None:
        return cached
    merged = _build_merged_postprocess_alg_from_config(pipeline.config)
    pipeline._merged_postprocess_alg_cache = merged
    return merged


def _demote_row_cls_conf_for_resume(
    row: dict[str, Any],
    alg: dict[str, Any],
    global_cls_conf: float | None,
) -> dict[str, Any] | None:
    """
    续跑时按当前 JSON 重放分类 top1 门限；未过门限则返回 filtered 行，否则 None。
    """
    top1 = str(row.get("cls_name") or row.get("name") or "").strip()
    if not top1 or _is_other_class_name(top1):
        return None
    try:
        conf = float(row.get("cls_conf", 0.0) or 0.0)
    except (TypeError, ValueError):
        conf = 0.0
    thr = resolve_cls_top1_threshold(alg, top1, global_cls_conf)
    if thr is None or conf > float(thr):
        return None
    det = dict(row)
    _stash_top1_cls_before_other(det)
    det["cls_conf_threshold"] = float(thr)
    reason = _format_cls_conf_other_reason(top1, conf, float(thr))
    det = _mark_row_cls_other(det, reason)
    det["name"] = "other"
    det["cls_conf"] = conf
    det["filter_reason"] = reason
    det["filtered"] = True
    return det


def _reapply_resume_postprocess_filters(
    rows: list[dict[str, Any]],
    pipeline: InsectPredictAll,
    *,
    label_alias_map: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    续跑从 xml 读回预测后，按当前 ``insect_alg_all.json`` 重放后置过滤。

    与线上一致：分类 ``cls_conf`` 门限 → ``report_tier`` 报出类过滤。
    """
    if not rows:
        return [], []
    alg = _get_merged_postprocess_alg(pipeline)
    global_cls_conf = _get_resume_global_cls_conf(pipeline)
    cn_to_key = _get_cn_to_class_key_index(pipeline)
    kept: list[dict[str, Any]] = []
    filtered: list[dict[str, Any]] = []
    for row in rows:
        norm = _normalize_resume_xml_row_class_keys(
            row,
            cn_to_key=cn_to_key,
            label_alias_map=label_alias_map,
        )
        demoted = _demote_row_cls_conf_for_resume(
            norm, alg, global_cls_conf=global_cls_conf
        )
        if demoted is not None:
            filtered.append(demoted)
            continue
        kept.append(norm)
    if filtered and logging.getLogger().isEnabledFor(logging.DEBUG):
        logging.debug(
            "续跑 cls_conf 重过滤: %d 条 → filtered-other",
            len(filtered),
        )
    kept = pipeline._apply_report_tier_filter(kept, filtered_acc=filtered)
    return kept, filtered


def _predict_results_from_voc_xml(xml_path: Path) -> list[dict[str, Any]]:
    """从已落盘的预测 VOC xml 还原 predict 结果行（供续跑累计校验统计）。"""
    if voc_xml_has_dual_confidence(xml_path):
        return parse_pascal_voc_pred_objects(xml_path)
    out: list[dict[str, Any]] = []
    for o in parse_pascal_voc_objects(str(xml_path)):
        name = str(o.get("name") or "")
        out.append(
            {
                "name": name,
                "cls_name": name,
                "cn_name": name,
                "location": [int(o["x1"]), int(o["y1"]), int(o["x2"]), int(o["y2"])],
                "score": 1.0,
                "det_conf": 1.0,
                "cls_conf": 1.0,
            }
        )
    return out


def _validate_predict_all_resume_config(
    *,
    skip_if_output_exists: bool,
    clean_output_before_run: bool,
    output_xml: bool,
) -> bool:
    """校验续跑配置；返回是否实际启用跳过。"""
    if not skip_if_output_exists:
        return False
    if clean_output_before_run:
        logging.warning(
            "SKIP_IF_OUTPUT_EXISTS=True 但 CLEAN_OUTPUT_BEFORE_RUN=True，增量跳过已关闭"
        )
        return False
    if not output_xml:
        raise ValueError(
            "增量续跑要求 OUTPUT_XML=True（输出目录预测 xml 标记推理完成）；"
            "请设置 OUTPUT_XML=True 或关闭 SKIP_IF_OUTPUT_EXISTS"
        )
    return True


def _save_predict_all_type_confusion_crops(
    *,
    img_bgr: np.ndarray,
    rel: Path,
    pred_rows: list[dict[str, Any]],
    gts: list[dict[str, Any]] | None,
    matches: list[tuple[int, int, float]] | None,
    gt_source: str,
    gt_name: str,
    save_type_confusion_crops: bool,
    output_dir: str,
    standard_eval_subdir: str,
    class_merge_eval: dict[str, list[str]] | None,
    cls_merge_type_index: Any,
    eval_label_alias_map: dict[str, str] | None,
    eval_fuzzy_only_wildcard: bool,
) -> None:
    """类型混淆/漏检切图：写入 eval_metrics/all/type_confusion_crops*。"""
    if not save_type_confusion_crops:
        return
    eval_crop_root = os.path.join(output_dir, standard_eval_subdir)
    if gt_source == "filename":
        _save_type_confusion_crops_for_filename_gt(
            img_bgr,
            rel,
            pred_rows,
            gt_name,
            class_merge_to_groups=class_merge_eval,
            hierarchy_type_index=cls_merge_type_index,
            eval_root=eval_crop_root,
            branch="all",
            label_alias_map=eval_label_alias_map,
            fuzzy_only_wildcard=eval_fuzzy_only_wildcard,
        )
        return
    if not gts:
        return
    _save_type_confusion_crops_for_branch(
        img_bgr,
        rel,
        pred_rows,
        gts,
        matches,
        class_merge_to_groups=class_merge_eval,
        hierarchy_type_index=cls_merge_type_index,
        eval_root=eval_crop_root,
        branch="all",
        label_alias_map=eval_label_alias_map,
        fuzzy_only_wildcard=eval_fuzzy_only_wildcard,
    )


def _load_resume_results_from_pred_xml(
    pred_xml: Path,
    pipeline: InsectPredictAll,
    *,
    label_alias_map: dict[str, str] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """续跑：从已落盘预测 xml 还原结果并按当前算法配置重放后置过滤。"""
    raw_results = _predict_results_from_voc_xml(pred_xml)
    return _reapply_resume_postprocess_filters(
        raw_results,
        pipeline,
        label_alias_map=label_alias_map,
    )


@dataclass(frozen=True)
class _DemoPostConfig:
    """批量 demo 后处理只读配置（推理完成后校验/画图/导出）。"""

    enable_validation: bool
    val_box_match_metric: str
    val_geom_threshold: float
    class_merge_eval: dict[str, list[str]] | None
    cls_merge_type_index: Any
    eval_label_alias_map: dict[str, str] | None
    tier_equivalence: ClassTierEquivalence
    eval_fuzzy_only_wildcard: bool
    save_type_confusion_crops: bool
    standard_eval_subdir: str
    output_dir: str
    save_image: bool
    draw_picture_again: bool
    draw_bbox: bool
    draw_polygon: bool
    draw_filter_sources: frozenset[str]
    draw_validation_focus: bool
    draw_eval_fp_non_focus: bool
    draw_label: bool
    draw_label_errors_only: bool
    report_focus: frozenset[str]
    label_font_size: int | None
    show_source_in_label: bool
    save_image_scale: float | None
    cn_index: dict[str, str]
    output_xml: bool
    output_xml_cn_name: bool
    output_xml_dual_conf: bool
    output_labelme: bool
    labelme_copy_image: bool
    enable_ls_ingest: bool
    ls_ingestor: Any
    ls_local_dir: Path | None
    ls_ingest_include_filtered: bool
    ls_ingest_skip_filtered: bool
    ls_ingest_skip_other: bool
    ls_ingest_crop_from_bbox: bool
    ls_ingest_pad_square: bool
    ls_ingest_dry_run: bool
    ls_ingest_jpeg_quality: int
    ls_export_seg_transparent_png: bool
    perf_quiet: bool = False


@dataclass
class _DemoImageJob:
    idx: int
    total: int
    rel: Path
    img: np.ndarray
    img_path: Path
    results: list[dict[str, Any]]
    filtered: list[dict[str, Any]]
    infer_cost: float
    save_path: Path
    out_dir: Path
    phase_line: str = ""
    skip_inference: bool = False


def _accumulate_other_filter_reason_stats(
    filtered: list[dict[str, Any]] | None,
    stats: dict[str, int],
) -> None:
    """汇总 filtered 中「后置 other」的 ``filter_reason`` 计数（便于日志排查）。"""
    for r in filtered or []:
        fr = str(r.get("filter_reason") or r.get("cls_other_reason") or "").strip()
        if not _is_other_filter_reason(fr):
            continue
        if fr.startswith("other_cls_conf:"):
            key = fr.split("(", 1)[0]
        else:
            key = fr
        stats[key] = int(stats.get(key, 0)) + 1


def _print_other_filter_reason_stats(stats: dict[str, int]) -> None:
    if not stats:
        return
    lines = ["后置 other 过滤原因汇总（分类模型无 other 类，均为后处理）:"]
    for k in sorted(stats.keys()):
        lines.append(f"  {k}: {stats[k]}")
    _demo_safe_print("\n".join(lines))


class _DemoPostShared:
    """批量 demo 后处理共享可变状态（流水线模式下加锁）。"""

    def __init__(
        self,
        *,
        stat_total: dict[str, int],
        stat_by_cls: defaultdict[str, dict[str, int]],
        std_cm: defaultdict[tuple[str, str], int] | None,
        use_lock: bool,
    ) -> None:
        self.stat_total = stat_total
        self.stat_by_cls = stat_by_cls
        self.std_cm = std_cm
        self.use_lock = use_lock
        self.lock = threading.Lock()
        self.ls_ingest_total_tasks = 0
        self.ls_ingest_total_pushed = 0
        self.ls_export_total_saved = 0
        self.stat_other_filter: dict[str, int] = {}

    def _guard(self) -> Any:
        return self.lock if self.use_lock else nullcontext()

    def postprocess(self, cfg: _DemoPostConfig, job: _DemoImageJob) -> None:
        with self._guard():
            self._postprocess_locked(cfg, job)

    def _postprocess_locked(self, cfg: _DemoPostConfig, job: _DemoImageJob) -> None:
        results = job.results
        filtered = job.filtered
        img = job.img
        rel = job.rel
        img_path = job.img_path
        save_path = job.save_path
        out_dir = job.out_dir
        infer_cost = job.infer_cost

        gts: list[dict] | None = None
        matches: list[tuple[int, int, float]] | None = None
        matched_p: set[int] | None = None
        eval_boxes: list[dict[str, Any]] | None = None
        results_for_draw: list[dict[str, Any]] = results
        stat_delta: dict[str, int] | None = None

        _accumulate_other_filter_reason_stats(filtered, self.stat_other_filter)

        if cfg.enable_validation:
            gts, gt_source = _load_predict_all_ground_truth(
                img_path,
                rel,
                img,
                label_alias_map=cfg.eval_label_alias_map,
            )
            if gts:
                _stat_before = {
                    k: int(self.stat_total.get(k, 0))
                    for k in ("tp", "fp", "fn", "cls_err")
                }
                if gt_source == "xml":
                    self.stat_total["img_with_xml"] += 1
                elif gt_source == "filename":
                    self.stat_total["img_with_filename_gt"] += 1
                gt_name = str(gts[0].get("name", "") or "")
                results_for_draw = [dict(r) for r in results]
                filtered_for_draw = [dict(r) for r in filtered] if filtered else None
                if gt_source == "filename":
                    matches, matched_p, pred_rows = (
                        _accumulate_predict_all_filename_gt_validation(
                            self.stat_total,
                            self.stat_by_cls,
                            results,
                            gt_name,
                            class_merge=cfg.class_merge_eval,
                            cls_merge_type_index=cfg.cls_merge_type_index,
                            label_alias_map=cfg.eval_label_alias_map,
                            fuzzy_only_wildcard=cfg.eval_fuzzy_only_wildcard,
                            tier_equivalence=cfg.tier_equivalence,
                        )
                    )
                    eval_boxes = _annotate_filename_gt_eval_tags(
                        results_for_draw,
                        gt_name,
                        class_merge=cfg.class_merge_eval,
                        cls_merge_type_index=cfg.cls_merge_type_index,
                        label_alias_map=cfg.eval_label_alias_map,
                        fuzzy_only_wildcard=cfg.eval_fuzzy_only_wildcard,
                        tier_equivalence=cfg.tier_equivalence,
                    )
                else:
                    matches, matched_p, pred_rows = _accumulate_predict_all_validation(
                        self.stat_total,
                        self.stat_by_cls,
                        results,
                        gts,
                        class_merge=cfg.class_merge_eval,
                        geom_metric=cfg.val_box_match_metric,
                        geom_threshold=cfg.val_geom_threshold,
                        cls_merge_type_index=cfg.cls_merge_type_index,
                        label_alias_map=cfg.eval_label_alias_map,
                        filtered=filtered,
                        fuzzy_only_wildcard=cfg.eval_fuzzy_only_wildcard,
                        tier_equivalence=cfg.tier_equivalence,
                    )
                    _pred_pool, _match_only_flags, filtered_other_indices = (
                        _build_eval_pred_pool(results, filtered)
                    )
                    eval_boxes = _annotate_validation_eval_tags(
                        results_for_draw,
                        filtered_for_draw,
                        gts=gts,
                        matches=matches,
                        matched_p=matched_p,
                        n_result_preds=len(results),
                        filtered_other_indices=filtered_other_indices,
                        class_merge=cfg.class_merge_eval,
                        cls_merge_type_index=cfg.cls_merge_type_index,
                        label_alias_map=cfg.eval_label_alias_map,
                        fuzzy_only_wildcard=cfg.eval_fuzzy_only_wildcard,
                        tier_equivalence=cfg.tier_equivalence,
                    )
                stat_delta = {
                    k: int(self.stat_total.get(k, 0)) - _stat_before[k]
                    for k in ("tp", "fp", "fn", "cls_err")
                }
                if filtered_for_draw is not None:
                    filtered = filtered_for_draw

                if self.std_cm is not None:
                    if gt_source == "filename":
                        _std_eval_collect_confusion_filename_gt(
                            self.std_cm,
                            pred_rows,
                            gt_name,
                            cfg.class_merge_eval,
                            cfg.cls_merge_type_index,
                            label_alias_map=cfg.eval_label_alias_map,
                            fuzzy_only_wildcard=cfg.eval_fuzzy_only_wildcard,
                            tier_equivalence=cfg.tier_equivalence,
                        )
                    else:
                        _std_eval_collect_confusion(
                            self.std_cm,
                            pred_rows,
                            gts,
                            matches,
                            matched_p,
                            cfg.class_merge_eval,
                            cfg.cls_merge_type_index,
                            label_alias_map=cfg.eval_label_alias_map,
                            fuzzy_only_wildcard=cfg.eval_fuzzy_only_wildcard,
                            tier_equivalence=cfg.tier_equivalence,
                        )
                if cfg.save_type_confusion_crops and gts:
                    _save_predict_all_type_confusion_crops(
                        img_bgr=img,
                        rel=rel,
                        pred_rows=pred_rows,
                        gts=gts if gt_source == "xml" else None,
                        matches=matches if gt_source == "xml" else None,
                        gt_source=gt_source,
                        gt_name=gt_name,
                        save_type_confusion_crops=cfg.save_type_confusion_crops,
                        output_dir=cfg.output_dir,
                        standard_eval_subdir=cfg.standard_eval_subdir,
                        class_merge_eval=cfg.class_merge_eval,
                        cls_merge_type_index=cfg.cls_merge_type_index,
                        eval_label_alias_map=cfg.eval_label_alias_map,
                        eval_fuzzy_only_wildcard=cfg.eval_fuzzy_only_wildcard,
                    )

        _t_post = time.perf_counter()
        will_draw_image = cfg.save_image and _should_draw_result_image(
            save_path,
            draw_picture_again=cfg.draw_picture_again,
        )
        if will_draw_image:
            filtered_for_draw = _select_filtered_for_draw(
                filtered,
                draw_filter_sources=cfg.draw_filter_sources,
                draw_validation_focus=cfg.draw_validation_focus,
                report_focus=cfg.report_focus,
                class_merge=cfg.class_merge_eval,
                label_alias_map=cfg.eval_label_alias_map,
            )
            draw_results(
                img,
                results_for_draw,
                str(save_path),
                draw_bbox=cfg.draw_bbox,
                draw_polygon=cfg.draw_polygon,
                draw_label=cfg.draw_label,
                draw_label_errors_only=cfg.draw_label_errors_only,
                label_font_size=cfg.label_font_size,
                show_source_in_label=cfg.show_source_in_label,
                eval_boxes=eval_boxes,
                filtered_results=filtered_for_draw,
                draw_filter=bool(filtered_for_draw),
                cn_index=cfg.cn_index,
                class_merge=cfg.class_merge_eval,
                label_alias_map=cfg.eval_label_alias_map,
                draw_focus=cfg.report_focus,
                output_image_scale=cfg.save_image_scale,
                draw_eval_fp_non_focus=cfg.draw_eval_fp_non_focus,
                cls_merge_type_index=cfg.cls_merge_type_index,
                fuzzy_only_wildcard=cfg.eval_fuzzy_only_wildcard,
                tier_equivalence=cfg.tier_equivalence,
            )

        if cfg.output_xml:
            save_pascal_voc_xml_for_results(
                img,
                results,
                out_dir=out_dir,
                rel_path=rel,
                use_cn_name=cfg.output_xml_cn_name,
                cn_index=cfg.cn_index,
                include_dual_conf=cfg.output_xml_dual_conf,
            )

        if cfg.output_labelme and not job.skip_inference:
            save_labelme_json_for_results(
                img,
                results,
                out_dir=out_dir,
                rel_path=rel,
                source_image_path=img_path,
                copy_image=cfg.labelme_copy_image,
                cn_index=cfg.cn_index,
            )

        if cfg.enable_ls_ingest and cfg.ls_ingestor is not None and not job.skip_inference:
            if cfg.ls_ingest_include_filtered:
                ingest_src = list(results) + list(filtered)
                eff_skip_filtered = False
            else:
                ingest_src = list(results)
                eff_skip_filtered = cfg.ls_ingest_skip_filtered
            try:
                ls_tasks, ls_resp = ingest_results_to_ls(
                    cfg.ls_ingestor,
                    ingest_src,
                    img,
                    source_image_name=str(rel),
                    skip_filtered=eff_skip_filtered,
                    skip_other=cfg.ls_ingest_skip_other,
                    ingest_crop_from_bbox=cfg.ls_ingest_crop_from_bbox,
                    pad_square=cfg.ls_ingest_pad_square,
                    post=not cfg.ls_ingest_dry_run,
                )
                self.ls_ingest_total_tasks += len(ls_tasks)
                if ls_resp:
                    self.ls_ingest_total_pushed += sum(
                        int(rr.get("task_count", 0) or 0) for rr in ls_resp
                    )
                _demo_safe_print(
                    f"    [LS] 上报分类任务={len(ls_tasks)}"
                    + ("（dry_run，未推送）" if cfg.ls_ingest_dry_run else "")
                )
            except Exception as e:
                logging.warning("LS 上报失败 %s: %s", rel, e, exc_info=True)

        if cfg.enable_ls_ingest and cfg.ls_local_dir is not None and not job.skip_inference:
            if cfg.ls_ingest_include_filtered:
                export_src = list(results) + list(filtered)
                eff_skip_filtered = False
            else:
                export_src = list(results)
                eff_skip_filtered = cfg.ls_ingest_skip_filtered
            try:
                n_saved = export_results_to_local_dir(
                    export_src,
                    img,
                    out_root=cfg.ls_local_dir,
                    source_image_name=str(rel),
                    skip_filtered=eff_skip_filtered,
                    skip_other=cfg.ls_ingest_skip_other,
                    crop_from_bbox=cfg.ls_ingest_crop_from_bbox,
                    pad_square=cfg.ls_ingest_pad_square,
                    jpeg_quality=cfg.ls_ingest_jpeg_quality,
                    seg_transparent_png=cfg.ls_export_seg_transparent_png,
                )
                self.ls_export_total_saved += n_saved
                _demo_safe_print(
                    f"    [本地导出] 分类框图={n_saved} -> {cfg.ls_local_dir}"
                )
            except Exception as e:
                logging.warning("本地导出失败 %s: %s", rel, e, exc_info=True)

        if cfg.enable_validation and gts:
            _write_eval_processed_stamp(cfg.output_dir, cfg.standard_eval_subdir, rel)

        post_cost = time.perf_counter() - _t_post

        n_filt_draw = sum(
            1
            for r in (filtered or [])
            if str(r.get("source") or "").strip() in cfg.draw_filter_sources
        )
        line = f"[{job.idx}/{job.total}] "
        if job.skip_inference:
            line += "跳过推理"
            if will_draw_image:
                line += "，重绘结果图"
            elif cfg.save_image:
                line += "，结果图已存在跳过重绘"
            line += f": {rel} -> {len(results)} 个目标"
        else:
            line += f"{rel} -> {len(results)} 个目标"
        line += (f"（过滤绘测 {n_filt_draw}）" if n_filt_draw else "")
        if not job.skip_inference:
            line += f" 推理={infer_cost:.3f}s"
        if post_cost >= 0.001:
            line += f" 绘测后处理={post_cost:.3f}s"
        if job.phase_line:
            line += f" {job.phase_line}"
        if cfg.enable_validation and gts and stat_delta is not None:
            line += (
                f" gt={len(gts)}"
                f" tp={stat_delta['tp']}"
                f" fp={stat_delta['fp']}"
                f" fn={stat_delta['fn']}"
                f" cls_err={stat_delta['cls_err']}"
            )
        _demo_safe_print(line)
        if cfg.perf_quiet:
            return
        for r in results:
            loc = r["location"]
            _demo_safe_print(
                f"    {r.get('cn_name') or r['name']} score={r['score']:.2f} "
                f"source={r.get('source')} box={loc}"
            )


# --------------------------------------------------------------------------- #
#  性能测试入口配置（与 test_api.py 同一套图片目录/文件名约定）
# --------------------------------------------------------------------------- #

# 与 test_api.py 中 ll_result 一致，便于 API 与本地 pipeline 对比同一批图
XINGNENG_DEFAULT_IMAGE_NAMES_TEXT = """202607092012_1952342830425243648.jpg
202607092021_1952342830425243648.jpg
202607092022_1952342830425243648.jpg
202607092031_1952342830425243648.jpg
202607092032_1952342830425243648.jpg
202607092041_1952342830425243648.jpg
202607092042_1952342830425243648.jpg
202607092052_1952342830425243648 (1).jpg
202607092052_1952342830425243648.jpg
202607092102_1952342830425243648 (1).jpg
202607092102_1952342830425243648.jpg
202607092112_1952342830425243648 (1).jpg
202607092112_1952342830425243648.jpg
202607092122_1952342830425243648 (1).jpg
202607092122_1952342830425243648.jpg
202607092132_1952342830425243648 (1).jpg
202607092132_1952342830425243648.jpg
202607092142_1952342830425243648 (1).jpg
202607092142_1952342830425243648.jpg
202607092152_1952342830425243648 (1).jpg
202607092152_1952342830425243648.jpg
202607092202_1952342830425243648 (1).jpg
202607092202_1952342830425243648.jpg
202607092212_1952342830425243648 (1).jpg
202607092212_1952342830425243648.jpg
202607092222_1952342830425243648 (1).jpg
202607092222_1952342830425243648.jpg
202607092232_1952342830425243648 (1).jpg
202607092232_1952342830425243648.jpg
202607092242_1952342830425243648 (1).jpg
202607092242_1952342830425243648.jpg
202607092252_1952342830425243648.jpg
202607092253_1952342830425243648.jpg"""


def resolve_xingneng_test_image_dir() -> str:
    """按平台返回默认测试图片目录（与 test_api.py TEST_IMAGE_DIR 对齐）。"""
    if platform.system() == "Darwin":
        candidates = [
            # 若已从服务端同步 test_api 同一批图，优先此目录
            "/Volumes/shunyao-h1/训练数据/北京比赛/田间采集/原始图片",
            "/Volumes/shunyao-h1/训练数据/北京比赛/性能测试",
        ]
        for path in candidates:
            if Path(path).is_dir():
                return path
        return candidates[0]
    return "/home/beyond/桌面/模型识别/田间采集/原始图片"


def resolve_local_test_image_paths(
    *,
    image_dir: str | None = None,
    image_names: list[str] | None = None,
    image_paths: list[str] | None = None,
    image_path: str | None = None,
    search_subdirs: bool = True,
) -> tuple[Path, list[Path]]:
    """
    解析本地性能测试图片路径（优先级与 test_api.resolve_test_image_urls 一致）：

    1. ``image_paths``：完整本地路径列表
    2. ``image_dir`` + ``image_names``：目录 + 文件名列表
    3. ``image_path``：单张完整路径
    4. 仅 ``image_dir``：递归收集目录下全部图片
    """
    if image_paths:
        files = [Path(p) for p in image_paths if p]
        if not files:
            raise ValueError("TEST_IMAGE_PATHS 为空")
        input_p = files[0].parent
        return input_p, files

    if image_names and image_dir:
        base = Path(image_dir)
        if not base.is_dir():
            raise ValueError(f"TEST_IMAGE_DIR 不存在或不是目录: {image_dir}")
        files: list[Path] = []
        missing: list[str] = []
        for raw in image_names:
            name = raw.strip()
            if not name:
                continue
            direct = base / name
            if direct.is_file():
                files.append(direct)
                continue
            if search_subdirs:
                found = next((p for p in base.rglob(name) if p.is_file()), None)
                if found is not None:
                    files.append(found)
                    continue
            missing.append(name)
        if missing:
            preview = ", ".join(missing[:5])
            if len(missing) > 5:
                preview += " ..."
            logging.warning(
                "以下测试图片未找到 (%d/%d): %s",
                len(missing),
                len(image_names),
                preview,
            )
        if not files:
            raise ValueError(
                f"目录 {image_dir} 下未找到任何指定测试图片"
                f"（共 {len(image_names)} 个文件名）"
            )
        return base, files

    if image_path:
        p = Path(image_path)
        if not p.is_file():
            raise ValueError(f"TEST_IMAGE_PATH 不存在: {image_path}")
        return p.parent, [p]

    if image_dir:
        return collect_images(image_dir)

    raise ValueError(
        "请设置 TEST_IMAGE_PATHS，或 TEST_IMAGE_DIR + TEST_IMAGE_NAMES，"
        "或 TEST_IMAGE_PATH / TEST_IMAGE_DIR"
    )


def print_xingneng_infer_summary(
    records: list[tuple[str, float, int]],
    *,
    wall_elapsed: float | None = None,
) -> None:
    """打印与 test_api 类似的单图/汇总耗时（仅 pipeline.predict 段）。"""
    if not records:
        return
    elapsed_list = [cost for _, cost, _ in records]
    total_cost = sum(elapsed_list)
    avg = total_cost / len(elapsed_list)
    min_cost = min(elapsed_list)
    max_cost = max(elapsed_list)
    min_name = min(records, key=lambda r: r[1])[0]
    max_name = max(records, key=lambda r: r[1])[0]

    print(f"\n{'=' * 60}")
    print("汇总:")
    for name, cost, n_results in records:
        print(f"  PASS  {name:45s}  results={n_results}  {cost:.2f}s")
    print(
        f"\n合计: {len(records)} 张"
    )
    print("\n耗时统计（pipeline.predict，含预处理/合并/路由）:")
    if wall_elapsed is not None:
        print(f"  总耗时: {wall_elapsed:.2f}s")
    print(f"  平均耗时: {avg:.2f}s")
    print(f"  最小耗时: {min_cost:.2f}s  ({min_name})")
    print(f"  最大耗时: {max_cost:.2f}s  ({max_name})")
    if wall_elapsed is not None and total_cost > 0:
        print(f"  单图推理耗时合计: {total_cost:.2f}s")
        print(
            f"  总耗时/单图合计 ≈ {wall_elapsed / total_cost:.2f} "
            f"({'接近串行' if wall_elapsed / total_cost < 1.15 else '存在重叠或抖动'})"
        )
    print(
        "  说明: 本地 pipeline 耗时不含 HTTP/网络；"
        "与 test_api 对比时请使用同一 TEST_IMAGE_DIR + TEST_IMAGE_NAMES。"
    )


if __name__ == "__main__":
    # mac:
    # /Users/shunyaoyin/miniconda310/miniconda3/envs/yolo11/bin/python3 \
    #   /Users/shunyaoyin/Documents/code/ai-company/insect/script/predict_all_xingneng.py
    # ubuntu:
    # nohup .../python3 /data/script/predict_all_xingneng.py > xingneng.log 2>&1 &
    print("predict_all_xingneng.py start")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    logging.info("predict_all_xingneng.py start")

    CONFIG_PATH = INSECT_ALG_ALL_JSON_REL

    # ===== 测试图片（与 test_api.py 同一套约定）=====
    # 方式一（推荐）：目录 + 文件名列表
    TEST_IMAGE_DIR = resolve_xingneng_test_image_dir()
    TEST_IMAGE_NAMES: list[str] = XINGNENG_DEFAULT_IMAGE_NAMES_TEXT.split("\n")
    # 方式二：完整本地路径列表（非空时优先于方式一）
    TEST_IMAGE_PATHS: list[str] = []
    # 方式三：单张完整路径（仅当方式一、二均未配置时使用）
    TEST_IMAGE_PATH = ""
    # 留空 TEST_IMAGE_NAMES 且未设方式二/三时，递归跑 TEST_IMAGE_DIR 下全部图片
    # （mac 未同步 33 张标准集时可设为 True，跑性能测试目录内全部图）
    USE_DIR_ALL_IMAGES = False

    try:
        if USE_DIR_ALL_IMAGES and not TEST_IMAGE_PATHS and not TEST_IMAGE_PATH:
            input_p, image_files = resolve_local_test_image_paths(
                image_dir=TEST_IMAGE_DIR,
            )
        else:
            input_p, image_files = resolve_local_test_image_paths(
                image_dir=TEST_IMAGE_DIR,
                image_names=TEST_IMAGE_NAMES or None,
                image_paths=TEST_IMAGE_PATHS or None,
                image_path=TEST_IMAGE_PATH or None,
            )
    except ValueError as exc:
        print(f"配置错误: {exc}")
        sys.exit(2)

    INPUT_PATH = str(input_p)
    OUTPUT_DIR = INPUT_PATH + "-xingneng"
    ROOT_IDS: list[str] | None = None

    # ===== 性能测试：关闭校验/可视化/导出，只测 pipeline.predict =====
    SAVE_IMAGE = False
    DRAW_PICTURE_AGAING = False
    DRAW_BBOX = False
    DRAW_POLYGON = False
    DRAW_VALIDATION_FOCUS = False
    DRAW_EVAL_FP_NON_FOCUS = False
    DRAW_LABEL = False
    DRAW_LABEL_ERRORS_ONLY = False
    LABEL_FONT_SIZE: int | None = None
    SHOW_SOURCE_IN_LABEL = False
    SAVE_IMAGE_SCALE: float | None = None

    OUTPUT_XML = False
    OUTPUT_XML_CN_NAME = False
    OUTPUT_XML_DUAL_CONF = False
    OUTPUT_LABELME = False
    LABELME_COPY_IMAGE = False

    ENABLE_VALIDATION = False
    VAL_BOX_MATCH_METRIC = "iou"
    VAL_GEOM_THRESHOLD = 0.25
    ENABLE_STANDARD_EVAL = False
    STANDARD_EVAL_SUBDIR = "eval_metrics"
    CLEAN_OUTPUT_BEFORE_RUN = False
    SKIP_IF_OUTPUT_EXISTS = False
    SAVE_TYPE_CONFUSION_CROPS = False
    SORT_STAT_BY_ACC = True
    CLASS_MERGE_TO_GROUPS: dict[str, list[str]] | None = None
    EVAL_INSECT_WILDCARD = True
    EVAL_INSECT_WILDCARD_STRICT = True

    ENABLE_MASK_RATE_FILTER = False

    POSTPROCESS_PIPELINE = False
    POSTPROCESS_WORKERS = 1
    PERF_QUIET = True

    ENABLE_LS_INGEST = False
    LS_INGEST_URL = ""
    LS_INGEST_DRY_RUN = False
    LS_INGEST_SKIP_FILTERED = False
    LS_INGEST_SKIP_OTHER = False
    LS_INGEST_INCLUDE_FILTERED = False
    LS_INGEST_CROP_FROM_BBOX = False
    LS_INGEST_PAD_SQUARE = True
    LS_EXPORT_SEG_TRANSPARENT_PNG = True
    CLEAN_LS_EXPORT_BEFORE_RUN = False
    LS_INGEST_BATCH_SIZE = 200
    LS_INGEST_JPEG_QUALITY = 95
    LS_INGEST_CHOICE_FROM_NAME = "choice"
    LS_INGEST_CHOICE_TO_NAME = "image"

    class_merge_eval: dict[str, list[str]] | None = None
    eval_label_alias_map: dict[str, str] | None = None
    eval_fuzzy_only_wildcard = False
    if ENABLE_VALIDATION:
        class_merge_eval = build_eval_class_merge(
            CLASS_MERGE_TO_GROUPS, insect_wildcard=EVAL_INSECT_WILDCARD
        )
        eval_fuzzy_only_wildcard = bool(
            EVAL_INSECT_WILDCARD and EVAL_INSECT_WILDCARD_STRICT
        )
        if eval_fuzzy_only_wildcard:
            logging.info(
                "校验 insect 通配为严格模式：仅 insect/other 前缀族彼此互匹，具体物种不与粗类互匹"
            )

    cls_merge_type_index = None

    stat_total: dict[str, int] = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "cls_err": 0,
        "geom_pairs": 0,
        "img_with_xml": 0,
        "img_with_filename_gt": 0,
    }
    stat_by_cls: dict[str, dict[str, int]] = defaultdict(
        lambda: {"gt": 0, "pred": 0, "tp": 0, "fp": 0, "fn": 0, "cls_err": 0}
    )
    std_cm: defaultdict[tuple[str, str], int] | None = (
        defaultdict(int) if (ENABLE_VALIDATION and ENABLE_STANDARD_EVAL) else None
    )

    # 推理耗时统计：仅统计 pipeline.predict 的纯推理时间（秒）
    infer_times: list[float] = []
    infer_records: list[tuple[str, float, int]] = []
    # 分阶段耗时：True 强制开启；None 读 predict_cfg.predict_phase_profile
    PHASE_PROFILE: bool | None = True
    images_skipped = 0

    skip_if_output_exists = _validate_predict_all_resume_config(
        skip_if_output_exists=SKIP_IF_OUTPUT_EXISTS,
        clean_output_before_run=CLEAN_OUTPUT_BEFORE_RUN,
        output_xml=OUTPUT_XML,
    )

    ls_ingestor = None
    ls_local_dir: Path | None = None
    ls_ingest_total_tasks = 0
    ls_ingest_total_pushed = 0
    ls_export_total_saved = 0
    if ENABLE_LS_INGEST:
        if _ls_target_is_local_dir(LS_INGEST_URL):
            # 本地目录模式：不构造 ingestor、不联网，仅落盘分类框图
            ls_local_dir = Path(LS_INGEST_URL)
        else:
            ls_ingestor = build_ls_ingestor(
                ls_ingest_url=LS_INGEST_URL,
                choice_from_name=LS_INGEST_CHOICE_FROM_NAME,
                choice_to_name=LS_INGEST_CHOICE_TO_NAME,
                ingest_batch_size=LS_INGEST_BATCH_SIZE,
                jpeg_quality=LS_INGEST_JPEG_QUALITY,
            )

    save_image_scale = resolve_output_image_scale(SAVE_IMAGE_SCALE)
    if save_image_scale is not None:
        _scale_label = {0.5: "1/2", 0.25: "1/4", 0.125: "1/8"}.get(
            save_image_scale, str(save_image_scale)
        )
        logging.info("可视化结果图保存缩放: %s（绘制原分辨率，落盘前缩小）", _scale_label)

    pipeline = create_pipeline(
        CONFIG_PATH,
        device=None,
        root_ids=ROOT_IDS,
        enable_mask_rate_filter=ENABLE_MASK_RATE_FILTER,
    )
    phase_profile_on = (
        bool(PHASE_PROFILE)
        if PHASE_PROFILE is not None
        else bool(
            resolve_predict_cfg_value(
                "predict_phase_profile", pipeline.config, default=False
            )
        )
    )
    if phase_profile_on:
        pipeline._phase_recorder.enabled = True
        logging.info(
            "分阶段耗时统计已开启: detect / seg / cls-batch / post（predict_phase_profile）"
        )
    logging.info(
        "[推理性能] detect_seg_batch_size=%s cls_batch_size=%s "
        "clip_profiles_enable=%s parallel_detect_seg=%s trt_switch=%s",
        resolve_predict_cfg_value(
            "detect_seg_batch_size", pipeline.config, default=32
        ),
        resolve_predict_cfg_value("cls_batch_size", pipeline.config, default=32),
        resolve_predict_cfg_value(
            "clip_profiles_enable", pipeline.config, default=True
        ),
        resolve_predict_cfg_value(
            "parallel_detect_seg", pipeline.config, default=True
        ),
        resolve_predict_cfg_value("trt_switch", pipeline.config, default=False),
    )
    validation_focus = resolve_validation_focus_config(
        pipeline.config, root_ids=ROOT_IDS
    )
    if not ENABLE_MASK_RATE_FILTER:
        logging.info("mask_rate 填充率过滤已关闭（ENABLE_MASK_RATE_FILTER=False）")
    if ENABLE_VALIDATION:
        eval_label_alias_map = load_eval_label_alias_map(alg_config=pipeline.config)
        logging.info(
            "校验中文标注别名映射已加载，条目数=%d",
            len(eval_label_alias_map or {}),
        )
        logging.info(
            "无 VOC xml 时将从文件名推断 GT 类别，并按图内报出框逐框分类校验"
        )
    tier_equivalence = load_class_tier_equivalence(alg_config=pipeline.config)
    report_focus = (
        build_eval_focus_set(
            validation_focus.report_classes,
            merge=class_merge_eval,
            label_alias_map=eval_label_alias_map,
        )
        if validation_focus.report_classes
        else frozenset()
    )
    top1_focus = (
        build_eval_focus_set(
            validation_focus.top1_classes,
            merge=class_merge_eval,
            label_alias_map=eval_label_alias_map,
        )
        if validation_focus.top1_classes
        else frozenset()
    )
    top2_focus = (
        build_eval_focus_set(
            validation_focus.top2_classes,
            merge=class_merge_eval,
            label_alias_map=eval_label_alias_map,
        )
        if validation_focus.top2_classes
        else frozenset()
    )
    top3_focus = (
        build_eval_focus_set(
            validation_focus.top3_classes,
            merge=class_merge_eval,
            label_alias_map=eval_label_alias_map,
        )
        if validation_focus.top3_classes
        else frozenset()
    )
    competition_focus = resolve_competition_counting_focus(
        validation_focus,
        class_merge=class_merge_eval,
        label_alias_map=eval_label_alias_map,
    )
    post_executor: ThreadPoolExecutor | None = None
    post_shared: _DemoPostShared | None = None
    t_wall_all = time.perf_counter()
    try:
        if CLEAN_OUTPUT_BEFORE_RUN:
            _clear_run_output_dir(OUTPUT_DIR)
        if (
            ENABLE_LS_INGEST
            and ls_local_dir is not None
            and CLEAN_LS_EXPORT_BEFORE_RUN
        ):
            _clear_run_output_dir(str(ls_local_dir))
        if ENABLE_LS_INGEST and ls_local_dir is not None:
            ls_local_dir.mkdir(parents=True, exist_ok=True)
        print(f"性能测试: 共 {len(image_files)} 张图片，目录={INPUT_PATH}")
        print(f"配置: {resolve_effective_insect_alg_path(CONFIG_PATH)}")
        if pipeline.run_model and pipeline.config_path:
            print(
                f"run_model: {pipeline.run_model} "
                f"(启动配置 {pipeline.config_path.name})"
            )
        print(f"根模型: {list(pipeline._roots.keys())}")
        if pipeline._postprocess_debug:
            print(
                "后处理 debug: 已开启 in_big 拼图 -> "
                f"{resolve_eval_metrics_debug_dir(OUTPUT_DIR)}"
            )
            if skip_if_output_exists:
                print(
                    "后处理 debug: 增量续跑跳过已关闭（将重新推理以生成 debug 图）"
                )
                skip_if_output_exists = False
        post_cfg = _postprocess_block(pipeline.config)
        if bool(post_cfg.get("report_all_switch", False)):
            print("报出模式: 全部报出 (report_all_switch=true)")
        else:
            print(
                "报出模式: top1+top2+top3+other (background 不报)"
                f"，允许 {len(pipeline._report_allowed or ())} 类"
            )
        print(f"共 {len(image_files)} 张图片")
        if not ENABLE_MASK_RATE_FILTER:
            print("mask_rate 填充率过滤: 已关闭")
        if not SAVE_IMAGE and not OUTPUT_XML and not ENABLE_VALIDATION:
            print("性能模式: 仅推理计时，不保存结果/不做校验")
        if OUTPUT_XML:
            print("已开启 Pascal VOC bbox xml 输出")
            if OUTPUT_XML_CN_NAME:
                print("  xml object name: 中文展示名（OUTPUT_XML_CN_NAME=True）")
            if OUTPUT_XML_DUAL_CONF:
                print("  xml object 附加 det_conf / cls_conf（OUTPUT_XML_DUAL_CONF=True）")
        if OUTPUT_LABELME:
            print(
                f"已开启 LabelMe JSON 输出（copy_image={LABELME_COPY_IMAGE}）；"
                f"校对: labelme {OUTPUT_DIR}"
            )
        if ENABLE_VALIDATION:
            print(
                f"已开启结果校验: metric={VAL_BOX_MATCH_METRIC} thr={VAL_GEOM_THRESHOLD}"
            )
            print(
                f"北京比赛计分: run_model={validation_focus.run_model} "
                f"({report_mode_label(run_model=validation_focus.run_model)})，"
                f"计分类 {len(competition_focus)} 类"
            )
            if validation_focus.run_model == "baipai":
                print(
                    "  摆拍误报矫正: MAX(0,平均-(额外非标样数/标样总数)×100)，"
                    "含 gt=0 多检与 background/report_tier 过滤框"
                )
            elif validation_focus.run_model == "shengchan":
                print(
                    "  生产计分: 配置报出各类(top1+top2+top3+other)计数准确率算术平均，"
                    "鉴定0识别≥1按0计"
                )
            if validation_focus.report_classes:
                print(
                    f"报出类清单: {len(validation_focus.report_classes)} 类 "
                    f"(top1={len(validation_focus.top1_classes)}, "
                    f"top2={len(validation_focus.top2_classes)}, "
                    f"top3={len(validation_focus.top3_classes)}, "
                    f"other={len(validation_focus.other_classes)}, "
                    f"bg={len(validation_focus.background_classes)})"
                )
        if ENABLE_LS_INGEST and ls_ingestor is not None:
            print(
                f"已开启分类框图上报 LS: url={ls_ingestor.ingest_url} "
                f"dry_run={LS_INGEST_DRY_RUN} include_filtered={LS_INGEST_INCLUDE_FILTERED}"
            )
        if ENABLE_LS_INGEST and ls_local_dir is not None:
            _export_fmt = (
                "jpg(bbox)"
                if LS_INGEST_CROP_FROM_BBOX
                else ("png+alpha" if LS_EXPORT_SEG_TRANSPARENT_PNG else "jpg(白底)")
            )
            print(
                f"已开启分类框图本地导出: dir={ls_local_dir} "
                f"crop={'bbox' if LS_INGEST_CROP_FROM_BBOX else 'polygon'} "
                f"fmt={_export_fmt} include_filtered={LS_INGEST_INCLUDE_FILTERED}"
            )
        if POSTPROCESS_PIPELINE:
            _pp_workers = max(1, int(POSTPROCESS_WORKERS))
            print(
                f"后处理流水线: 已开启 workers={_pp_workers} "
                f"（推理与校验/画图/导出并行）"
            )
        if skip_if_output_exists:
            print(
                "增量续跑: 已开启 SKIP_IF_OUTPUT_EXISTS "
                "（预测 xml 存在即跳过推理；"
                f"{'结果图按 xml 重绘；' if SAVE_IMAGE and DRAW_PICTURE_AGAING else ''}"
                f"{'结果图仅缺失时绘制；' if SAVE_IMAGE and not DRAW_PICTURE_AGAING else ''}"
                f"{'eval_metrics 完成戳' if ENABLE_VALIDATION else '无 GT 校验戳'}；"
                "含 det_conf/cls_conf 的 xml 将按当前算法配置重算校验）"
            )

        draw_filter_sources = collect_draw_filter_model_sources(pipeline.config)
        if draw_filter_sources:
            logging.info(
                "filtered 绘测已启用根模型: %s",
                ", ".join(sorted(draw_filter_sources)),
            )

        post_cfg = _DemoPostConfig(
            enable_validation=ENABLE_VALIDATION,
            val_box_match_metric=VAL_BOX_MATCH_METRIC,
            val_geom_threshold=VAL_GEOM_THRESHOLD,
            class_merge_eval=class_merge_eval,
            cls_merge_type_index=cls_merge_type_index,
            eval_label_alias_map=eval_label_alias_map,
            tier_equivalence=tier_equivalence,
            eval_fuzzy_only_wildcard=eval_fuzzy_only_wildcard,
            save_type_confusion_crops=SAVE_TYPE_CONFUSION_CROPS,
            standard_eval_subdir=STANDARD_EVAL_SUBDIR,
            output_dir=OUTPUT_DIR,
            save_image=SAVE_IMAGE,
            draw_picture_again=DRAW_PICTURE_AGAING,
            draw_bbox=DRAW_BBOX,
            draw_polygon=DRAW_POLYGON,
            draw_filter_sources=draw_filter_sources,
            draw_validation_focus=DRAW_VALIDATION_FOCUS,
            draw_eval_fp_non_focus=DRAW_EVAL_FP_NON_FOCUS,
            draw_label=DRAW_LABEL,
            draw_label_errors_only=DRAW_LABEL_ERRORS_ONLY,
            report_focus=report_focus,
            label_font_size=LABEL_FONT_SIZE,
            show_source_in_label=SHOW_SOURCE_IN_LABEL,
            save_image_scale=save_image_scale,
            cn_index=dict(pipeline._cn_display_index),
            output_xml=OUTPUT_XML,
            output_xml_cn_name=OUTPUT_XML_CN_NAME,
            output_xml_dual_conf=OUTPUT_XML_DUAL_CONF,
            output_labelme=OUTPUT_LABELME,
            labelme_copy_image=LABELME_COPY_IMAGE,
            enable_ls_ingest=ENABLE_LS_INGEST,
            ls_ingestor=ls_ingestor,
            ls_local_dir=ls_local_dir,
            ls_ingest_include_filtered=LS_INGEST_INCLUDE_FILTERED,
            ls_ingest_skip_filtered=LS_INGEST_SKIP_FILTERED,
            ls_ingest_skip_other=LS_INGEST_SKIP_OTHER,
            ls_ingest_crop_from_bbox=LS_INGEST_CROP_FROM_BBOX,
            ls_ingest_pad_square=LS_INGEST_PAD_SQUARE,
            ls_ingest_dry_run=LS_INGEST_DRY_RUN,
            ls_ingest_jpeg_quality=LS_INGEST_JPEG_QUALITY,
            ls_export_seg_transparent_png=LS_EXPORT_SEG_TRANSPARENT_PNG,
            perf_quiet=PERF_QUIET,
        )
        post_shared = _DemoPostShared(
            stat_total=stat_total,
            stat_by_cls=stat_by_cls,
            std_cm=std_cm,
            use_lock=POSTPROCESS_PIPELINE,
        )
        if POSTPROCESS_PIPELINE:
            post_executor = ThreadPoolExecutor(
                max_workers=max(1, int(POSTPROCESS_WORKERS)),
                thread_name_prefix="demo-post",
            )

        for idx, img_path in enumerate(image_files, 1):
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[{idx}/{len(image_files)}] 跳过无法读取: {img_path}")
                continue

            if input_p.is_dir():
                rel = img_path.relative_to(input_p)
            else:
                rel = Path(img_path.name)
            save_path = Path(OUTPUT_DIR) / rel
            out_dir = save_path.parent

            has_validation_gt = (
                _predict_all_has_validation_gt(
                    img_path, rel, label_alias_map=eval_label_alias_map
                )
                if ENABLE_VALIDATION
                else False
            )
            if skip_if_output_exists and predict_all_image_complete_for_skip(
                output_dir=OUTPUT_DIR,
                eval_subdir=STANDARD_EVAL_SUBDIR,
                rel=rel,
                out_dir=out_dir,
                has_validation_gt=has_validation_gt,
                enable_validation=ENABLE_VALIDATION,
            ):
                pred_xml = out_dir / f"{rel.stem}.xml"
                images_skipped += 1
                try:
                    results, filtered = _load_resume_results_from_pred_xml(
                        pred_xml,
                        pipeline,
                        label_alias_map=eval_label_alias_map,
                    )
                except Exception as e:
                    logging.warning(
                        "续跑跳过推理：读取预测 xml 失败 %s: %s", pred_xml, e
                    )
                    print(f"[{idx}/{len(image_files)}] 续跑读 xml 失败，改走推理: {rel}")
                else:
                    job = _DemoImageJob(
                        idx=idx,
                        total=len(image_files),
                        rel=rel,
                        img=img.copy(),
                        img_path=img_path,
                        results=[dict(r) for r in results],
                        filtered=[dict(r) for r in filtered],
                        infer_cost=0.0,
                        save_path=save_path,
                        out_dir=out_dir,
                        skip_inference=True,
                    )
                    if post_executor is not None:
                        post_executor.submit(post_shared.postprocess, post_cfg, job)
                    else:
                        post_shared.postprocess(post_cfg, job)
                    continue

            _need_filtered = (
                bool(draw_filter_sources)
                or (ENABLE_VALIDATION and DRAW_VALIDATION_FOCUS)
                or (
                    ENABLE_VALIDATION
                    and validation_focus.run_model == "baipai"
                )
                or (ENABLE_LS_INGEST and LS_INGEST_INCLUDE_FILTERED)
            )
            _t_infer = time.perf_counter()
            _predict_kw: dict[str, Any] = {
                "source_image_stem": rel.stem,
                "result_output_dir": str(out_dir),
                "eval_metrics_root": OUTPUT_DIR,
                "phase_profile": phase_profile_on,
            }
            if _need_filtered:
                results, filtered = pipeline.predict(
                    img,
                    collect_filtered=True,
                    **_predict_kw,
                )
            else:
                results = pipeline.predict(img, **_predict_kw)
                filtered = []
            _infer_cost = time.perf_counter() - _t_infer
            infer_times.append(_infer_cost)
            infer_records.append((rel.name, _infer_cost, len(results)))
            _phase_line = format_predict_phase_line(pipeline._phase_recorder.last_sample())

            job = _DemoImageJob(
                idx=idx,
                total=len(image_files),
                rel=rel,
                img=img.copy(),
                img_path=img_path,
                results=[dict(r) for r in results],
                filtered=[dict(r) for r in filtered],
                infer_cost=_infer_cost,
                save_path=save_path,
                out_dir=out_dir,
                phase_line=_phase_line,
            )
            if post_executor is not None:
                post_executor.submit(post_shared.postprocess, post_cfg, job)
            else:
                post_shared.postprocess(post_cfg, job)
    finally:
        if post_executor is not None and post_shared is not None:
            post_executor.shutdown(wait=True)
            ls_ingest_total_tasks = post_shared.ls_ingest_total_tasks
            ls_ingest_total_pushed = post_shared.ls_ingest_total_pushed
            ls_export_total_saved = post_shared.ls_export_total_saved
        pipeline.release()

    if ENABLE_VALIDATION and (
        stat_total["img_with_xml"] > 0 or stat_total["img_with_filename_gt"] > 0
    ):
        print(
            "======== predict_all 验证汇总"
            f"（xml={stat_total['img_with_xml']} 张，"
            f"文件名GT={stat_total['img_with_filename_gt']} 张"
            f"{'，文件名GT按报出框分类' if stat_total['img_with_filename_gt'] else ''}） ========"
        )
        _print_overall_stat_summary("统一管线验证汇总", stat_total)
        if post_shared is not None:
            _print_other_filter_reason_stats(post_shared.stat_other_filter)
        eval_class_display_index = build_eval_class_display_index(
            alg_config=pipeline.config
        )
        stat_by_cls_merged = merge_stat_by_cls(
            dict(stat_by_cls),
            merge=class_merge_eval,
            label_alias_map=eval_label_alias_map,
            tier_equivalence=tier_equivalence,
        )
        _print_stat_by_cls(
            "按合并类统计",
            stat_by_cls_merged,
            sort_by_acc=SORT_STAT_BY_ACC,
            class_display_index=eval_class_display_index,
            focus=report_focus or None,
        )
        _print_overall_stat_summary(
            "一级重点关注(top1)汇总",
            sum_stat_by_cls_focus(stat_by_cls_merged, top1_focus),
        )
        _print_stat_by_cls(
            "一级重点关注(top1)按类统计",
            stat_by_cls_merged,
            sort_by_acc=SORT_STAT_BY_ACC,
            class_display_index=eval_class_display_index,
            focus=top1_focus or None,
        )
        _print_overall_stat_summary(
            "二级重点关注(top2)汇总",
            sum_stat_by_cls_focus(stat_by_cls_merged, top2_focus),
        )
        _print_stat_by_cls(
            "二级重点关注(top2)按类统计",
            stat_by_cls_merged,
            sort_by_acc=SORT_STAT_BY_ACC,
            class_display_index=eval_class_display_index,
            focus=top2_focus or None,
        )
        _print_overall_stat_summary(
            "三级重点关注(top3)汇总",
            sum_stat_by_cls_focus(stat_by_cls_merged, top3_focus),
        )
        _print_stat_by_cls(
            "三级重点关注(top3)按类统计",
            stat_by_cls_merged,
            sort_by_acc=SORT_STAT_BY_ACC,
            class_display_index=eval_class_display_index,
            focus=top3_focus or None,
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
        if ENABLE_STANDARD_EVAL and std_cm is not None:
            eval_root = os.path.join(OUTPUT_DIR, STANDARD_EVAL_SUBDIR)
            _std_eval_save_branch(
                eval_root,
                "all",
                dict(std_cm),
                stat_total,
                {
                    "source": "predict_all.py",
                    "match_metric": VAL_BOX_MATCH_METRIC,
                    "geom_threshold": VAL_GEOM_THRESHOLD,
                    "insect_wildcard_strict": bool(
                        ENABLE_VALIDATION and eval_fuzzy_only_wildcard
                    ),
                },
            )
            _export_overall_summary_csv(eval_root, "all", stat_total)
            _export_stat_by_cls_csv(
                eval_root,
                "all",
                stat_by_cls_merged,
                sort_by_acc=SORT_STAT_BY_ACC,
                class_display_index=eval_class_display_index,
                focus=report_focus or None,
            )
            _export_overall_summary_csv(
                eval_root,
                "top1",
                sum_stat_by_cls_focus(stat_by_cls_merged, top1_focus),
            )
            _export_stat_by_cls_csv(
                eval_root,
                "top1",
                stat_by_cls_merged,
                sort_by_acc=SORT_STAT_BY_ACC,
                class_display_index=eval_class_display_index,
                focus=top1_focus or None,
            )
            _export_overall_summary_csv(
                eval_root,
                "top2",
                sum_stat_by_cls_focus(stat_by_cls_merged, top2_focus),
            )
            _export_stat_by_cls_csv(
                eval_root,
                "top2",
                stat_by_cls_merged,
                sort_by_acc=SORT_STAT_BY_ACC,
                class_display_index=eval_class_display_index,
                focus=top2_focus or None,
            )
            _export_overall_summary_csv(
                eval_root,
                "top3",
                sum_stat_by_cls_focus(stat_by_cls_merged, top3_focus),
            )
            _export_stat_by_cls_csv(
                eval_root,
                "top3",
                stat_by_cls_merged,
                sort_by_acc=SORT_STAT_BY_ACC,
                class_display_index=eval_class_display_index,
                focus=top3_focus or None,
            )
            print(f"标准评估输出目录: {eval_root}")
            if SAVE_TYPE_CONFUSION_CROPS:
                print(
                    f"类型混淆/漏检切图: {eval_root}/all/type_confusion_crops/ "
                    f"与 .../type_confusion_crops_by_pred/（漏检 pred=__miss__）"
                )
    elif ENABLE_VALIDATION:
        print("校验已开启，但输入集中无有效标注 xml 且文件名未能解析 GT，未生成统计")

    if ENABLE_LS_INGEST and ls_ingestor is not None:
        print(
            f"======== LS 上报汇总 ======== 组装任务={ls_ingest_total_tasks} "
            f"推送={ls_ingest_total_pushed} "
            + ("(dry_run)" if LS_INGEST_DRY_RUN else "")
        )
    if ENABLE_LS_INGEST and ls_local_dir is not None:
        print(
            f"======== 本地导出汇总 ======== 分类框图={ls_export_total_saved} "
            f"目录={ls_local_dir}"
        )

    if images_skipped > 0:
        print(f"======== 增量续跑 ======== 跳过已完整处理图片数={images_skipped}")

    if infer_times:
        _total_cost = sum(infer_times)
        print("======== 推理耗时统计（pipeline.predict，含预处理/合并/路由） ========")
        print(
            f"图片数={len(infer_times)} 总耗时={_total_cost:.3f}s "
            f"平均={_total_cost / len(infer_times):.3f}s "
            f"最大={max(infer_times):.3f}s 最小={min(infer_times):.3f}s"
        )
        if phase_profile_on:
            print_predict_phase_summary(
                pipeline._phase_recorder,
                image_count=len(infer_times),
            )
        print_xingneng_infer_summary(
            infer_records,
            wall_elapsed=time.perf_counter() - t_wall_all,
        )

    print(f"完成，性能测试结束（无结果落盘时 OUTPUT_DIR={OUTPUT_DIR} 可忽略）")
