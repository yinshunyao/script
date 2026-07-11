#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Detail  : 分割 + 可选分类推理（独立于 PredictSize）。
#           阈值：script/insect_alg_seg.json（detect_conf / cls_conf）。
#           分类裁剪：默认 polygon 掩码抠图；可选 cls_crop_from_bbox 改为 bbox 矩形截取。

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

_FILE = Path(__file__).resolve()
_ROOT = _FILE.parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from script.predict.model_cls_factory import ClsModel, create_classifier
from script.predict.model_seg import ModelSegmenter
from script.predict_seg_lib import (
    DEFAULT_INSECT_ALG_SEG_JSON,
    INSECT_ALG_SEG_JSON_REL,
    crop_instance_bgr_from_bbox,
    crop_instance_bgr_from_polygon,
    load_insect_alg,
    resolve_cls_crop_background,
    resolve_cls_pad_color,
    resolve_cls_top1_threshold,
    resolve_insect_alg_seg_path,
    resolve_seg_detect_conf,
    should_classify_seg_instance,
)

DEFAULT_INSECT_ALG_SEG_JSON = DEFAULT_INSECT_ALG_SEG_JSON  # re-export
INSECT_ALG_SEG_JSON_REL = INSECT_ALG_SEG_JSON_REL  # re-export


class PredictSeg:
    """
    分割推理 + 可选逐实例分类。

    流程：
        1. ModelSegmenter 滑窗/整图分割
        2. 对每个实例按 polygon 掩码或 bbox 矩形裁剪（可选 pad_square / gray_binarize 后送分类）
        3. 分类门限来自 insect_alg_seg.json 的 cls_conf（按预测类名）或 predict 传入的全局值
    """

    def __init__(
        self,
        seg_model_path: str,
        *,
        cls_model_path: str | None = None,
        cls_list: list[str] | None = None,
        conf_thresh: float = 0.25,
        conf_merge: float = 0.1,
        conf_merge_draw: float = 0.01,
        ior_threshold: float = 0.5,
        device: str | None = None,
        augment: bool = False,
        cls_pad_square: bool = False,
        cls_gray_binarize: bool = False,
        cls_to_gray: bool = False,
        # None 表示不加载 insect_alg_seg.json（仅使用参数与模型默认值；不产生“文件不存在”警告）
        insect_alg: dict[str, Any] | None = None,
        insect_alg_path: str | Path | None = None,
        insect_alg_profile: str | None = None,
        seg_imgsz: int = 0,
        nms_iou: float | None = None,
        max_det: int | None = None,
        nms_agnostic: bool | None = None,
        retina_masks: bool = False,
        poly_merge: bool = False,
        poly_merge_edge_px: float = 5.0,
        poly_merge_edge_ratio: float = 0.5,
        poly_merge_contain_ratio: float = 0.7,
        poly_merge_cross_class: bool = True,
        poly_merge_max_points: int = 80,
        roi_disk_edge_hug_ratio: float = 0.55,
        roi_disk_edge_outside_min: float = 0.02,
        filter_rows_by_roi_boundary: bool = True,
        crop_pad_ratio: float = 0.05,
        cls_crop_from_bbox: bool = False,
        cls_crop_background: Any = None,
        cls_pad_color: Any = None,
        cls_backend: str | None = None,
        timm_model: str | None = None,
        image_size: int | None = None,
        log_prefix: str | None = None,
        cls_deferred: bool = False,
        gray_contrast_enhance: bool = False,
        gray_clahe_clip: float = 2.0,
        gray_clahe_tile: int = 8,
        gray_contrast_debug_save: bool = False,
    ):
        self.cls_list = list(cls_list) if cls_list else None
        self.cls_pad_square = bool(cls_pad_square)
        self.cls_gray_binarize = bool(cls_gray_binarize)
        self.cls_to_gray = bool(cls_to_gray)
        self.crop_pad_ratio = float(crop_pad_ratio)
        self.cls_crop_from_bbox = bool(cls_crop_from_bbox)
        self.cls_crop_background = resolve_cls_crop_background(cls_crop_background)
        self.cls_pad_color = resolve_cls_pad_color(cls_pad_color)
        self._log_prefix = (log_prefix or "").strip() or None

        self._insect_alg: dict[str, Any] | None = None
        if insect_alg is not None:
            self._insect_alg = insect_alg
        elif insect_alg_path is not None:
            p = resolve_insect_alg_seg_path(insect_alg_path)
            self.insect_alg_seg_path = p
            if p.is_file():
                with open(p, "r", encoding="utf-8") as f:
                    self._insect_alg = json.load(f)
            else:
                logging.warning("insect_alg_seg 配置文件不存在，忽略: %s", p)
        else:
            self.insect_alg_seg_path = None

        effective_conf = resolve_seg_detect_conf(
            self._insect_alg,
            conf_thresh,
            insect_alg_profile=insect_alg_profile,
            cls_list=self.cls_list,
        )
        if self._insect_alg and effective_conf != conf_thresh:
            logging.info(
                "分割置信度: insect_alg_seg detect_conf=%.4f (全局 conf_thresh=%.4f)",
                effective_conf,
                conf_thresh,
            )

        self.segmenter = ModelSegmenter(
            model_path=seg_model_path,
            conf_thresh=effective_conf,
            conf_merge=conf_merge,
            conf_merge_draw=conf_merge_draw,
            ior_threshold=ior_threshold,
            device=device,
            augment=augment,
            nms_iou=nms_iou,
            max_det=max_det,
            nms_agnostic=nms_agnostic,
            imgsz=seg_imgsz,
            retina_masks=retina_masks,
            poly_merge=poly_merge,
            poly_merge_edge_px=poly_merge_edge_px,
            poly_merge_edge_ratio=poly_merge_edge_ratio,
            poly_merge_contain_ratio=poly_merge_contain_ratio,
            poly_merge_cross_class=poly_merge_cross_class,
            poly_merge_max_points=poly_merge_max_points,
            roi_disk_edge_hug_ratio=roi_disk_edge_hug_ratio,
            roi_disk_edge_outside_min=roi_disk_edge_outside_min,
            filter_rows_by_roi_boundary=filter_rows_by_roi_boundary,
            gray_contrast_enhance=gray_contrast_enhance,
            gray_clahe_clip=gray_clahe_clip,
            gray_clahe_tile=gray_clahe_tile,
            gray_contrast_debug_save=gray_contrast_debug_save,
        )

        _pfx = f"[{self._log_prefix}] " if self._log_prefix else ""
        if nms_iou is not None:
            logging.info("%s分割 NMS iou=%.4f", _pfx, float(nms_iou))
        if cls_model_path:
            cls_cfg: dict[str, Any] = {}
            if cls_backend:
                cls_cfg["cls_backend"] = cls_backend
            if timm_model:
                cls_cfg["timm_model"] = timm_model
            if image_size is not None and int(image_size) > 0:
                cls_cfg["image_size"] = int(image_size)
            self.classifier: ClsModel | None = create_classifier(
                str(cls_model_path),
                device=device,
                pad_square=cls_pad_square,
                gray_binarize=cls_gray_binarize,
                pad_color_bgr=self.cls_pad_color,
                to_gray=cls_to_gray,
                cfg=cls_cfg or None,
            )
            logging.info(
                "%s已加载内嵌分类模型: %s backend=%s",
                _pfx,
                cls_model_path,
                cls_cfg.get("cls_backend", "auto"),
            )
        else:
            self.classifier = None
            if cls_deferred:
                logging.info(
                    "%sPredictSeg 不内嵌分类模型（预期）：输出保留分割类别与 seg_conf，"
                    "细分类由外部 out/models.cls 路由承担",
                    _pfx,
                )
            else:
                logging.info(
                    "%s未配置分类模型，输出保留分割类别与 seg_conf",
                    _pfx,
                )

    def _cls_top1_threshold(self, predicted_cls_name: str, cls_top1_conf_threshold: float | None):
        return resolve_cls_top1_threshold(
            self._insect_alg, predicted_cls_name, cls_top1_conf_threshold
        )

    def apply_classification(
        self,
        image_bgr,
        rows: list[dict[str, Any]],
        *,
        cls_top1_conf_threshold: float | None = None,
        cls_pad_square: bool | None = None,
        cls_gray_binarize: bool | None = None,
        cls_to_gray: bool | None = None,
        crop_pad_ratio: float | None = None,
        cls_crop_from_bbox: bool | None = None,
        cls_crop_background: Any = None,
        cls_pad_color: Any = None,
    ) -> list[dict[str, Any]]:
        """对已有分割结果逐实例分类（原地更新 rows 并返回）。"""
        if self.classifier is None or not rows:
            out_none: list[dict[str, Any]] = []
            for r in rows:
                det = dict(r)
                seg_nm = str(det.get("class_name", det.get("seg_cls_name", "")) or "")
                det["seg_cls_name"] = seg_nm
                det["seg_conf"] = float(det.get("conf", 0.0) or 0.0)
                det["cls_name"] = seg_nm
                det["cls_conf"] = det["seg_conf"]
                one = [{"class_name": seg_nm, "conf": det["seg_conf"]}]
                det["cls_topk"] = one
                det["cls_top3"] = one
                out_none.append(det)
            return out_none

        pad_sq = self.cls_pad_square if cls_pad_square is None else bool(cls_pad_square)
        gray_bin = self.cls_gray_binarize if cls_gray_binarize is None else bool(cls_gray_binarize)
        gray = self.cls_to_gray if cls_to_gray is None else bool(cls_to_gray)
        pad_ratio = self.crop_pad_ratio if crop_pad_ratio is None else float(crop_pad_ratio)
        crop_bbox = (
            self.cls_crop_from_bbox
            if cls_crop_from_bbox is None
            else bool(cls_crop_from_bbox)
        )
        mask_bg = (
            self.cls_crop_background
            if cls_crop_background is None
            else resolve_cls_crop_background(cls_crop_background)
        )
        pad_clr = (
            self.cls_pad_color
            if cls_pad_color is None
            else resolve_cls_pad_color(cls_pad_color)
        )
        dev = getattr(self.segmenter, "device", None)

        out: list[dict[str, Any]] = []
        for raw in rows:
            det = dict(raw)
            seg_nm = str(det.get("class_name", det.get("seg_cls_name", "")) or "")
            det["seg_cls_name"] = seg_nm
            det["seg_conf"] = float(det.get("conf", 0.0) or 0.0)

            if not should_classify_seg_instance(seg_nm, self.cls_list):
                det["cls_name"] = seg_nm
                det["cls_conf"] = det["seg_conf"]
                det.setdefault("cls_topk", [{"class_name": seg_nm, "conf": det["seg_conf"]}])
                det["cls_top3"] = det["cls_topk"][:3]
                out.append(det)
                continue

            poly = list(det.get("polygon") or [])
            if crop_bbox:
                crop = crop_instance_bgr_from_bbox(image_bgr, det)
            else:
                crop = crop_instance_bgr_from_polygon(
                    image_bgr,
                    poly,
                    pad_ratio=pad_ratio,
                    background_bgr=mask_bg,
                )
            if crop is None or crop.size == 0:
                det["cls_name"] = "other"
                det["cls_conf"] = 0.0
                det["cls_topk"] = []
                det["cls_top3"] = []
                det["filter"] = True
                out.append(det)
                continue

            cls_result = self.classifier.predict(
                crop,
                device=dev,
                pad_square=pad_sq,
                gray_binarize=gray_bin,
                pad_color_bgr=pad_clr,
                to_gray=gray,
            )
            if cls_result is not None:
                det["cls_name"] = cls_result["class_name"]
                det["cls_conf"] = cls_result["conf"]
                _tk = cls_result.get("topk") or cls_result.get("top3") or []
                det["cls_topk"] = list(_tk)
                det["cls_top3"] = det["cls_topk"][:3]
                cls_thr = self._cls_top1_threshold(
                    det["cls_name"], cls_top1_conf_threshold
                )
                if cls_thr is not None and det["cls_conf"] <= cls_thr:
                    _nm = str(det.get("cls_name") or "").strip()
                    if _nm and _nm.lower() != "other":
                        det["cls_name_top1"] = _nm
                    det["cls_name"] = "other"
            else:
                det["cls_name"] = "other"
                det["cls_conf"] = 0.0
                det["cls_topk"] = []
                det["cls_top3"] = []

            out.append(det)
        return out

    def predict(
        self,
        image_bgr,
        *,
        clip_size: int = 960,
        overlap_size: int = 200,
        clip_profiles: list[ClipProfile] | None = None,
        clip_start: int = 0,
        clip_batch_size: int = 1,
        padding: bool = True,
        pad_full_image_to_square: bool = False,
        min_size: int = 3,
        max_size: int | None = None,
        imgsz: int | None = None,
        nms_iou: float | None = None,
        max_det: int | None = None,
        nms_agnostic: bool | None = None,
        retina_masks: bool | None = None,
        cls_top1_conf_threshold: float | None = None,
        cls_pad_square: bool | None = None,
        cls_gray_binarize: bool | None = None,
        cls_to_gray: bool | None = None,
        cls_crop_from_bbox: bool | None = None,
        cls_crop_background: Any = None,
        cls_pad_color: Any = None,
        debug: bool = False,
        debug_clip: bool = False,
        return_all_rows: bool = False,
        debug_image_stem: str | None = None,
        gray_contrast_debug_dir: str | None = None,
        roi_circle: tuple[int, int, int] | None = None,
    ) -> list[dict[str, Any]]:
        """
        分割 + 分类一站式推理。

        :param return_all_rows: True 时包含 filter=True 的实例（与 PredictSize.return_full_final 类似）
        """
        rows = self.segmenter.predict(
            image_bgr,
            clip_size=clip_size,
            overlap_size=overlap_size,
            clip_profiles=clip_profiles,
            clip_start=clip_start,
            clip_batch_size=clip_batch_size,
            padding=padding,
            pad_full_image_to_square=pad_full_image_to_square,
            min_size=min_size,
            max_size=max_size,
            imgsz=imgsz,
            nms_iou=nms_iou,
            max_det=max_det,
            nms_agnostic=nms_agnostic,
            retina_masks=retina_masks,
            debug=debug,
            debug_clip=debug_clip,
            return_all_rows=return_all_rows,
            debug_image_stem=debug_image_stem,
            gray_contrast_debug_dir=gray_contrast_debug_dir,
            roi_circle=roi_circle,
        )
        rows = self.apply_classification(
            image_bgr,
            rows,
            cls_top1_conf_threshold=cls_top1_conf_threshold,
            cls_pad_square=cls_pad_square,
            cls_gray_binarize=cls_gray_binarize,
            cls_to_gray=cls_to_gray,
            cls_crop_from_bbox=cls_crop_from_bbox,
            cls_crop_background=cls_crop_background,
            cls_pad_color=cls_pad_color,
        )
        if return_all_rows:
            return rows
        return [r for r in rows if not r.get("filter")]

    def release(self) -> None:
        if getattr(self, "segmenter", None) is not None:
            del self.segmenter.model
            self.segmenter = None
        if getattr(self, "classifier", None) is not None:
            del self.classifier.model
            self.classifier = None
        logging.info("PredictSeg 模型资源已释放")
