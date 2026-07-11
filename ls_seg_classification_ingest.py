#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将昆虫分割+分类结果以「图像分类」任务推送到 LS 标注系统数据接口（api_interface）。

分割推理与分类裁剪逻辑对齐 ``predict_seg_validate.py``（``PredictSeg``；``cls_crop_from_bbox`` 控制分类输入裁剪方式）；
上报格式与 ``ls_classification_ingest.py`` 一致（``image_classification`` + ``native_json``）。

典型用法::

    from script.predict_seg import PredictSeg
    from script.ls_seg_classification_ingest import LsSegClassificationIngestor, ingest_predict_seg_results

    predictor = PredictSeg(seg_model_path=..., cls_model_path=...)
    ingestor = LsSegClassificationIngestor(
        ingest_url="http://127.0.0.1:8080/api/data-sources/ingest/{ingest_token}/",
    )
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        rows = predictor.predict(img, ...)
        ingest_predict_seg_results(ingestor, rows, img, source_image_name=img_path.name)
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import cv2
import numpy as np

_FILE = Path(__file__).resolve()
_ROOT = _FILE.parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from script.ls_classification_ingest import (
    LsClassificationIngestor,
    _safe_slug,
    build_image_classification_task,
    cls_name_to_zh,
    crop_box,
    encode_crop_jpeg_base64,
    is_other_cls_name,
    normalize_cls_name_for_ingest,
    resolve_ingest_pad_square,
    resolve_ls_ingest_url,
)
from script.predict.model_cls import ModelCls
from script.predict_seg import INSECT_ALG_SEG_JSON_REL, PredictSeg
from script.predict_seg_lib import (
    bbox_from_row,
    collect_images,
    crop_instance_bgr_from_polygon,
    filter_rows_by_bbox_diag_range,
    load_insect_alg,
    resolve_seg_dia_range,
)

logger = logging.getLogger(__name__)

LsSegClassificationIngestor = LsClassificationIngestor


def crop_instance_for_ingest(
    image: np.ndarray,
    row: Mapping[str, Any],
    *,
    ingest_crop_from_bbox: bool = True,
    pad_ratio: float = 0.05,
    pad_square: bool = False,
) -> np.ndarray:
    """
    为上报 LS 裁剪虫体图。

    - ``ingest_crop_from_bbox=True``（默认）：按实例外接框从大图直接矩形截取。
    - ``ingest_crop_from_bbox=False``：按分割多边形掩码裁剪（白底）。
    - ``pad_square``：默认 False（bbox 直裁无白边）；由 ``ingest_match_inference_crop`` 打开时可与推理 ``cls_pad_square`` 一致。
    """
    if image is None or image.size == 0:
        raise ValueError("image 为空")
    x1, y1, x2, y2 = bbox_from_row(row)
    if ingest_crop_from_bbox:
        crop = crop_box(image, x1, y1, x2, y2, pad_square=pad_square)
        return crop

    poly = list(row.get("polygon") or [])
    crop = crop_instance_bgr_from_polygon(
        image,
        poly,
        pad_ratio=pad_ratio,
        background_bgr=(255, 255, 255),
    )
    if crop is None or crop.size == 0:
        raise ValueError("polygon 裁剪为空")
    if pad_square:
        crop = ModelCls.pad_bgr_to_square(crop)
    return crop


def crop_seg_instance(
    image: np.ndarray,
    row: Mapping[str, Any],
    *,
    pad_ratio: float = 0.05,
    pad_square: bool = False,
) -> np.ndarray:
    """按分割多边形掩码裁剪实例（兼容旧调用）。"""
    return crop_instance_for_ingest(
        image,
        row,
        ingest_crop_from_bbox=False,
        pad_ratio=pad_ratio,
        pad_square=pad_square,
    )


def seg_row_to_task(
    ingestor: LsClassificationIngestor,
    image: np.ndarray,
    row: Mapping[str, Any],
    *,
    source_image_name: str,
    instance_index: int,
    ingest_crop_from_bbox: bool = True,
    pad_ratio: float = 0.05,
    pad_square: bool = False,
    choice_from_name: Optional[str] = None,
    choice_to_name: Optional[str] = None,
) -> dict[str, Any]:
    """单条 ``PredictSeg`` 实例 → 掩码裁剪、Base64 编码并组装 LS task。"""
    x1, y1, x2, y2 = bbox_from_row(row)
    cls_name_raw = str(row.get("cls_name") or row.get("class_name") or "unknown")
    cls_name = normalize_cls_name_for_ingest(cls_name_raw)
    cls_conf = float(row.get("cls_conf", row.get("conf", 0.0)) or 0.0)
    seg_cls_name = str(row.get("seg_cls_name") or row.get("class_name") or "")
    seg_conf = float(row.get("seg_conf", row.get("conf", 0.0)) or 0.0)

    crop = crop_instance_for_ingest(
        image,
        row,
        ingest_crop_from_bbox=ingest_crop_from_bbox,
        pad_ratio=pad_ratio,
        pad_square=pad_square,
    )
    stem = Path(source_image_name).stem
    fname = f"{_safe_slug(stem)}_{instance_index}_{_safe_slug(cls_name)}.jpg"
    b64 = encode_crop_jpeg_base64(crop, quality=ingestor.jpeg_quality)
    meta = {
        "source_image": source_image_name,
        "box": [x1, y1, x2, y2],
        "seg_cls_name": seg_cls_name,
        "seg_conf": seg_conf,
        "cls_name": cls_name,
        "cls_name_zh": cls_name_to_zh(cls_name),
        "ingest_crop_mode": "bbox" if ingest_crop_from_bbox else "polygon",
        "ingest_pad_square": pad_square,
    }
    poly = row.get("polygon")
    if poly:
        meta["polygon_point_count"] = len(poly)
    return build_image_classification_task(
        cls_name,
        cls_conf,
        image_base64=b64,
        image_filename=fname,
        choice_from_name=choice_from_name or ingestor.choice_from_name,
        choice_to_name=choice_to_name or ingestor.choice_to_name,
        meta=meta,
    )


def seg_rows_to_tasks(
    ingestor: LsClassificationIngestor,
    image: np.ndarray,
    rows: Iterable[Mapping[str, Any]],
    *,
    source_image_name: str,
    skip_filtered: bool = True,
    skip_other: bool = True,
    ingest_crop_from_bbox: bool = True,
    pad_ratio: float = 0.05,
    pad_square: bool = False,
) -> list[dict[str, Any]]:
    """将一张图的多条分割实例转为 LS 图像分类任务列表。"""
    tasks: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if skip_filtered and row.get("filter"):
            continue
        cls_name = str(row.get("cls_name") or row.get("class_name") or "")
        if skip_other and is_other_cls_name(cls_name):
            continue
        poly = row.get("polygon") or []
        if len(poly) < 3:
            logger.warning("跳过实例 %s #%s: polygon 不足 3 点", source_image_name, idx)
            continue
        try:
            tasks.append(
                seg_row_to_task(
                    ingestor,
                    image,
                    row,
                    source_image_name=source_image_name,
                    instance_index=idx,
                    ingest_crop_from_bbox=ingest_crop_from_bbox,
                    pad_ratio=pad_ratio,
                    pad_square=pad_square,
                )
            )
        except Exception as exc:
            logger.warning(
                "跳过实例 %s #%s: %s", source_image_name, idx, exc, exc_info=True
            )
    return tasks


def ingest_predict_seg_results(
    ingestor: LsClassificationIngestor,
    rows: Iterable[Mapping[str, Any]],
    image: np.ndarray,
    *,
    source_image_name: str,
    skip_filtered: bool = True,
    skip_other: bool = True,
    ingest_crop_from_bbox: bool = True,
    pad_ratio: float = 0.05,
    pad_square: bool | None = None,
    ingest_match_inference_crop: bool = False,
    inference_pad_square: bool = True,
    post: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    将 ``PredictSeg.predict`` 返回列表转为任务并可选立即推送。

    :param pad_square: 显式指定上报是否 pad 正方形；为 ``None`` 时由 ``ingest_match_inference_crop`` 决定。
    :param ingest_match_inference_crop: True 时上报图与推理分类裁剪一致（可含 pad 转正方形）。
    :param inference_pad_square: 推理侧 ``cls_pad_square``，仅 ``ingest_match_inference_crop=True`` 时生效。
    :return: (tasks, ingest_responses) — 若 ``post=False``，ingest_responses 为空列表。
    """
    effective_pad = (
        bool(pad_square)
        if pad_square is not None
        else resolve_ingest_pad_square(
            ingest_match_inference_crop=ingest_match_inference_crop,
            inference_pad_square=inference_pad_square,
        )
    )
    tasks = seg_rows_to_tasks(
        ingestor,
        image,
        rows,
        source_image_name=source_image_name,
        skip_filtered=skip_filtered,
        skip_other=skip_other,
        ingest_crop_from_bbox=ingest_crop_from_bbox,
        pad_ratio=pad_ratio,
        pad_square=effective_pad,
    )
    responses: list[dict[str, Any]] = []
    if post and tasks:
        responses = ingestor.post_tasks_in_batches(tasks)
    return tasks, responses


def run_demo(
    *,
    # --- 输入与推理（PredictSeg，对齐 predict_seg_validate.py）---
    input_path: str | Path,
    seg_model_path: str | Path,
    cls_model_path: str | Path | None = None,
    cls_list: Sequence[str] | None = None,
    conf_thresh: float = 0.25,
    conf_merge: float = 0.1,
    ior_threshold: float = 0.5,
    clip_size: int = 0,
    overlap_size: int = 200,
    infer_imgsz: int = 960,
    pad_full_image_to_square: bool = True,
    detect_nms_iou: float = 0.75,
    detect_max_det: int = 1000,
    detect_nms_agnostic: bool = True,
    min_instance_size: int = 3,
    insect_alg_seg_path: str | Path = INSECT_ALG_SEG_JSON_REL,
    insect_alg_profile: str | None = "insect",
    insect_dia_filter_class_keys: set[str] | None = None,
    cls_top1_conf_threshold: float | None = 0.3,
    cls_pad_square: bool = True,
    cls_gray_binarize: bool = False,
    crop_pad_ratio: float = 0.05,
    cls_crop_from_bbox: bool = True,
    augment: bool = False,
    device: str | None = None,
    # --- LS 数据接口 ---
    ls_ingest_url: str,
    choice_from_name: str = "choice",
    choice_to_name: str = "image",
    ingest_batch_size: int = 200,
    jpeg_quality: int = 90,
    # --- 上报过滤 ---
    skip_filtered: bool = True,
    skip_other: bool = True,
    ingest_crop_from_bbox: bool = True,
    ingest_match_inference_crop: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    分割 + 分类推理后，将裁剪图以图像分类任务推送到 LS 数据接口（demo 管线）。

    :param cls_crop_from_bbox: True 时分类输入按实例外接框矩形截取；False 时按分割多边形掩码裁剪。
    :param ingest_crop_from_bbox: True 时上报图按外接框从大图矩形截取；False 时按分割掩码裁剪。
    :param ingest_match_inference_crop: True 时上报裁剪与推理一致（含 pad 转正方形）；默认 bbox 直裁无白边。
    :return: 汇总 dict：total_images, total_instances, total_tasks, total_pushed, ingest_url
    """
    ingest_pad_square = resolve_ingest_pad_square(
        ingest_match_inference_crop=ingest_match_inference_crop,
        inference_pad_square=cls_pad_square,
    )
    ingestor = LsSegClassificationIngestor(
        ingest_url=resolve_ls_ingest_url(ingest_url=ls_ingest_url),
        choice_from_name=choice_from_name,
        choice_to_name=choice_to_name,
        ingest_batch_size=max(1, int(ingest_batch_size)),
        jpeg_quality=max(50, min(100, int(jpeg_quality))),
    )

    t0 = time.perf_counter()
    predictor = PredictSeg(
        seg_model_path=str(seg_model_path),
        cls_model_path=str(cls_model_path) if cls_model_path else None,
        cls_list=list(cls_list) if cls_list else None,
        conf_thresh=conf_thresh,
        conf_merge=conf_merge,
        ior_threshold=ior_threshold,
        device=device,
        augment=augment,
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
    logger.info("模型加载 %.2fs", time.perf_counter() - t0)

    input_p, image_files = collect_images(str(input_path))
    insect_alg = load_insect_alg(insect_alg_seg_path)
    dia_range = resolve_seg_dia_range(
        insect_alg,
        insect_alg_profile=insect_alg_profile,
        cls_list=list(cls_list) if cls_list else None,
    )
    logger.info(
        "共 %s 张图片，ingest URL=%s，分类裁剪=%s，上报裁剪=%s，上报 pad_square=%s（match_inference=%s）%s",
        len(image_files),
        ingestor.ingest_url,
        "bbox" if cls_crop_from_bbox else "polygon",
        "bbox 直裁" if ingest_crop_from_bbox else "分割掩码",
        ingest_pad_square,
        ingest_match_inference_crop,
        f"，dia 过滤 [{dia_range[0]:.0f}, {dia_range[1]:.0f}]" if dia_range else "",
    )

    total_instances = 0
    total_tasks = 0
    total_pushed = 0
    total_dia_filtered = 0
    try:
        for idx, img_path in enumerate(image_files, 1):
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning("[%s/%s] 无法读取: %s", idx, len(image_files), img_path)
                continue
            rel_name = (
                str(img_path.relative_to(input_p))
                if input_p.is_dir()
                else img_path.name
            )
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
                return_all_rows=True,
            )
            for r in rows:
                x1, y1, x2, y2 = bbox_from_row(r)
                r["x1"], r["y1"], r["x2"], r["y2"] = x1, y1, x2, y2

            results_visible = list(rows)
            if dia_range is not None:
                results_visible, n_dia_drop = filter_rows_by_bbox_diag_range(
                    results_visible,
                    dia_range[0],
                    dia_range[1],
                    class_keys=insect_dia_filter_class_keys,
                )
                total_dia_filtered += n_dia_drop

            total_instances += len(results_visible)
            tasks, responses = ingest_predict_seg_results(
                ingestor,
                results_visible,
                img,
                source_image_name=rel_name,
                skip_filtered=skip_filtered,
                skip_other=skip_other,
                ingest_crop_from_bbox=ingest_crop_from_bbox,
                pad_ratio=crop_pad_ratio,
                pad_square=ingest_pad_square,
                post=not dry_run,
            )
            total_tasks += len(tasks)
            if responses:
                total_pushed += sum(int(r.get("task_count", 0) or 0) for r in responses)
            logger.info(
                "[%s/%s] %s  实例=%s  上报分类任务=%s",
                idx,
                len(image_files),
                rel_name,
                len(results_visible),
                len(tasks),
            )
    finally:
        predictor.release()

    summary = {
        "total_images": len(image_files),
        "total_instances": total_instances,
        "total_tasks": total_tasks,
        "total_pushed": total_pushed,
        "dia_filtered": total_dia_filtered,
        "cls_crop_from_bbox": cls_crop_from_bbox,
        "ingest_crop_from_bbox": ingest_crop_from_bbox,
        "ingest_match_inference_crop": ingest_match_inference_crop,
        "ingest_pad_square": ingest_pad_square,
        "dry_run": dry_run,
        "ingest_url": ingestor.ingest_url,
    }
    logger.info("完成: %s", summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # ----------------------- 模型与推理（对齐 predict_seg_validate.py）-----------------------
    # 分割 YOLO 权重路径（.pt）
    SEG_MODEL_PATH = (
        # "/Users/shunyaoyin/Documents/code/ai-company/insect/doc/测试结果/大虫框选/v2.9-0518/best-2.9.4-45.pt",
        "/Users/shunyaoyin/Documents/code/ai-company/insect/doc/测试结果/大虫框选/v2.11/best-2.11-seg.pt"
    )
    # 分类模型路径；None 则仅输出分割类别、不上报细分类
    CLS_MODEL_PATH = (
        # "/Users/shunyaoyin/Documents/code/ai-company/insect/doc/测试结果/分类测试/v2.5-seg-cls/best.pt"
        "/Users/shunyaoyin/Documents/code/ai-company/insect/doc/测试结果/分类测试/v2.3-all/best-2.3.pt"
    )
    # 仅对这些「分割输出的类名」跑分类；None/[] 表示全部实例都分类
    CLS_LIST = None

    # 分割置信度门限（与 insect_alg 叠加时取更严一侧，见 predict_seg）
    CONF_THRESH = 0.25
    # 切片合并时同类框置信度融合阈值
    CONF_MERGE = 0.1
    # 切片合并同类实例的 IoR 阈值
    IOR_THRESHOLD = 0.5
    # 大图切片边长；0 表示整图推理不切片
    CLIP_SIZE = 0
    # 相邻切片重叠像素（仅 clip_size>0 时生效）
    OVERLAP_SIZE = 200
    # 分割推理输入边长（imgsz）
    INFER_IMGSZ = 960
    # 整图推理前是否 pad 成正方形
    PAD_FULL_IMAGE_TO_SQUARE = True
    # 分割检测 NMS IoU
    DETECT_NMS_IOU = 0.75
    # 单图最大检测实例数
    DETECT_MAX_DET = 1000
    # NMS 是否类别无关
    DETECT_NMS_AGNOSTIC = True
    # 实例最小宽/高（像素），低于则丢弃
    MIN_INSTANCE_SIZE = 3

    # 算法阈值 JSON（相对 script/）；含 detect_conf / cls_conf / dia
    INSECT_ALG_SEG_PATH = INSECT_ALG_SEG_JSON_REL
    # JSON 内 profile 名，用于读取 dia 等配置
    INSECT_ALG_PROFILE = "insect"
    # dia 过滤限定类名集合；None=profile 有 dia 时过滤全部实例
    INSECT_DIA_FILTER_CLASS_KEYS = None
    # 分类 top1 全局门限；JSON 未命中 per-class 阈值时使用
    CLS_TOP1_CONF_THRESHOLD = 0.3

    # 分类前是否将裁剪图 pad 成正方形（与 train_cls 一致）
    CLS_PAD_SQUARE = True
    # 分类前是否灰度二值化
    CLS_GRAY_BINARIZE = False
    # polygon 裁剪外扩比例（cls_crop_from_bbox=False 时生效）
    CROP_PAD_RATIO = 0.05
    # 分类输入裁剪：False=多边形掩码；True=实例外接框矩形
    CLS_CROP_FROM_BBOX = True
    # 推理是否 TTA 增强
    AUGMENT = False
    # 推理设备，如 "cuda:0" / "cpu"；None 自动选择
    DEVICE = None

    # 待处理图片目录或单张图片路径
    # INPUT_PATH = "/Users/shunyaoyin/Documents/code/datasets/insect-data/测试集/北京0518"
    INPUT_PATH = "/Users/shunyaoyin/Documents/code/datasets/insect-data/训练集/bj-0509-0511"

    # ----------------------- LS 数据接口 -----------------------
    # LS api_interface 数据接入完整 URL（含 ingest token）
    LS_INGEST_URL = (
        # 北京0518 polygon
        # "http://8.137.33.38:48080/api/data-sources/ingest/ingest_1779634061843_dcf978e82520e/"
        # 北京0518 bbox
        # "http://8.137.33.38:48080/api/data-sources/ingest/ingest_1779637990782_405f10b3f24288/"
        # 生产 0509~0511
        "http://8.137.33.38:48080/api/data-sources/ingest/ingest_1779689776429_0db3d4b01a2ad8/"
    )
    # 上报裁剪图 JPEG 质量（50–100）
    JPEG_QUALITY = 95
    # LS 任务中分类选项控件 from_name
    CHOICE_FROM_NAME = "choice"
    # LS 任务中图片控件 to_name
    CHOICE_TO_NAME = "image"
    # 单次 POST 批量条数上限
    INGEST_BATCH_SIZE = 200

    # 是否跳过 predict 结果中 filter=True 的实例
    SKIP_FILTERED = True
    # 是否跳过 cls_name 为 other/其它 的实例
    SKIP_OTHER = False
    # 上报 LS 的裁剪图：True=bbox 矩形；False=分割多边形掩码（与分类裁剪独立）
    INGEST_CROP_FROM_BBOX = True
    # True：上报图与推理分类裁剪一致（含 pad 转正方形）；False：bbox 直裁无白边（默认）
    INGEST_MATCH_INFERENCE_CROP = False
    # True 只构建任务不推送（调试用）
    DRY_RUN = False

    run_demo(
        input_path=INPUT_PATH,
        seg_model_path=SEG_MODEL_PATH,
        cls_model_path=CLS_MODEL_PATH,
        cls_list=CLS_LIST,
        conf_thresh=CONF_THRESH,
        conf_merge=CONF_MERGE,
        ior_threshold=IOR_THRESHOLD,
        clip_size=CLIP_SIZE,
        overlap_size=OVERLAP_SIZE,
        infer_imgsz=INFER_IMGSZ,
        pad_full_image_to_square=PAD_FULL_IMAGE_TO_SQUARE,
        detect_nms_iou=DETECT_NMS_IOU,
        detect_max_det=DETECT_MAX_DET,
        detect_nms_agnostic=DETECT_NMS_AGNOSTIC,
        min_instance_size=MIN_INSTANCE_SIZE,
        insect_alg_seg_path=INSECT_ALG_SEG_PATH,
        insect_alg_profile=INSECT_ALG_PROFILE,
        insect_dia_filter_class_keys=INSECT_DIA_FILTER_CLASS_KEYS,
        cls_top1_conf_threshold=CLS_TOP1_CONF_THRESHOLD,
        cls_pad_square=CLS_PAD_SQUARE,
        cls_gray_binarize=CLS_GRAY_BINARIZE,
        crop_pad_ratio=CROP_PAD_RATIO,
        cls_crop_from_bbox=CLS_CROP_FROM_BBOX,
        augment=AUGMENT,
        device=DEVICE,
        ls_ingest_url=LS_INGEST_URL,
        choice_from_name=CHOICE_FROM_NAME,
        choice_to_name=CHOICE_TO_NAME,
        ingest_batch_size=INGEST_BATCH_SIZE,
        jpeg_quality=JPEG_QUALITY,
        skip_filtered=SKIP_FILTERED,
        skip_other=SKIP_OTHER,
        ingest_crop_from_bbox=INGEST_CROP_FROM_BBOX,
        ingest_match_inference_crop=INGEST_MATCH_INFERENCE_CROP,
        dry_run=DRY_RUN,
    )
