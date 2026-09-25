#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将昆虫检测+分类结果以「图像分类」任务推送到 LS 标注系统数据接口（api_interface）。

格式对齐 ``ls/label-studio/label_studio/data_sources/api_interface.py`` 中
``image_classification`` + ``native_json`` 预标注范例。

典型用法::

    from script.predict_size import PredictSize
    from script.ls_classification_ingest import LsClassificationIngestor, ingest_predict_size_results

    predictor = PredictSize(...)
    ingestor = LsClassificationIngestor(
        ingest_url="http://127.0.0.1:8080/api/data-sources/ingest/{ingest_token}/",
    )
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        rows = predictor.predict(img, ...)
        ingest_predict_size_results(ingestor, rows, img, source_image_name=img_path.name)
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence
from urllib import error, request

import cv2
import numpy as np

_FILE = Path(__file__).resolve()
_ROOT = _FILE.parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from script.config.insect_info import json_record
from script.predict.model_cls import ModelCls

logger = logging.getLogger(__name__)

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

# 历史/拼写别名 → insect_info.json 拼音 key
_CLS_PINYIN_ALIASES: dict[str, str] = {
    "baifeifeishi": "baibeifeishi",
}

# insect_info 未收录时的常用兜底（稻飞虱细分类等）
_CLS_ZH_FALLBACK: dict[str, str] = {
    "huifeishi": "灰飞虱",
    "other": "其他",
    "unknown": "未知",
}
_OTHER_CANONICAL = "other"
_OTHER_ZH_PREFIX = "其他"


def is_other_cls_name(cls_name: str) -> bool:
    """
    判定是否应视为「其他」类：拼音键为 ``other`` 或以 ``other`` 开头；中文以「其他」开头。
    """
    raw = str(cls_name or "").strip()
    if not raw:
        return False
    if _CJK_RE.search(raw):
        return raw.startswith(_OTHER_ZH_PREFIX)
    key = _CLS_PINYIN_ALIASES.get(raw.lower(), raw.lower())
    return key == _OTHER_CANONICAL or key.startswith("other")


def normalize_cls_name_for_ingest(cls_name: str) -> str:
    """上报 LS 前将 other 系别名统一为 canonical 拼音键 ``other``（展示为「其他」）。"""
    raw = str(cls_name or "").strip()
    if not raw:
        return "unknown"
    if is_other_cls_name(raw):
        return _OTHER_CANONICAL
    return raw


def cls_name_to_zh(cls_name: str) -> str:
    """
    将模型输出的分类键（拼音/英文）映射为中文展示名，用于 LS ``choices`` 预标注。

    优先查 ``insect_info.json``；已是中文则原样返回；无映射时退回原字符串。
    """
    raw = str(cls_name or "").strip()
    if not raw:
        return _CLS_ZH_FALLBACK.get("unknown", "未知")
    if is_other_cls_name(raw):
        return _CLS_ZH_FALLBACK[_OTHER_CANONICAL]
    if _CJK_RE.search(raw):
        return raw
    key = raw.lower()
    pinyin_key = _CLS_PINYIN_ALIASES.get(key, key)
    rec = json_record(pinyin_key)
    if rec is not None and str(rec.name_zh or "").strip():
        return str(rec.name_zh).strip()
    fb = _CLS_ZH_FALLBACK.get(key)
    if fb:
        return fb
    return raw

_PIC_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_INGEST_PATH_MARKER = "/api/data-sources/ingest/"


def _safe_slug(text: str, max_len: int = 48) -> str:
    s = re.sub(r"[^\w.\-]+", "_", str(text or "").strip(), flags=re.UNICODE)
    s = s.strip("._") or "item"
    return s[:max_len]


def normalize_ingest_url(url: str) -> str:
    """规范化数据接口完整 URL（须含 ``/api/data-sources/ingest/{token}/``）。"""
    u = str(url or "").strip()
    if not u:
        raise ValueError("ingest URL 为空")
    if _INGEST_PATH_MARKER not in u:
        raise ValueError(
            f"ingest URL 须包含 {_INGEST_PATH_MARKER} 与 ingest_token，"
            "示例: http://host:8080/api/data-sources/ingest/<token>/"
        )
    if not u.endswith("/"):
        u += "/"
    return u


def resolve_ingest_pad_square(
    *,
    ingest_match_inference_crop: bool,
    inference_pad_square: bool,
) -> bool:
    """
    上报 LS 裁剪图是否 pad 成正方形。

    默认（``ingest_match_inference_crop=False``）：bbox 直裁，不补白边。
    开关打开：与推理/分类前处理一致（``inference_pad_square``，通常为 ``cls_pad_square``）。
    """
    if ingest_match_inference_crop:
        return bool(inference_pad_square)
    return False


def crop_box(
    image: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    *,
    pad_square: bool = False,
) -> np.ndarray:
    """从 BGR 整图裁剪检测框区域；可选白边补成正方形（与分类训练一致）。"""
    if image is None or image.size == 0:
        raise ValueError("image 为空")
    h_img, w_img = image.shape[:2]
    xi1 = max(0, int(x1))
    yi1 = max(0, int(y1))
    xi2 = min(w_img, int(x2))
    yi2 = min(h_img, int(y2))
    if xi2 <= xi1 or yi2 <= yi1:
        raise ValueError(f"无效框: ({x1},{y1},{x2},{y2})")
    crop = image[yi1:yi2, xi1:xi2].copy()
    if pad_square:
        crop = ModelCls.pad_bgr_to_square(crop)
    return crop


def encode_crop_jpeg_base64(crop_bgr: np.ndarray, *, quality: int = 90) -> str:
    """将 BGR 裁剪图编码为 JPEG Base64（无 data URL 前缀）。"""
    ok, buf = cv2.imencode(".jpg", crop_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise ValueError("JPEG 编码失败")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def build_image_classification_task(
    cls_name: str,
    cls_conf: float,
    *,
    image_base64: str,
    image_filename: Optional[str] = None,
    choice_from_name: str = "choice",
    choice_to_name: str = "image",
    meta: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """
    组装单条图像分类任务（含 choices 预标注）。

    与 ``api_interface._native_result_item('image_classification')`` 一致；
    裁剪图以 ``data.image_base64`` 随 ingest POST，由服务端落盘。
    """
    raw_label = normalize_cls_name_for_ingest(str(cls_name or "").strip() or "unknown")
    label = cls_name_to_zh(raw_label)
    score = float(cls_conf or 0.0)
    data: dict[str, Any] = {"image_base64": str(image_base64)}
    if image_filename:
        data["image_filename"] = str(image_filename)
    task: dict[str, Any] = {
        "data": data,
        "predictions": [
            {
                "result": [
                    {
                        "from_name": str(choice_from_name),
                        "to_name": str(choice_to_name),
                        "type": "choices",
                        "value": {"choices": [label]},
                    }
                ],
                "score": score,
            }
        ],
    }
    if meta:
        task["meta"] = dict(meta)
    return task


@dataclass
class LsClassificationIngestor:
    """向 LS api_interface 推送图像分类任务（裁剪图经 HTTP JSON 内嵌 Base64）。"""

    ingest_url: str
    jpeg_quality: int = 90
    choice_from_name: str = "choice"
    choice_to_name: str = "image"
    ingest_batch_size: int = 200  # 单次 POST 最大 task 数

    def __post_init__(self) -> None:
        self.ingest_url = normalize_ingest_url(self.ingest_url)

    def post_tasks(self, tasks: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """POST ``{"tasks": [...]}`` 到数据接口。"""
        if not tasks:
            return {"success": True, "task_count": 0, "skipped": True}
        payload = {"tasks": list(tasks)}
        return self._post_json(payload)

    def post_tasks_in_batches(self, tasks: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        """分批推送，返回每批 ingest 响应列表。"""
        batch_size = max(1, int(self.ingest_batch_size))
        results: list[dict[str, Any]] = []
        buf: list[dict[str, Any]] = []
        for task in tasks:
            buf.append(task)
            if len(buf) >= batch_size:
                results.append(self.post_tasks(buf))
                buf = []
        if buf:
            results.append(self.post_tasks(buf))
        return results

    def _post_json(self, body: Mapping[str, Any]) -> dict[str, Any]:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json; charset=utf-8"}
        req = request.Request(self.ingest_url, data=data, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=120) as resp:
                raw = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LS ingest HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"LS ingest 请求失败: {exc}") from exc
        try:
            parsed = json.loads(raw) if raw else {}
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"LS ingest 响应非 JSON: {raw[:500]}") from exc
        if not parsed.get("success", True):
            raise RuntimeError(f"LS ingest 失败: {parsed}")
        return parsed


def detection_row_to_task(
    ingestor: LsClassificationIngestor,
    image: np.ndarray,
    det: Mapping[str, Any],
    *,
    source_image_name: str,
    box_index: int,
    pad_square: bool = False,
    choice_from_name: Optional[str] = None,
    choice_to_name: Optional[str] = None,
) -> dict[str, Any]:
    """单条 ``predict_size`` 检测结果 → 裁剪、Base64 编码并组装 LS task。"""
    x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
    cls_name_raw = str(det.get("cls_name") or det.get("class_name") or "unknown")
    cls_name = normalize_cls_name_for_ingest(cls_name_raw)
    cls_conf = float(det.get("cls_conf", det.get("conf", 0.0)) or 0.0)
    crop = crop_box(image, x1, y1, x2, y2, pad_square=pad_square)
    stem = Path(source_image_name).stem
    fname = f"{_safe_slug(stem)}_{box_index}_{_safe_slug(cls_name)}.jpg"
    b64 = encode_crop_jpeg_base64(crop, quality=ingestor.jpeg_quality)
    meta = {
        "source_image": source_image_name,
        "box": [x1, y1, x2, y2],
        "det_conf": float(det.get("conf", 0.0) or 0.0),
        "cls_name": cls_name,
        "cls_name_zh": cls_name_to_zh(cls_name),
        "ingest_pad_square": pad_square,
    }
    return build_image_classification_task(
        cls_name,
        cls_conf,
        image_base64=b64,
        image_filename=fname,
        choice_from_name=choice_from_name or ingestor.choice_from_name,
        choice_to_name=choice_to_name or ingestor.choice_to_name,
        meta=meta,
    )


def detections_to_tasks(
    ingestor: LsClassificationIngestor,
    image: np.ndarray,
    detections: Iterable[Mapping[str, Any]],
    *,
    source_image_name: str,
    skip_filtered: bool = True,
    skip_other: bool = True,
    pad_square: bool = False,
) -> list[dict[str, Any]]:
    """将一张图的多条检测框转为 LS 图像分类任务列表。"""
    tasks: list[dict[str, Any]] = []
    for idx, det in enumerate(detections):
        if skip_filtered and det.get("filter"):
            continue
        cls_name = str(det.get("cls_name") or det.get("class_name") or "")
        if skip_other and is_other_cls_name(cls_name):
            continue
        try:
            tasks.append(
                detection_row_to_task(
                    ingestor,
                    image,
                    det,
                    source_image_name=source_image_name,
                    box_index=idx,
                    pad_square=pad_square,
                )
            )
        except Exception as exc:
            logger.warning(
                "跳过框 %s #%s: %s", source_image_name, idx, exc, exc_info=True
            )
    return tasks


def ingest_predict_size_results(
    ingestor: LsClassificationIngestor,
    detections: Iterable[Mapping[str, Any]],
    image: np.ndarray,
    *,
    source_image_name: str,
    skip_filtered: bool = True,
    skip_other: bool = True,
    pad_square: bool | None = None,
    ingest_match_inference_crop: bool = False,
    inference_pad_square: bool = True,
    post: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    将 ``PredictSize.predict`` 返回列表转为任务并可选立即推送。

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
    tasks = detections_to_tasks(
        ingestor,
        image,
        detections,
        source_image_name=source_image_name,
        skip_filtered=skip_filtered,
        skip_other=skip_other,
        pad_square=effective_pad,
    )
    responses: list[dict[str, Any]] = []
    if post and tasks:
        responses = ingestor.post_tasks_in_batches(tasks)
    return tasks, responses


def resolve_ls_ingest_url(*, ingest_url: Optional[str] = None) -> str:
    """从参数或环境变量 ``LS_INGEST_URL`` 读取完整 ingest 地址。"""
    raw = (ingest_url or os.environ.get("LS_INGEST_URL") or "").strip()
    if not raw:
        raise ValueError("缺少 LS_INGEST_URL（完整 ingest 地址，含 ingest_token）")
    return normalize_ingest_url(raw)


def build_ingestor_from_env(**kwargs: Any) -> LsClassificationIngestor:
    return LsClassificationIngestor(
        ingest_url=resolve_ls_ingest_url(ingest_url=kwargs.get("ingest_url")),
        jpeg_quality=max(50, min(100, int(kwargs.get("jpeg_quality") or 90))),
        choice_from_name=str(
            kwargs.get("choice_from_name") or os.environ.get("LS_CHOICE_FROM_NAME") or "choice"
        ),
        choice_to_name=str(
            kwargs.get("choice_to_name") or os.environ.get("LS_CHOICE_TO_NAME") or "image"
        ),
        ingest_batch_size=max(1, int(kwargs.get("ingest_batch_size") or 200)),
    )


def iter_image_files(input_path: str | Path) -> list[Path]:
    p = Path(input_path)
    if p.is_file():
        return [p] if p.suffix.lower() in _PIC_EXT else []
    if p.is_dir():
        return sorted(
            f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in _PIC_EXT
        )
    return []


def run_demo(
    *,
    # --- 输入与推理（PredictSize）---
    input_path: str | Path,
    cls_list: Sequence[str],
    detect_model_path: str | Path,
    cls_model_path: str | Path | None,
    size_config_path: str | Path | None = None,
    clip_size: int = 640,
    overlap_size: int = 120,
    conf_thresh: float = 0.6,
    edge_reject_distance: int = 5,
    edge_reject_conf_threshold: float = 0.8,
    edge_reject_cls_conf_threshold: float = 0.8,
    inner_boxes_fp_threshold: int = 8,
    cls_pad_square: bool = True,
    diag_filter_range: tuple[float, float] | list[float] | None = (50, 260),
    augment: bool = True,
    half: bool = True,
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
    ingest_match_inference_crop: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    检测 + 分类推理后，将裁剪图以图像分类任务推送到 LS 数据接口（demo 管线）。

    :param ingest_match_inference_crop: True 时上报裁剪与推理一致（含 pad 转正方形）；默认 bbox 直裁无白边。
    :return: 汇总 dict：total_images, total_detections, total_tasks, total_pushed, ingest_url
    """
    ingest_pad_square = resolve_ingest_pad_square(
        ingest_match_inference_crop=ingest_match_inference_crop,
        inference_pad_square=cls_pad_square,
    )
    from script.predict_size import PredictSize

    ingestor = LsClassificationIngestor(
        ingest_url=resolve_ls_ingest_url(ingest_url=ls_ingest_url),
        choice_from_name=choice_from_name,
        choice_to_name=choice_to_name,
        ingest_batch_size=max(1, int(ingest_batch_size)),
        jpeg_quality=max(50, min(100, int(jpeg_quality))),
    )

    t0 = time.perf_counter()
    predictor = PredictSize(
        detect_model_path=detect_model_path,
        size_config_path=size_config_path,
        cls_list=list(cls_list),
        cls_model_path=cls_model_path,
        conf_thresh=conf_thresh,
        device=device,
        augment=augment,
        half=half,
        diag_filter_range=diag_filter_range,
    )
    logger.info("模型加载 %.2fs", time.perf_counter() - t0)

    input_p = Path(input_path)
    image_files = iter_image_files(input_p)
    logger.info(
        "共 %s 张图片，ingest URL=%s，上报 pad_square=%s（match_inference=%s）",
        len(image_files),
        ingestor.ingest_url,
        ingest_pad_square,
        ingest_match_inference_crop,
    )

    total_detections = 0
    total_tasks = 0
    total_pushed = 0
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
            results = predictor.predict(
                img,
                clip_size=clip_size,
                overlap_size=overlap_size,
                edge_reject_distance=edge_reject_distance,
                edge_reject_conf_threshold=edge_reject_conf_threshold,
                edge_reject_cls_conf_threshold=edge_reject_cls_conf_threshold,
                cls_pad_square=cls_pad_square,
                inner_boxes_fp_threshold=inner_boxes_fp_threshold,
            )
            total_detections += len(results)
            tasks, responses = ingest_predict_size_results(
                ingestor,
                results,
                img,
                source_image_name=rel_name,
                skip_filtered=skip_filtered,
                skip_other=skip_other,
                pad_square=ingest_pad_square,
                post=not dry_run,
            )
            total_tasks += len(tasks)
            if responses:
                total_pushed += sum(int(r.get("task_count", 0) or 0) for r in responses)
            logger.info(
                "[%s/%s] %s  检测=%s  上报分类任务=%s",
                idx,
                len(image_files),
                rel_name,
                len(results),
                len(tasks),
            )
    finally:
        predictor.release()

    summary = {
        "total_images": len(image_files),
        "total_detections": total_detections,
        "total_tasks": total_tasks,
        "total_pushed": total_pushed,
        "ingest_match_inference_crop": ingest_match_inference_crop,
        "ingest_pad_square": ingest_pad_square,
        "dry_run": dry_run,
        "ingest_url": ingestor.ingest_url,
    }
    logger.info("完成: %s", summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # ---------- 推理模型 ----------
    CLS_LIST = ["daofeishi"]
    DETECT_MODEL_PATH = Path(
        "/Users/shunyaoyin/Documents/code/models/daofeishi-detect-0415.pt"
    )
    CLS_MODEL_PATH = Path("/Users/shunyaoyin/Documents/code/models/daofeishi-cls.pt")
    SIZE_CONFIG_PATH = None
    INPUT_PATH = (
        "/Users/shunyaoyin/Documents/code/datasets/insect-data/测试集/daofeishi-0522-miss"
    )

    # 稻飞虱参数
    CLIP_SIZE = 640
    OVERLAP_SIZE = 120
    CONF_THRESH = 0.6
    EDGE_REJECT_DISTANCE = 5
    EDGE_REJECT_CONF_THRESHOLD = 0.8
    EDGE_REJECT_CLS_CONF_THRESHOLD = 0.8
    INNER_BOXES_FP_THRESHOLD = 8
    CLS_PAD_SQUARE = True
    DIAG_FILTER_RANGE = (50, 260)
    AUGMENT = False
    HALF = True
    DEVICE = None

    # ---------- LS 数据接口（未填写时可读环境变量 LS_INGEST_URL）----------
    LS_INGEST_URL = (
        # "http://127.0.0.1:8080/api/data-sources/ingest/007e26044b7f44fd940248d169f4b262/"
        "http://8.137.33.38/api/data-sources/ingest/ingest_1779689776429_0db3d4b01a2ad8/"
    )
    JPEG_QUALITY = 95
    CHOICE_FROM_NAME = "choice"
    CHOICE_TO_NAME = "image"
    INGEST_BATCH_SIZE = 200

    SKIP_FILTERED = True
    SKIP_OTHER = False
    # True：上报图与推理分类裁剪一致（含 pad 转正方形）；False：bbox 直裁无白边（默认）
    INGEST_MATCH_INFERENCE_CROP = False
    DRY_RUN = False

    run_demo(
        input_path=INPUT_PATH,
        cls_list=CLS_LIST,
        detect_model_path=DETECT_MODEL_PATH,
        cls_model_path=CLS_MODEL_PATH,
        size_config_path=SIZE_CONFIG_PATH,
        clip_size=CLIP_SIZE,
        overlap_size=OVERLAP_SIZE,
        conf_thresh=CONF_THRESH,
        edge_reject_distance=EDGE_REJECT_DISTANCE,
        edge_reject_conf_threshold=EDGE_REJECT_CONF_THRESHOLD,
        edge_reject_cls_conf_threshold=EDGE_REJECT_CLS_CONF_THRESHOLD,
        inner_boxes_fp_threshold=INNER_BOXES_FP_THRESHOLD,
        cls_pad_square=CLS_PAD_SQUARE,
        diag_filter_range=DIAG_FILTER_RANGE,
        augment=AUGMENT,
        half=HALF,
        device=DEVICE,
        ls_ingest_url=LS_INGEST_URL,
        choice_from_name=CHOICE_FROM_NAME,
        choice_to_name=CHOICE_TO_NAME,
        ingest_batch_size=INGEST_BATCH_SIZE,
        jpeg_quality=JPEG_QUALITY,
        skip_filtered=SKIP_FILTERED,
        skip_other=SKIP_OTHER,
        ingest_match_inference_crop=INGEST_MATCH_INFERENCE_CROP,
        dry_run=DRY_RUN,
    )
