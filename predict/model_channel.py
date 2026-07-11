#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Detail  : YOLO 推理输入通道自适应工具。
#           以加载到的模型自身期望输入通道数为准，在送入 model.predict 前把图像
#           调整到对应通道：ch=1 → 单通道灰度 (H,W,1)；ch=3 → 三通道 BGR。
#           背景：Ultralytics 的 numpy 通路不会按模型通道数自动转灰度，
#           对 ch=1 单通道模型必须由调用方送入单通道数组，否则通道不匹配。

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

_MPS_CH1_FIX_INSTALLED = False


def detect_model_input_channels(yolo_model, default: int = 3) -> int:
    """
    从已加载的 ultralytics ``YOLO`` 读取期望输入通道数（1 或 3）。

    优先 ``model.model.yaml["channels"]``（训练 data.yaml 的 ``channels`` 会落入模型 yaml），
    兼容 ``yaml["ch"]`` 与 ``getattr(model, "ch", ...)``；任何异常都安全回退 ``default``。
    """
    try:
        inner = getattr(yolo_model, "model", None)
        yaml = getattr(inner, "yaml", None)
        if isinstance(yaml, dict):
            ch = yaml.get("channels", yaml.get("ch"))
            if ch:
                ch_int = int(ch)
                if ch_int > 0:
                    return ch_int
        for holder in (inner, yolo_model):
            ch = getattr(holder, "ch", None)
            if ch:
                ch_int = int(ch)
                if ch_int > 0:
                    return ch_int
    except Exception:  # noqa: BLE001 — 识别失败一律回退，绝不中断推理
        logging.debug("检测模型输入通道数失败，回退 %d", default, exc_info=True)
    return int(default)


def _is_mps_device(device) -> bool:
    if isinstance(device, str):
        return device.strip().lower().startswith("mps")
    try:
        import torch

        if isinstance(device, torch.device):
            return device.type == "mps"
    except Exception:  # noqa: BLE001
        pass
    return False


def ensure_ultralytics_mps_single_channel_fix() -> None:
    """
    修补 Ultralytics ``BasePredictor.preprocess``：单通道 BHWC→BCHW 后
    ``np.ascontiguousarray`` 因 size-1 通道维不拷贝，MPS 深层 ``.view()`` 报错。
    对 ``C==1`` 在 transpose 后强制 ``.copy()`` 再转 tensor。
    """
    global _MPS_CH1_FIX_INSTALLED
    if _MPS_CH1_FIX_INSTALLED:
        return
    try:
        import torch

        if not torch.backends.mps.is_available():
            _MPS_CH1_FIX_INSTALLED = True
            return
        from ultralytics.engine.predictor import BasePredictor

        if getattr(BasePredictor, "_insect_mps_ch1_fix", False):
            _MPS_CH1_FIX_INSTALLED = True
            return

        _orig_preprocess = BasePredictor.preprocess

        def _preprocess_mps_ch1_safe(self, im):
            not_tensor = not isinstance(im, torch.Tensor)
            if not_tensor:
                im_arr = np.stack(self.pre_transform(im))
                if im_arr.shape[-1] == 3:
                    im_arr = im_arr[..., ::-1]
                im_arr = im_arr.transpose((0, 3, 1, 2))
                if im_arr.shape[1] == 1:
                    im_arr = np.ascontiguousarray(im_arr.copy())
                else:
                    im_arr = np.ascontiguousarray(im_arr)
                im = torch.from_numpy(im_arr)
                im = im.to(self.device)
                im = im.half() if self.model.fp16 else im.float()
                im /= 255
                return im
            return _orig_preprocess(self, im)

        BasePredictor.preprocess = _preprocess_mps_ch1_safe
        BasePredictor._insect_mps_ch1_fix = True
        _MPS_CH1_FIX_INSTALLED = True
        logging.info(
            "已安装 Ultralytics MPS 单通道(ch=1) preprocess stride 修复"
        )
    except Exception:  # noqa: BLE001
        logging.warning(
            "安装 Ultralytics MPS 单通道 preprocess 修复失败，"
            "ch=1 模型在 MPS 上可能仍报 stride 错误",
            exc_info=True,
        )


def mps_safe_device(device, model_ch: int):
    """
    单通道(ch=1)模型在 Apple MPS 上推理前安装 preprocess stride 修复，设备保持 ``mps``。

    历史：torch 2.7.x + ultralytics 单通道 numpy 预处理会在 MPS 上触发
    ``view size is not compatible ... stride``；先前版本回退 ``cpu``，现改为补丁修复。
    """
    if int(model_ch) == 1 and _is_mps_device(device):
        ensure_ultralytics_mps_single_channel_fix()
    return device


def _bgr_or_gray_to_gray(image: np.ndarray) -> np.ndarray | None:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0]
    return None


def gray_apply_clahe(
    gray: np.ndarray,
    *,
    clahe_clip: float = 2.0,
    clahe_tile: int = 8,
) -> np.ndarray:
    tile = max(1, int(clahe_tile))
    clahe = cv2.createCLAHE(
        clipLimit=float(clahe_clip),
        tileGridSize=(tile, tile),
    )
    return clahe.apply(gray)


def bgr_apply_gray_clahe(
    image: np.ndarray | None,
    *,
    clahe_clip: float = 2.0,
    clahe_tile: int = 8,
) -> np.ndarray | None:
    """
    对 BGR/灰度图做 CLAHE 对比度增强，返回三通道 BGR（R=G=B）。

    仅增强亮度对比，不做 Otsu 二值化，适合检测/分割输入（保留梯度信息）。
    """
    if image is None or getattr(image, "size", 0) == 0:
        return image
    gray = _bgr_or_gray_to_gray(image)
    if gray is None:
        return image
    enhanced = gray_apply_clahe(gray, clahe_clip=clahe_clip, clahe_tile=clahe_tile)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def _resize_for_yolo_imgsz(image: np.ndarray, target_imgsz: int) -> np.ndarray:
    side = int(target_imgsz)
    if side <= 0:
        return image
    h, w = image.shape[:2]
    if h == side and w == side:
        return image
    return cv2.resize(image, (side, side), interpolation=cv2.INTER_LINEAR)


def yolo_input_coord_scale(
    source_shape: tuple[int, ...],
    preprocessed: np.ndarray | None,
) -> tuple[float, float]:
    """
    YOLO 在 ``preprocessed`` 尺寸下输出框坐标，映射回 ``source_shape`` 像素的 (sx, sy)。

    ``gray_contrast_enhance`` 预 resize 到 ``imgsz`` 后须用本函数把检测框还原到送入预处理前的图幅。
    """
    if preprocessed is None or getattr(preprocessed, "size", 0) == 0:
        return 1.0, 1.0
    sh, sw = int(source_shape[0]), int(source_shape[1])
    ph, pw = int(preprocessed.shape[0]), int(preprocessed.shape[1])
    if sh <= 0 or sw <= 0 or ph <= 0 or pw <= 0 or (sh, sw) == (ph, pw):
        return 1.0, 1.0
    return sw / pw, sh / ph


def scale_xyxy(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    sx: float,
    sy: float,
) -> tuple[int, int, int, int]:
    return (
        int(round(x1 * sx)),
        int(round(y1 * sy)),
        int(round(x2 * sx)),
        int(round(y2 * sy)),
    )


def scale_polygon_points(
    polygon: list[list[float | int]],
    sx: float,
    sy: float,
) -> list[list[int]]:
    if not polygon:
        return []
    return [[int(round(float(x) * sx)), int(round(float(y) * sy))] for x, y in polygon]


def compute_gray_contrast_enhanced(
    image: np.ndarray,
    *,
    target_imgsz: int,
    clahe_clip: float = 2.0,
    clahe_tile: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    与 ``preprocess_yolo_input(gray_contrast_enhance=True)`` 一致的中间结果。

    Returns:
        (resize 后灰度, CLAHE 后灰度)，均为 ``(H,W)`` uint8。
    """
    side = int(target_imgsz)
    work = _resize_for_yolo_imgsz(image, side) if side > 0 else image
    gray = _bgr_or_gray_to_gray(work)
    if gray is None:
        raise ValueError("无法从输入解析灰度图")
    enhanced = gray_apply_clahe(gray, clahe_clip=clahe_clip, clahe_tile=clahe_tile)
    return gray, enhanced


def imwrite_bgr_or_gray(path: str | Path, image: np.ndarray) -> bool:
    """写入 BGR 或单通道灰度；含中文路径时用 imencode 兜底。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if image.ndim == 2:
        out = image
    elif image.ndim == 3 and image.shape[2] == 1:
        out = image[:, :, 0]
    else:
        out = image
    path_str = str(p)
    if cv2.imwrite(path_str, out):
        return True
    ok, buf = cv2.imencode(p.suffix or ".png", out)
    if not ok:
        return False
    p.write_bytes(buf.tobytes())
    return True


def save_gray_contrast_preview(
    save_dir: str | Path,
    image_bgr: np.ndarray,
    *,
    stem: str,
    target_imgsz: int,
    clahe_clip: float,
    clahe_tile: int,
) -> Path:
    """
    保存 CLAHE 后灰度预览（与送入 YOLO 的增强结果一致）。

    文件名：``{stem}_gray_clahe_c{clip}_t{tile}.png``。
    """
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _gray, enhanced = compute_gray_contrast_enhanced(
        image_bgr,
        target_imgsz=target_imgsz,
        clahe_clip=clahe_clip,
        clahe_tile=clahe_tile,
    )
    tag_clip = str(clahe_clip).replace(".", "p")
    out_path = out_dir / f"{stem}_gray_clahe_c{tag_clip}_t{int(clahe_tile)}.png"
    imwrite_bgr_or_gray(out_path, enhanced)
    logging.info(
        "已保存灰度 CLAHE 预览: %s (clip=%s tile=%s imgsz=%s)",
        out_path,
        clahe_clip,
        clahe_tile,
        target_imgsz,
    )
    return out_path


def preprocess_yolo_input(
    image: np.ndarray | None,
    model_ch: int,
    *,
    gray_contrast_enhance: bool = False,
    clahe_clip: float = 2.0,
    clahe_tile: int = 8,
    target_imgsz: int = 0,
    debug_save_dir: str | None = None,
    debug_save_stem: str | None = None,
) -> np.ndarray | None:
    """
    检测/分割送入 YOLO 前的统一预处理。

    ``gray_contrast_enhance=True`` 时顺序固定为：
    **resize 到 ``target_imgsz``（与 YOLO ``imgsz`` 一致）→ 灰度 → CLAHE → 通道适配**。
    关闭增强时仍为原图通道适配，由 YOLO 内部做 ``imgsz`` 缩放。
    """
    if image is None or getattr(image, "size", 0) == 0:
        return image

    if gray_contrast_enhance:
        side = int(target_imgsz)
        if side <= 0:
            logging.warning(
                "gray_contrast_enhance 已开启但 target_imgsz=%s，跳过预 resize，"
                "仍按原图尺寸做灰度 CLAHE",
                target_imgsz,
            )
        else:
            image = _resize_for_yolo_imgsz(image, side)
        gray = _bgr_or_gray_to_gray(image)
        if gray is None:
            return adapt_image_to_model_channels(image, model_ch)
        enhanced = gray_apply_clahe(gray, clahe_clip=clahe_clip, clahe_tile=clahe_tile)
        if debug_save_dir and debug_save_stem:
            try:
                save_gray_contrast_preview(
                    debug_save_dir,
                    image,
                    stem=debug_save_stem,
                    target_imgsz=side if side > 0 else int(gray.shape[0]),
                    clahe_clip=clahe_clip,
                    clahe_tile=clahe_tile,
                )
            except Exception:
                logging.warning("保存 gray CLAHE 预览失败", exc_info=True)
        if int(model_ch) == 1:
            return enhanced[..., None]
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    return adapt_image_to_model_channels(image, model_ch)


def adapt_image_to_model_channels(image: np.ndarray | None, model_ch: int) -> np.ndarray | None:
    """
    送入 YOLO 前把图像调整到模型期望通道数。

    - ``model_ch == 1``：``(H,W,3)`` BGR 或 ``(H,W)`` 灰度 → ``(H,W,1)`` 单通道
      （``cv2.COLOR_BGR2GRAY``，等价 ITU-R 601-2，与训练侧灰度一致）；``(H,W,1)`` 幂等。
    - ``model_ch == 3``：``(H,W)`` / ``(H,W,1)`` 单通道 → ``(H,W,3)`` BGR；其余原样。
    - 空图 / ``None``：原样返回。
    """
    if image is None or getattr(image, "size", 0) == 0:
        return image

    if int(model_ch) == 1:
        if image.ndim == 2:
            return image[..., None]
        if image.ndim == 3:
            if image.shape[2] == 1:
                return image
            if image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                return gray[..., None]
        return image

    # model_ch == 3（及其它非 1 取值的兜底）：保证三通道
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 1:
        return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    return image
