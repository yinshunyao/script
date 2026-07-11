#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Detail  : 预览 detect/seg 推理前 gray CLAHE 增强效果（与 model_channel 管线一致）。

from __future__ import annotations

import sys
from pathlib import Path

import cv2

_FILE = Path(__file__).resolve()
_INSECT_ROOT = _FILE.parents[2]
if str(_INSECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_INSECT_ROOT))

from script.predict.model_channel import save_gray_contrast_preview
from script.predict.model_seg import _pad_tile_to_clip_square
from script.predict_all import load_insect_alg_all, resolve_gray_contrast_options, resolve_root_cfg_from_alg, resolve_seg_imgsz

if __name__ == "__main__":
    # 改这里后直接运行（不走命令行参数）
    CONFIG_PATH = "config/insect_alg_all.json"
    ROOT_ID = "detect_big"
    IMAGE_PATH = "/Volumes/shunyao-h1/训练数据/测试集/北京设备全标注"
    IMAGE_NAME = ""  # 空则取目录下第一张 jpg/png
    # 预览输出目录（请指定到结果目录，勿写入 script/ 代码目录）
    OUTPUT_DIR = "/Volumes/shunyao-h1/训练数据/测试集/北京设备全标注-gray-preview"

    cfg_all = load_insect_alg_all(_INSECT_ROOT / "script" / CONFIG_PATH)
    root_cfg = resolve_root_cfg_from_alg(cfg_all, ROOT_ID)
    gray_opts = resolve_gray_contrast_options(root_cfg)
    seg_imgsz = resolve_seg_imgsz(root_cfg)
    clip = float(gray_opts["gray_clahe_clip"])
    tile = int(gray_opts["gray_clahe_tile"])

    img_path = Path(IMAGE_PATH)
    if IMAGE_NAME:
        src = img_path / IMAGE_NAME if img_path.is_dir() else Path(IMAGE_PATH)
    elif img_path.is_dir():
        cands = sorted(
            list(img_path.glob("*.jpg"))
            + list(img_path.glob("*.jpeg"))
            + list(img_path.glob("*.png"))
        )
        if not cands:
            raise SystemExit(f"目录无图片: {img_path}")
        src = cands[0]
    else:
        src = img_path

    bgr = cv2.imread(str(src))
    if bgr is None:
        raise SystemExit(f"无法读取: {src}")

    if bool(root_cfg.get("to_square", True)):
        h, w = bgr.shape[:2]
        if w != h:
            side = max(w, h)
            bgr, _, _, _, _ = _pad_tile_to_clip_square(bgr, w, h, side)

    out_dir = Path(OUTPUT_DIR)
    out_path = save_gray_contrast_preview(
        out_dir,
        bgr,
        stem=src.stem,
        target_imgsz=seg_imgsz,
        clahe_clip=clip,
        clahe_tile=tile,
    )
    print(str(out_path))
    print(f"配置: clip={clip} tile={tile} seg_imgsz={seg_imgsz}")
