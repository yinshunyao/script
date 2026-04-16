#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Detail  : 在检测图上绘制与推理一致的分片网格（不加载模型、不做推理）

import logging
import os
import sys
from pathlib import Path

import cv2

_FILE = Path(__file__).resolve()
_ROOT = _FILE.parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from script.predict.model_detect import get_clip


def uses_slice_inference_path(w, h, clip_size, overlap_size):
    """
    与 script/predict/model_detect.py 中 ModelDetector.predict 开头分支一致。
    返回 True 表示与推理相同地遍历 get_clip；False 表示整图单块。
    """
    if not clip_size or not overlap_size and (
        clip_size >= w and clip_size >= h or clip_size <= overlap_size <= 1
    ):
        return False
    return True


def iter_clip_rects(w, h, clip_size, overlap_size):
    """
    与推理切片循环一致的全图坐标矩形序列 (x1,y1,x2,y2)。
    整图分支时为单块 (0,0,w,h)。
    """
    if not uses_slice_inference_path(w, h, clip_size, overlap_size):
        yield (0, 0, w, h)
        return
    for clip_x1, clip_y1, clip_x2, clip_y2 in get_clip(w, h, clip_size, overlap_size):
        yield (clip_x1, clip_y1, clip_x2, clip_y2)


def slice_step(clip_size, overlap_size):
    """与 script/predict/model_detect.get_clip 中 step 一致。"""
    step = clip_size - overlap_size
    if step <= 0:
        step = 1
    return step


def _draw_rect_inset(
    img,
    x1,
    y1,
    x2,
    y2,
    color,
    thickness,
    inset,
):
    """
    仅用于可视化：相对分片矩形四边向内缩 inset 像素再画框，减轻重叠区域线叠在一起难辨色。
    若缩进后宽或高无效则跳过绘制。
    """
    nx1 = x1 + inset
    ny1 = y1 + inset
    nx2 = x2 - inset
    ny2 = y2 - inset
    if nx2 <= nx1 or ny2 <= ny1:
        return
    cv2.rectangle(img, (nx1, ny1), (nx2, ny2), color, thickness)


def draw_clip_grid(
    image,
    clip_size,
    overlap_size,
    palette=((0, 255, 255), (255, 0, 255)),
    thickness=2,
    draw_index=False,
    index_color=(255, 255, 255),
    draw_inset_px=2,
):
    """
    在 BGR 图像上绘制推理分片矩形框。

    每个分片整框使用一种颜色；切片网格上按棋盘格用 ``palette[0]`` / ``palette[1]`` 交替，
    使**四邻接**相邻分片颜色不同。仅 1 色时全部分片同色；多于 2 色时只取前 2 色参与棋盘。

    :param palette: BGR 元组序列，至少 2 个时取前两种交替；仅 1 个时所有分片同色。
    :param index_color: 序号文字颜色（BGR）
    :param draw_index: True 时在每片左上角标注序号（与 get_clip 遍历顺序一致）
    :param draw_inset_px: 画框时相对分片矩形向内缩进的像素（仅影响描边，不影响裁剪与棋盘格下标）；缩进后无效则跳过该框。
    :return: 绘制后的图像副本
    """
    img_draw = image.copy()
    h, w = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    colors = list(palette) if palette else [(0, 255, 0)]
    slice_mode = uses_slice_inference_path(w, h, clip_size, overlap_size)
    step = slice_step(clip_size, overlap_size) if slice_mode else 1
    c0, c1 = colors[0], colors[1] if len(colors) >= 2 else colors[0]

    for idx, (x1, y1, x2, y2) in enumerate(
        iter_clip_rects(w, h, clip_size, overlap_size)
    ):
        if not slice_mode or len(colors) < 2:
            color = colors[0]
        else:
            col = x1 // step
            row = y1 // step
            color = c0 if (col + row) % 2 == 0 else c1
        inset = max(0, int(draw_inset_px))
        _draw_rect_inset(
            img_draw, x1, y1, x2, y2, color, thickness, inset
        )
        if draw_index:
            cv2.putText(
                img_draw,
                str(idx),
                (x1 + 4, y1 + 22),
                font,
                0.7,
                index_color,
                2,
                cv2.LINE_AA,
            )
    return img_draw


def save_clip_tile_images(
    image,
    clip_size,
    overlap_size,
    out_dir,
    name_prefix="tile",
    ext=".jpg",
    draw_index=False,
    index_color=(255, 255, 255),
    index_font_scale=0.7,
    index_thickness=2,
):
    """
    按与推理一致的分片矩形裁剪并逐张保存（不画框）。
    当 draw_index=True 时，在每张分片左上角绘制序列号（与 get_clip 遍历顺序一致）。

    :param image: BGR 图像
    :param out_dir: 输出目录（不存在则创建）
    :param name_prefix: 文件名前缀
    :param ext: 保存扩展名，须带点（如 ``.jpg``、``.png``）
    :param draw_index: True 时绘制分片序号到分片图左上角
    :param index_color: 序号文字颜色（BGR）
    :return: 写入的分片数量
    """
    os.makedirs(out_dir, exist_ok=True)
    h, w = image.shape[:2]
    n = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    for idx, (x1, y1, x2, y2) in enumerate(
        iter_clip_rects(w, h, clip_size, overlap_size)
    ):
        tile = image[y1:y2, x1:x2]
        if tile.size == 0:
            logging.warning(
                "分片为空，跳过: idx=%s rect=(%s,%s,%s,%s)",
                idx,
                x1,
                y1,
                x2,
                y2,
            )
            continue
        if draw_index:
            cv2.putText(
                tile,
                str(idx),
                (4, 22),
                font,
                index_font_scale,
                index_color,
                index_thickness,
                cv2.LINE_AA,
            )
        fname = f"{name_prefix}_{idx:03d}_{x1}_{y1}{ext}"
        fpath = os.path.join(out_dir, fname)
        cv2.imwrite(fpath, tile)
        n += 1
    return n


# =============================================================================
# 示例：单文件或目录递归，输出目录保持相对结构（用法对齐 predict_size_daofeishi.py）
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    PIC_EXT = {".jpg", ".jpeg", ".png"}

    # 输入：单张图片路径，或文件夹路径（递归子目录）
    input_path = "/Users/shunyaoyin/Documents/code/datasets/insect-data/test-data/daofeishi-边缘0的问题"
    output_dir = str(input_path) + "_clip_grid"
    clip_size = 640
    overlap_size = 120
    clip_size = 960
    overlap_size = 200
    draw_index = True
    # 每个分片一种颜色；相邻分片（四邻接）交替 palette 中的颜色
    palette = ((0, 255, 255), (255, 0, 255))
    # 画框相对分片矩形向内缩进（像素），减轻重叠处线条糊在一起；仅影响网格效果图描边
    draw_inset_px = 2
    # 为 True 时额外将每个分片裁剪为独立图片，写入「网格效果图」同目录下子文件夹
    save_clip_tiles = True

    input_p = Path(input_path)
    if input_p.is_file():
        image_files = [input_p] if input_p.suffix.lower() in PIC_EXT else []
    elif input_p.is_dir():
        image_files = sorted(
            p for p in input_p.rglob("*") if p.is_file() and p.suffix.lower() in PIC_EXT
        )
    else:
        image_files = []
        logging.warning("路径不存在或不是图片: %s", input_path)

    logging.info("共 %s 张图片", len(image_files))

    for idx, img_path in enumerate(image_files, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            logging.warning("[%s/%s] 无法读取，跳过: %s", idx, len(image_files), img_path)
            continue

        if input_p.is_dir():
            rel_path = img_path.relative_to(input_p)
        else:
            rel_path = Path(img_path.name)

        save_sub_dir = (
            os.path.join(output_dir, str(rel_path.parent))
            if str(rel_path.parent) != "."
            else output_dir
        )
        os.makedirs(save_sub_dir, exist_ok=True)
        out_path = os.path.join(save_sub_dir, rel_path.name)
        ext = rel_path.suffix.lower() if rel_path.suffix else ".jpg"

        drawn = draw_clip_grid(
            img,
            clip_size,
            overlap_size,
            palette=palette,
            draw_index=draw_index,
            draw_inset_px=draw_inset_px,
        )
        cv2.imwrite(out_path, drawn)
        logging.info("[%s/%s] 已保存: %s", idx, len(image_files), out_path)

        if save_clip_tiles:
            tiles_dir = os.path.join(
                save_sub_dir, f"{rel_path.stem}_clip_tiles"
            )
            n_tiles = save_clip_tile_images(
                img,
                clip_size,
                overlap_size,
                tiles_dir,
                name_prefix=rel_path.stem,
                ext=ext,
                draw_index=draw_index,
            )
            logging.info(
                "[%s/%s] 分片图已保存 %s 张: %s",
                idx,
                len(image_files),
                n_tiles,
                tiles_dir,
            )

    logging.info("处理完成")
