#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
demo: 批量图片检测 + 结果绘制保存

参考:
- script/predict_size_daofeishi.py（目录遍历与保存方式）
- script/merge_beyond.py（统一推理入口）
"""
import logging
import os
import sys
from pathlib import Path

import cv2

# 确保项目根目录在 path 中
_FILE = Path(__file__).resolve()
_ROOT = _FILE.parents[1]

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from script.merge_beyond import predict, MODEL_ROOT

logging.warning(f"_ROOT:{_ROOT}")
logging.warning(f"MODEL_ROOT:{MODEL_ROOT}")


def draw_results(image, results, output_path=None):
    """
    将 merge_beyond.predict 的结果绘制到图片上并可选保存。

    results 元素格式:
    {
        "name": "xxx",
        "score": 0.95,
        "location": [x1, y1, x2, y2],
        "msg": "",
        "source": "daofeishi|beyond|..."
    }
    """
    img_draw = image.copy()

    # 按来源区分颜色；未识别来源使用灰色
    colors = {
        "daofeishi": (0, 255, 0),
        "beyond": (0, 191, 255),
        "orig": (255, 165, 0),
        "yumiming": (255, 0, 255),
        "cls_12": (128, 255, 0),
    }

    for r in results:
        x1, y1, x2, y2 = r["location"]
        name = r.get("name", "unknown")
        score = r.get("score", 0.0)
        source = r.get("source", "")
        color = colors.get(source, (180, 180, 180))

        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
        label = f"{name} {score:.2f} [{source}]"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.rectangle(img_draw, (x1, y1 - th - baseline - 4), (x1 + tw, y1), color, -1)
        cv2.putText(img_draw, label, (x1, y1 - 4), font, font_scale, (255, 255, 255), thickness)

    if output_path is not None:
        output_parent = os.path.dirname(output_path)
        if output_parent:
            os.makedirs(output_parent, exist_ok=True)
        cv2.imwrite(output_path, img_draw)

    return img_draw


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 支持图片扩展名
    PIC_EXT = {".jpg", ".jpeg", ".png"}

    # 输入: 可填单张图片或目录（目录将递归遍历）
    input_path = "/Users/shunyaoyin/Documents/code/ai-company/insect/data/test-data/虫情4模型测试数据"
    # input_path = "/Users/shunyaoyin/Documents/code/ai-company/insect/data/test-data/稻飞虱 0209-测试"

    # 输出目录: 保存绘制结果；目录输入时保持子目录结构
    output_dir = input_path + "_result"
    # 是否启用 merge_beyond 中的 "merage" 逻辑（沿用原参数名）
    merage = True

    input_p = Path(input_path)
    if input_p.is_file():
        image_files = [input_p] if input_p.suffix.lower() in PIC_EXT else []
    elif input_p.is_dir():
        image_files = sorted(
            p for p in input_p.rglob("*")
            if p.is_file() and p.suffix.lower() in PIC_EXT
        )
    else:
        image_files = []
        print(f"路径不存在: {input_path}")

    print(f"共找到 {len(image_files)} 张图片")

    for idx, img_path in enumerate(image_files, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[{idx}/{len(image_files)}] 无法读取图片，跳过: {img_path}")
            continue

        # 调用统一推理入口
        results = predict(img, merage=merage)

        # 保持输出目录结构与输入一致
        if input_p.is_dir():
            rel_path = img_path.relative_to(input_p)
        else:
            rel_path = Path(img_path.name)
        save_path = Path(output_dir) / rel_path

        draw_results(img, results, str(save_path))

        print(f"[{idx}/{len(image_files)}] {rel_path} 检测到 {len(results)} 个目标")
        for r in results:
            x1, y1, x2, y2 = r["location"]
            print(
                f"    [{r.get('name', 'unknown')}] score={r.get('score', 0):.2f} "
                f"source={r.get('source', '')} box=({x1},{y1},{x2},{y2})"
            )

    print(f"处理完成，结果保存在: {output_dir}")

