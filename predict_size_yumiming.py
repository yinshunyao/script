#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/02/07
# @Author  : ysy
# @Email   : xxx@qq.com 
# @Detail  : 图片检测和稻飞虱分类算法
# @Software: PyCharm
import logging
import os
import cv2
from predict_size_daofeishi import PredictSize, size_config_path, current_dir, Path

# ====================================================================== #
#  使用示例 — 支持给定文件夹，遍历目录及子目录下的图片
# ====================================================================== #
if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    # 支持的图片扩展名
    PIC_EXT = {".jpg", ".jpeg", ".png"}

    # yumiming
    cls_list = ['yumiming']
    # detect_model_path =  current_dir / "yumiming-detect.pt"
    detect_model_path = "/Users/shunyaoyin/Downloads/train7/epoch152.pt"
    cls_model_path = None
    # 输入：可以是单张图片路径，也可以是文件夹路径（递归遍历子目录）
    input_path = "/Users/shunyaoyin/Documents/code/other/insect/data/稻飞虱和玉米螟/玉米螟02"
    # input_path = "/Users/shunyaoyin/Documents/code/other/insect/data/稻飞虱和玉米螟/拍摄"
    # 输出目录：保存绘制结果（保持与输入相同的子目录结构和文件名）
    output_dir = input_path + "_result0215_best"
    clip_size = None
    overlap_size = 200

    # cls_list = ["hefeishi", "baibeifeishi", "huifeishi"]
    # ---- 实例化（只需一次） ----
    # cls_model_path 为可选参数:
    #   - 传入路径: 检测 + 尺寸过滤 + 分类识别
    #   - 不传 / None: 仅检测 + 尺寸过滤，直接使用检测模型输出的类别
    predictor = PredictSize(
        detect_model_path=detect_model_path,
        size_config_path=size_config_path,
        cls_list=cls_list,
        cls_model_path=cls_model_path,   # 可选，设为 None 则跳过分类
        offset_rate=1.2,
        conf_thresh=0.3,
        ior_threshold=0.7,
        device=None,  # 自动检测
    )

    # ---- 收集图片列表 ----
    input_p = Path(input_path)
    if input_p.is_file():
        image_files = [input_p] if input_p.suffix.lower() in PIC_EXT else []
    elif input_p.is_dir():
        image_files = sorted(
            p for p in input_p.rglob("*") if p.is_file() and p.suffix.lower() in PIC_EXT
        )
    else:
        image_files = []
        print(f"路径不存在: {input_path}")

    print(f"共找到 {len(image_files)} 张图片")

    # ---- 逐张推理 ----
    for idx, img_path in enumerate(image_files, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[{idx}/{len(image_files)}] 无法读取图片，跳过: {img_path}")
            continue

        # 计算相对路径，保持输出目录结构与输入一致
        if input_p.is_dir():
            rel_path = img_path.relative_to(input_p)
        else:
            rel_path = Path(img_path.name)

        # 输出子目录 = output_dir / 相对父目录
        save_sub_dir = os.path.join(output_dir, str(rel_path.parent)) if str(rel_path.parent) != "." else output_dir

        results = predictor.predict(
            img, clip_size=clip_size, overlap_size=overlap_size,
            output=save_sub_dir,
            image_name=rel_path.name,
        )

        print(f"[{idx}/{len(image_files)}] {rel_path}  检测到 {len(results)} 个目标")
        for r in results:
            print(
                f"    [{r['cls_name']}] conf={r.get('cls_conf', 0):.2f}  "
                f"det_conf={r['conf']:.2f}  "
                f"box=({r['x1']},{r['y1']},{r['x2']},{r['y2']})"
                f"wh={r['x2']-r['x1']},{r['y2']-r['y1']}"
            )

    # ---- 释放 ----
    predictor.release()
    print("处理完成")
