#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/02/07
# @Author  : ysy
# @Email   : xxx@qq.com 
# @Detail  : 图片检测和稻飞虱分类算法
# @Software: PyCharm
import logging
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

import cv2
from predict_size_daofeishi import PredictSize, current_dir, Path


def _write_pascal_voc_xml(
    xml_path: str,
    folder_name: str,
    image_filename: str,
    width: int,
    height: int,
    depth: int,
    results,
):
    """
    将检测结果写成 Pascal VOC 格式的单个 xml 文件。
    results: predict 返回的列表，元素含 x1,y1,x2,y2, cls_name, conf 等。
    """
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = folder_name or ""
    ET.SubElement(annotation, "filename").text = image_filename
    src = ET.SubElement(annotation, "source")
    ET.SubElement(src, "database").text = "Unknown"
    size_el = ET.SubElement(annotation, "size")
    ET.SubElement(size_el, "width").text = str(int(width))
    ET.SubElement(size_el, "height").text = str(int(height))
    ET.SubElement(size_el, "depth").text = str(int(depth))
    ET.SubElement(annotation, "segmented").text = "0"

    for r in results:
        x1 = int(round(max(0, min(r["x1"], width - 1))))
        y1 = int(round(max(0, min(r["y1"], height - 1))))
        x2 = int(round(max(0, min(r["x2"], width))))
        y2 = int(round(max(0, min(r["y2"], height))))
        if x2 <= x1:
            x2 = min(width, x1 + 1)
        if y2 <= y1:
            y2 = min(height, y1 + 1)

        obj = ET.SubElement(annotation, "object")
        name = r.get("cls_name", r.get("class_name", "unknown"))
        ET.SubElement(obj, "name").text = str(name)
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        bnd = ET.SubElement(obj, "bndbox")
        ET.SubElement(bnd, "xmin").text = str(x1)
        ET.SubElement(bnd, "ymin").text = str(y1)
        ET.SubElement(bnd, "xmax").text = str(x2)
        ET.SubElement(bnd, "ymax").text = str(y2)

    rough = ET.tostring(annotation, encoding="utf-8")
    parsed = minidom.parseString(rough)
    pretty = parsed.toprettyxml(indent="\t", encoding="utf-8")
    os.makedirs(os.path.dirname(xml_path) or ".", exist_ok=True)
    with open(xml_path, "wb") as f:
        f.write(pretty)

# ====================================================================== #
#  使用示例 — 支持给定文件夹，遍历目录及子目录下的图片
# ====================================================================== #
if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    # 支持的图片扩展名
    PIC_EXT = {".jpg", ".jpeg", ".png"}

    # yumiming
    # cls_list = [
    #     'daozongjuanyeming', 'dongfangnianchong', 'erhuaming', 'huangchizhuiyeyeming',
    #     'laoshinianchong', 'yumiming', 'bazidilaohu-bei', 'bazidilaohu-fu', 'daming', 'taozhuming', 'yinwenyee']
    cls_list = None
    cls_model_path = None
    detect_model_path = "/Users/shunyaoyin/Documents/code/models/kuangxuan_0209.pt"
    # detect_model_path = "/Users/shunyaoyin/Documents/code/models/big-041501.pt"
    # detect_model_path = "/Users/shunyaoyin/Documents/code/models-temp/train-big-0415-s5/weights/best.pt"
    # detect_model_path = "/Users/shunyaoyin/Downloads/best.pt"
    # detect_model_path = "/Users/shunyaoyin/Downloads/epoch122.pt"
    cls_model_path = "/Users/shunyaoyin/Downloads/epoch169.pt"
    # 输入：可以是单张图片路径，也可以是文件夹路径（递归遍历子目录）
    input_path = "/Users/shunyaoyin/Documents/code/datasets/insect-data/test-data/dachong-测试数据集"
    # input_path = "/Users/shunyaoyin/Documents/code/datasets/insect-data/temp/caotan-test"
    # input_path = "/Users/shunyaoyin/Documents/code/datasets/insect-data/test-data/dachong-honghe-temp/2044762495767023616-草地贪夜蛾.jpg"
    # input_path = "/Users/shunyaoyin/Documents/code/datasets/insect-data/test-data/dachong-honghe-temp"
    # input_path = "/Users/shunyaoyin/Documents/code/datasets/insect-data/test-data/2039327322625220608.jpg"
    # 输出目录：保存绘制结果（保持与输入相同的子目录结构和文件名）
    output_dir = input_path + "_0422_cls"
    # 5472 * 3648
    clip_size = 0
    # clip_size = 3000
    # clip_size = 3648
    overlap_size = 600
    # 正常参数
    conf_thresh = 0.3
    edge_reject_distance = 5
    edge_reject_conf_threshold = 1
    edge_reject_cls_conf_threshold = 0.66
    # 分类 top1 置信度：仅当大于该门限才保留类别名，否则标为 other（与边缘联合过滤无关）
    cls_top1_conf_threshold = 0.3

    predict_debug = False
    debug_clip = False
    cls_pad_square = True
    # 漏检排查
    # edge_reject_conf_threshold = 0
    # edge_reject_distance = 0
    # conf_thresh = 0.3
    # clip_size = 1280
    # overlap_size = 200
    # size_config_path = current_dir / "size.json"
    size_config_path = None
    # clip_size = 1280
    # overlap_size = 100

    predictor = PredictSize(
        detect_model_path=detect_model_path,
        size_config_path=size_config_path,
        cls_list=cls_list,
        cls_model_path=cls_model_path,  # 可选，设为 None 则跳过分类
        offset_rate=1.2,
        conf_thresh=conf_thresh,
        device=None,  # 自动检测
        augment=False,

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

        result_output_dir = save_sub_dir
        results = predictor.predict(
            img, clip_size=clip_size, overlap_size=overlap_size,
            edge_reject_distance=edge_reject_distance,
            edge_reject_conf_threshold=edge_reject_conf_threshold,
            edge_reject_cls_conf_threshold=edge_reject_cls_conf_threshold,
            cls_top1_conf_threshold=cls_top1_conf_threshold,
            output=result_output_dir,
            image_name=rel_path.name,
            debug=predict_debug,
            debug_clip=debug_clip,
            cls_pad_square=cls_pad_square
        )

        # 与保存结果图一致：需要写 output 图时，同步输出 Pascal VOC 同名 xml
        if result_output_dir is not None:
            h_img, w_img = img.shape[:2]
            depth = 3 if img.ndim >= 3 else 1
            xml_name = Path(rel_path.name).stem + ".xml"
            xml_path = os.path.join(result_output_dir, xml_name)
            _write_pascal_voc_xml(
                xml_path,
                folder_name=os.path.basename(os.path.normpath(result_output_dir)) or "",
                image_filename=rel_path.name,
                width=w_img,
                height=h_img,
                depth=depth,
                results=results,
            )

        print(f"[{idx}/{len(image_files)}] {rel_path}  检测到 {len(results)} 个目标")
        for r in results:
            print(
                f"    [{r['cls_name']}] conf={r.get('cls_conf', 0):.2f}  "
                f"det_conf={r['conf']:.2f}  "
                f"box=({r['x1']},{r['y1']},{r['x2']},{r['y2']})"
                f"wh={r['x2'] - r['x1']},{r['y2'] - r['y1']}"
            )

    # ---- 释放 ----
    predictor.release()
    print("处理完成")