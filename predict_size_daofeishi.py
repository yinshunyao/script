#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/02/07
# @Author  : ysy
# @Email   : xxx@qq.com 
# @Detail  : 图片检测和稻飞虱分类算法
# @Software: PyCharm

import json
import logging
import os
import sys
import cv2
from pathlib import Path

# 确保项目根目录在 path 中，便于作为模块导入
_FILE = Path(__file__).resolve()
_ROOT = _FILE.parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from script.predict.model_detect import ModelDetector
from script.predict.model_cls import ModelCls

class PredictSize:
    """
    基于尺寸过滤的检测+分类推理器

    处理流程:
        1. 切片/整图检测 (clip_size, overlap_size 控制)
        2. 根据尺寸配置过滤检测框，判断是否为疑似目标
        3. 对疑似目标调用分类模型进行识别，其余归类为 other
    """

    def __init__(self, detect_model_path, size_config_path,
                 cls_list, cls_model_path=None, offset_rate=1.2,
                 conf_thresh=0.3, conf_merge=0.1,
                 iou_threshold=0.3, ior_threshold=0.4,
                 device=None):
        """
        初始化检测和分类预测器（一次实例化，重复使用）

        :param detect_model_path: 检测模型路径
        :param size_config_path:  尺寸配置文件路径 (size.json)
        :param cls_list:          需要分类的类别列表, 例如 ["hefeishi", "baibeifeishi", "huifeishi"]
        :param cls_model_path:    分类模型路径, 为 None 时跳过分类步骤，仅做检测+尺寸过滤
        :param offset_rate:       尺寸容错系数, 一般 0~2, 默认 1.2
        :param conf_thresh:       检测输出置信度阈值
        :param conf_merge:        merge 前置信度
        :param iou_threshold:     IOU 阈值
        :param ior_threshold:     IOR 阈值
        :param device:            设备类型 ('cuda', 'mps', 'cpu')，None 则自动检测
        """
        self.cls_list = cls_list
        self.offset_rate = offset_rate

        # ---------- 计算尺寸过滤参数（只在 __init__ 中算一次） ----------
        self.size_min, self.size_max = self._compute_size_filter(
            size_config_path, cls_list, offset_rate
        )
        logging.info(
            f"尺寸过滤参数: size_min={self.size_min:.1f}, size_max={self.size_max:.1f}, "
            f"cls_list={cls_list}, offset_rate={offset_rate}"
        )

        # ---------- 加载检测模型 ----------
        self.detector = ModelDetector(
            model_path=detect_model_path,
            conf_thresh=conf_thresh,
            conf_merge=conf_merge,
            iou_threshold=iou_threshold,
            ior_threshold=ior_threshold,
            device=device,
        )

        # ---------- 加载分类模型（可选） ----------
        if cls_model_path is not None:
            self.classifier = ModelCls(model_path=cls_model_path)
        else:
            self.classifier = None
            logging.info("未传递分类模型路径，将跳过分类步骤，仅做检测+尺寸过滤")

    # ------------------------------------------------------------------ #
    #  尺寸配置解析
    # ------------------------------------------------------------------ #
    @staticmethod
    def _compute_size_filter(size_config_path, cls_list, offset_rate):
        """
        根据分类列表和尺寸配置，计算最终的宽高过滤上下限

        逻辑:
            1. 遍历 cls_list 中每个类别，收集 width_px / height_px 的 min/max
            2. 取所有最小值中的最小 -> overall_min
            3. 取所有最大值中的最大 -> overall_max
            4. 应用容错系数: size_min = overall_min / offset_rate
                            size_max = overall_max * offset_rate

        :return: (size_min, size_max)
        """
        with open(size_config_path, "r", encoding="utf-8") as f:
            size_config = json.load(f)

        all_mins = []
        all_maxs = []

        for cls_name in cls_list:
            if cls_name not in size_config:
                logging.warning(f"尺寸配置中未找到类别: {cls_name}，跳过")
                continue
            cfg = size_config[cls_name]
            all_mins.append(cfg["width_px"]["min"])
            all_mins.append(cfg["height_px"]["min"])
            all_maxs.append(cfg["width_px"]["max"])
            all_maxs.append(cfg["height_px"]["max"])

        if not all_mins or not all_maxs:
            raise ValueError(
                f"无法从尺寸配置中获取有效的尺寸信息, cls_list={cls_list}"
            )

        size_min = min(all_mins) / offset_rate
        size_max = max(all_maxs) * offset_rate

        return size_min, size_max

    # ------------------------------------------------------------------ #
    #  尺寸过滤
    # ------------------------------------------------------------------ #
    def _filter_by_size(self, box):
        """
        判断检测框是否在尺寸范围内

        :param box: 检测结果字典，包含 x1, y1, x2, y2
        :return: True -> 疑似需要分类; False -> 超出范围归类 other
        """
        w = box["x2"] - box["x1"]
        h = box["y2"] - box["y1"]

        if w < self.size_min or h < self.size_min:
            return False
        if w > self.size_max or h > self.size_max:
            return False

        return True

    # ------------------------------------------------------------------ #
    #  绘制结果
    # ------------------------------------------------------------------ #
    @staticmethod
    def _draw_results(image, results):
        """
        将检测和分类结果绘制到图片上

        :param image:   原始图像 (numpy array, BGR)
        :param results: predict 返回的结果列表
        :return: 绘制后的图像副本
        """
        img_draw = image.copy()

        # 颜色方案: 分类/检测目标用绿色，other 用灰色
        color_cls = (0, 255, 0)
        color_other = (180, 180, 180)

        for r in results:
            x1, y1, x2, y2 = r["x1"], r["y1"], r["x2"], r["y2"]
            cls_name = r.get("cls_name", "unknown")
            cls_conf = r.get("cls_conf", 0.0)
            det_conf = r.get("conf", 0.0)

            is_other = (cls_name == "other")
            color = color_other if is_other else color_cls

            # 绘制矩形框
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)

            # 标签文字：先检出置信度，再分类置信度
            label = f"{cls_name} det:{det_conf:.2f} cls:{cls_conf:.2f}"

            # 文字背景
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(img_draw,
                          (x1, y1 - th - baseline - 4),
                          (x1 + tw, y1),
                          color, -1)
            cv2.putText(img_draw, label, (x1, y1 - 4),
                        font, font_scale, (255, 255, 255), thickness)

        return img_draw

    # ------------------------------------------------------------------ #
    #  检测 + 分类 推理
    # ------------------------------------------------------------------ #
    def predict(self, image, clip_size=2500, overlap_size=800,
                output=None, image_name=None, debug=False, debug_clip=False,
                edge_reject_distance=5):
        """
        完成检测和分类推理

        流程:
            1. 切片/整图 -> 调用 detect 模型
            2. 对每个检测框按尺寸判断是否为疑似目标
            3. 疑似目标裁剪后送入分类模型
            4. 其余归类为 other
            5. 如果指定 output 目录，将结果绘制到图片并保存

        :param image:        输入图像 (numpy array, BGR)
        :param clip_size:    切片大小, >= 全图时不切片
        :param overlap_size: 切片重叠大小 (平移算法)
        :param edge_reject_distance: 切片边缘过滤阈值（像素），<=0 表示关闭
        :param output:       保存目录路径, 为 None 则不保存绘制结果
        :param image_name:   保存的文件名 (如 "test.jpg"), 为 None 时使用默认名 "result.jpg"
        :param debug:        调试模式
        :return: 结果列表, 每个元素为 dict:
                 {x1, y1, x2, y2, conf, cls_id, class_name, detect_id,
                  cls_name, cls_conf}
        """
        # ---- 第 1 步: 检测 ----
        detections = self.detector.predict(
            image,
            clip_size=clip_size,
            overlap_size=overlap_size,
            debug=debug, debug_clip=debug_clip,
            edge_reject_distance=edge_reject_distance,
        )

        h_img, w_img = image.shape[:2]
        results = []

        for det in detections:
            # ---- 第 2 步: 尺寸过滤 ----
            if self._filter_by_size(det):
                if self.classifier is not None:
                    # 裁剪检测框区域（确保坐标不越界）
                    x1 = max(0, det["x1"])
                    y1 = max(0, det["y1"])
                    x2 = min(w_img, det["x2"])
                    y2 = min(h_img, det["y2"])

                    crop = image[y1:y2, x1:x2]
                    if crop.size == 0:
                        det["cls_name"] = "other"
                        det["cls_conf"] = 0.0
                        results.append(det)
                        continue

                    # ---- 第 3 步: 分类 ----
                    dev = getattr(self.detector, "device", None)
                    cls_result = self.classifier.predict(crop, device=dev)
                    if cls_result is not None:
                        det["cls_name"] = cls_result["class_name"]
                        det["cls_conf"] = cls_result["conf"]
                    else:
                        det["cls_name"] = "other"
                        det["cls_conf"] = 0.0
                else:
                    # 无分类模型，直接使用检测模型输出的类别
                    det["cls_name"] = det.get("class_name", "unknown")
                    det["cls_conf"] = det.get("conf", 0.0)
            else:
                # 不在尺寸范围内 -> other
                det["cls_name"] = "other"
                det["cls_conf"] = 0.0

            results.append(det)

        # ---- 第 4 步: 保存绘制结果 ----
        if output is not None:
            os.makedirs(output, exist_ok=True)
            save_name = image_name if image_name else "result.jpg"
            save_path = os.path.join(output, save_name)
            img_draw = self._draw_results(image, results)
            cv2.imwrite(save_path, img_draw)
            logging.info(f"结果图片已保存: {save_path}")

        return results

    # ------------------------------------------------------------------ #
    #  释放资源
    # ------------------------------------------------------------------ #
    def release(self):
        """释放检测模型和分类模型占用的资源"""
        if hasattr(self, "detector") and self.detector is not None:
            del self.detector.model
            self.detector = None

        if hasattr(self, "classifier") and self.classifier is not None:
            del self.classifier.model
            self.classifier = None

        logging.info("模型资源已释放")

current_dir = Path(__file__).parent
size_config_path = current_dir / "size.json"
# ====================================================================== #
#  使用示例 — 支持给定文件夹，遍历目录及子目录下的图片
# ====================================================================== #
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 支持的图片扩展名
    PIC_EXT = {".jpg", ".jpeg", ".png"}

    # daofeishi
    cls_list = ['daofeishi']

    model_path = current_dir.parent / "models" / "20260123"
    # detect_model_path = model_path / "daofeishi-detect.pt"
    detect_model_path = model_path / "daofeishi-detect-0320.pt"
    # detect_model_path = model_path / "kuangxuan_0209.pt"
    # detect_model_path = model_path / "daofeishi-detect-0320.pt"
    cls_model_path = model_path / "daofeishi-cls.pt"
    # 输入：可以是单张图片路径，也可以是文件夹路径（递归遍历子目录）
    input_path = '/Users/shunyaoyin/Documents/code/ai-company/insect/data/test-data/虫情4模型测试数据'
    # input_path = '/Users/shunyaoyin/Documents/code/ai-company/insect/data/test-data/稻飞虱 0209-测试'
    # input_path = '/Users/shunyaoyin/Documents/code/ai-company/insect/data/test-data/虫情4模型测试数据/混合'
    # 输出目录：保存绘制结果（保持与输入相同的子目录结构和文件名）
    output_dir = input_path + "_big"
    clip_size = 640
    overlap_size = 120
    edge_reject_distance = 0
    predict_debug = False
    debug_clip = False
    conf_thresh = 0.65
    # clip_size = 1280
    # overlap_size = 100

    predictor = PredictSize(
        detect_model_path=detect_model_path,
        size_config_path=size_config_path,
        cls_list=cls_list,
        cls_model_path=cls_model_path,   # 可选，设为 None 则跳过分类
        offset_rate=1.2,
        conf_thresh=conf_thresh,
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
            edge_reject_distance=edge_reject_distance,
            output=save_sub_dir,
            image_name=rel_path.name,
            debug=predict_debug,
            debug_clip=debug_clip
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
