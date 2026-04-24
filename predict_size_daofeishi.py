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
                 device=None, augment=False, cls_pad_square=False,
                 cls_gray_binarize=False):
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
        :param cls_pad_square:    分类前是否将非正方形裁剪白边补成正方形（与训练白边正方形一致）
        :param cls_gray_binarize: 分类前是否先做灰度+CLAHE+Otsu 二值化再扩成三通道 BGR（默认关）
        """
        self.cls_list = cls_list
        self.offset_rate = offset_rate

        # ---------- 计算尺寸过滤参数（只在 __init__ 中算一次） ----------
        if size_config_path is None:
            self.size_max = None
            self.size_min = None
            logging.info(f"无尺寸过滤参数")
        else:
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
            device=device, augment=augment
        )

        # ---------- 加载分类模型（可选） ----------
        if cls_model_path is not None:
            self.classifier = ModelCls(
                model_path=cls_model_path,
                pad_square=cls_pad_square,
                gray_binarize=cls_gray_binarize,
            )
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

        # size_config_path 为 None 时不做尺寸过滤，全部视为疑似目标
        if self.size_min is None or self.size_max is None:
            return True

        if w < self.size_min or h < self.size_min:
            return False
        if w > self.size_max or h > self.size_max:
            return False

        return True

    # ------------------------------------------------------------------ #
    #  分片检测判定（与 model_detect.ModelDetector.predict 入口一致）
    # ------------------------------------------------------------------ #
    @staticmethod
    def _uses_clip_predict_path(w, h, clip_size, overlap_size):
        """
        True 表示走切片循环（含单张大图仅一块切片的情况），作图时可标「框到整图边缘」像素。
        与 script/predict/model_detect.py 中 predict 的分支条件保持一致。
        """
        if not clip_size or not overlap_size and (
            clip_size >= w and clip_size >= h or clip_size <= overlap_size <= 1
        ):
            return False
        return True

    @staticmethod
    def _parse_detect_clip_origin(detect_id):
        """
        解析 model_detect 切片推理写入的 detect_id，格式为 "{clip_x1}-{clip_y1}"。
        :return: (clip_x1, clip_y1) 或解析失败为 None
        """
        if not detect_id or not isinstance(detect_id, str):
            return None
        parts = detect_id.split("-", 1)
        if len(parts) != 2:
            return None
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            return None

    @staticmethod
    def _min_dist_to_clip_edge(x1, y1, x2, y2, clip_x1, clip_y1, clip_size, w_img, h_img):
        """
        检测框（全图坐标）到当前切片矩形四边的距离最小值，与 model_detect._is_near_clip_edge 语义一致（全图坐标版）。
        切片右下与 get_clip 一致：clip_x2 = min(w, clip_x1+clip_size)，clip_y2 同理。
        """
        clip_x2 = min(w_img, clip_x1 + clip_size)
        clip_y2 = min(h_img, clip_y1 + clip_size)
        d_left = x1 - clip_x1
        d_right = clip_x2 - x2
        d_top = y1 - clip_y1
        d_bottom = clip_y2 - y2
        return min(d_left, d_right, d_top, d_bottom)

    # ------------------------------------------------------------------ #
    #  绘制结果
    # ------------------------------------------------------------------ #
    @staticmethod
    def _draw_results(image, results, draw_edge_px=False, clip_size=None, debug_filter_palette=False):
        """
        将检测和分类结果绘制到图片上

        :param image:         原始图像 (numpy array, BGR)
        :param results:       predict 返回的结果列表
        :param draw_edge_px:  True 时在标签区增加一行「edge-N」：框到所属切片边缘的像素距离最小值 N
        :param clip_size:     与推理时切片边长一致，用于由 detect_id 还原切片右下边界
        :param debug_filter_palette: True 时按 filter 上色：未过滤红、已过滤灰（调试用）
        :return: 绘制后的图像副本
        """
        img_draw = image.copy()
        h_img, w_img = image.shape[:2]

        # 颜色方案: 对外输出图与 debug 一致——有效/输出框红色，other 灰色；debug_filter 模式为已过滤灰
        color_out = (0, 0, 255)
        color_other = (180, 180, 180)
        color_drop_debug = (180, 180, 180)

        for r in results:
            x1, y1, x2, y2 = r["x1"], r["y1"], r["x2"], r["y2"]
            cls_name = r.get("cls_name", "unknown")
            cls_conf = r.get("cls_conf", 0.0)
            det_conf = r.get("conf", 0.0)

            if debug_filter_palette:
                color = color_drop_debug if r.get("filter") else color_out
            else:
                is_other = (cls_name == "other")
                color = color_other if is_other else color_out

            # 绘制矩形框
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)

            # 标签文字：先检出置信度，再分类置信度
            label = f"{cls_name} det:{det_conf:.2f} cls:{cls_conf:.2f}"
            edge_label = None
            if draw_edge_px and clip_size:
                m = r.get("edge_min_dist")
                if m is None:
                    origin = PredictSize._parse_detect_clip_origin(r.get("detect_id", ""))
                    if origin is not None:
                        cx1, cy1 = origin
                        m = PredictSize._min_dist_to_clip_edge(
                            x1, y1, x2, y2, cx1, cy1, clip_size, w_img, h_img
                        )
                if m is not None:
                    edge_label = f"edge-{int(round(m))}"

            # 文字背景
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            # 第一行类别/置信度，第二行（可选）edge-到切片边缘最小像素距离
            line_specs = [(label, font_scale, thickness)]
            if edge_label:
                line_specs.append((edge_label, 0.5, 1))

            # 自下而上叠行：最底行 baseline 贴近 y1（与原单行逻辑一致）
            gap = 2
            rows = []
            baseline_y = y1 - 4
            for line, fs, thk in reversed(line_specs):
                (tw, th), bl = cv2.getTextSize(line, font, fs, thk)
                rows.append((line, fs, thk, tw, th, bl, baseline_y))
                baseline_y -= th + bl + gap

            max_tw = max(r[3] for r in rows)
            pad_x = 4
            bg_bottom = y1
            bg_top = baseline_y + gap
            bg_left = x1
            bg_right = x1 + max_tw + pad_x
            if bg_top < 0:
                dy = -bg_top
                bg_top += dy
                bg_bottom += dy
                rows = [
                    (ln, fs, thk, tw, th, bl, by + dy)
                    for (ln, fs, thk, tw, th, bl, by) in rows
                ]

            cv2.rectangle(
                img_draw, (bg_left, bg_top), (bg_right, bg_bottom), color, -1
            )
            for line, fs, thk, _tw, _th, _bl, by in rows:
                cv2.putText(
                    img_draw, line, (x1 + 2, by),
                    font, fs, (255, 255, 255), thk,
                )

        return img_draw

    # ------------------------------------------------------------------ #
    #  检测 + 分类 推理
    # ------------------------------------------------------------------ #
    def predict(self, image, clip_size=2500, overlap_size=800,
                output=None, image_name=None, debug=False, debug_clip=False,
                edge_reject_distance=5,
                edge_reject_conf_threshold=None,
                edge_reject_cls_conf_threshold=None,
                cls_top1_conf_threshold=None,
                cls_pad_square=None,
                cls_gray_binarize=None,
                detect_pad_square=True,
                return_full_final=False):
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
        :param edge_reject_distance: merge 之后到切片边缘距离阈值（像素），<=0 表示不按距离滤除
        :param edge_reject_conf_threshold: 与距离联合过滤的检测置信度阈值（不含），None 时使用检测器 conf_thresh
        :param edge_reject_cls_conf_threshold: 与距离联合过滤的分类置信度阈值（不含），None 时默认 0.0（等价不因分类置信度触发）
        :param cls_top1_conf_threshold: 分类 top1 置信度门限；不为 None 时，仅当 top1 置信度 **大于** 该值才保留模型类别名，
                否则将 cls_name 置为 other（cls_conf 仍为 top1 原始值便于排查）。None 表示不做该判定。
        :param cls_pad_square: 本次推理是否对分类裁剪做白边正方形 padding；None 时用构造 PredictSize 时的默认值
        :param cls_gray_binarize: 本次是否对分类裁剪做灰度+CLAHE+Otsu；None 时用构造时的默认值
        :param detect_pad_square: 检测切片不足 clip_size 时是否补成正方形（原图居中、黑边）；默认 True
        :param return_full_final: 为 True 时返回含 filter 标记在内的全部 final_results；默认 False 仅返回未过滤框
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
            padding=bool(detect_pad_square),
            debug=debug, debug_clip=debug_clip,
            # 边缘过滤依赖分类置信度，因此在本层（分类后）统一处理；
            # detector 层传 0 仅用于保留 edge_min_dist 字段，不在 detector 层做丢弃。
            edge_reject_distance=0,
            edge_reject_conf_threshold=None,
        )

        h_img, w_img = image.shape[:2]
        final_results = []

        for det in detections:
            det = dict(det)
            det.setdefault("filter", False)
            if det.get("filter"):
                final_results.append(det)
                continue

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
                        final_results.append(det)
                        continue

                    # ---- 第 3 步: 分类 ----
                    dev = getattr(self.detector, "device", None)
                    cls_result = self.classifier.predict(
                        crop,
                        device=dev,
                        pad_square=cls_pad_square,
                        gray_binarize=cls_gray_binarize,
                    )
                    if cls_result is not None:
                        det["cls_name"] = cls_result["class_name"]
                        det["cls_conf"] = cls_result["conf"]
                        det["cls_top3"] = cls_result.get("top3", [])
                        if cls_top1_conf_threshold is not None:
                            if det["cls_conf"] <= float(cls_top1_conf_threshold):
                                det["cls_name"] = "other"
                    else:
                        det["cls_name"] = "other"
                        det["cls_conf"] = 0.0
                        det["cls_top3"] = []
                else:
                    # 无分类模型，直接使用检测模型输出的类别
                    det["cls_name"] = det.get("class_name", "unknown")
                    det["cls_conf"] = det.get("conf", 0.0)
                    det["cls_top3"] = []
            else:
                # 不在尺寸范围内 -> 标记过滤，不跑分类
                det["cls_name"] = "other"
                det["cls_conf"] = 0.0
                det["cls_top3"] = []
                det["filter"] = True

            final_results.append(det)

        # ---- 第 3.5 步: merge 后边缘距离过滤（依赖分类结果）----
        if edge_reject_cls_conf_threshold is None:
            edge_reject_cls_conf_threshold = 0.0
        det_conf_threshold = (
            self.detector.conf_thresh
            if edge_reject_conf_threshold is None
            else float(edge_reject_conf_threshold)
        )
        if edge_reject_distance is not None and edge_reject_distance > 0:
            for det in final_results:
                if det.get("filter"):
                    continue
                # 分类为 other 的不处理（不做该边缘距离过滤）
                if det.get("cls_name") == "other":
                    continue
                dm = det.get("edge_min_dist", None)
                if dm is None:
                    continue
                try:
                    dm = float(dm)
                except Exception:
                    continue
                # 边缘距离大于等于阈值：有效，保留
                if dm >= float(edge_reject_distance):
                    continue
                # 边缘距离小于阈值：不直接过滤，仅当检测置信度或分类置信度低于阈值时过滤
                if det.get("conf", 0.0) < det_conf_threshold or det.get("cls_conf", 0.0) < float(edge_reject_cls_conf_threshold):
                    det["filter"] = True

        # todo 红河临时方案，灰飞虱转成白背飞虱
        for det in final_results:
            cls_name = det["cls_name"]
            if cls_name == "huifeishi":
                det["cls_name"] = "baifeifeishi"

            if cls_name == "zijiaochie":
                det["cls_name"] = "zitiaochie"

        # ---- 第 4 步: 保存绘制结果 ----
        if output is not None:
            os.makedirs(output, exist_ok=True)
            save_name = image_name if image_name else "result.jpg"
            save_path = os.path.join(output, save_name)
            draw_edge = self._uses_clip_predict_path(
                w_img, h_img, clip_size, overlap_size
            )
            draw_rows = final_results if debug else [r for r in final_results if not r.get("filter")]
            img_draw = self._draw_results(
                image,
                draw_rows,
                draw_edge_px=draw_edge,
                clip_size=clip_size,
                debug_filter_palette=debug,
            )
            cv2.imwrite(save_path, img_draw)
            logging.info(f"结果图片已保存: {save_path}")

        if return_full_final:
            return final_results
        return [r for r in final_results if not r.get("filter")]

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
    # detect_model_path = model_path / "daofeishi-detect-0211.pt"
    # detect_model_path = model_path / "daofeishi-detect-0320.pt"
    # detect_model_path = model_path / "kuangxuan_0209.pt"
    detect_model_path = model_path / "daofeishi-detect-0405.pt"
    # detect_model_path = model_path / "daofeishi-detect-040502.pt"
    detect_model_path = Path('/Users/shunyaoyin/Documents/code/models/daofeishi-detect-0405.pt')
    # detect_model_path = Path('/Users/shunyaoyin/Documents/code/models-temp/train-daofeishi-041501/weights/best.pt')
    cls_model_path = Path('/Users/shunyaoyin/Documents/code/models/daofeishi-cls.pt')
    # cls_model_path = Path('/Users/shunyaoyin/Downloads/train5/weights/epoch28.pt')
    # 输入：可以是单张图片路径，也可以是文件夹路径（递归遍历子目录）
    # input_path = '/Users/shunyaoyin/Documents/code/datasets/insect-data/test-data/daofeishi-边缘0的问题'
    input_path = '/Users/shunyaoyin/Documents/code/datasets/insect-data/test-data/daofeishi-测试数据集'
    # input_path = '/Users/shunyaoyin/Documents/code/datasets/insect-data/test-data/daofeishi-边缘0的问题/2039010078717022208_018_1040_1040.jpg'
    # input_path = '/Users/shunyaoyin/Documents/code/datasets/insect-data/test-data/daofeishi-边缘0的问题/00_20260414145928_103_95.png'
    # input_path = '/Users/shunyaoyin/Documents/code/datasets/insect-data/test-data/daofeishi-边缘0的问题'
    # input_path = '/Users/shunyaoyin/Documents/code/datasets/insect-data/test-data/daofeishi-边缘0的问题/2039010078717022208_006_760_760-切片漏检.jpg'
    input_path = '/Users/shunyaoyin/Downloads/2044389074738368512.jpg'
    # 输出目录：保存绘制结果（保持与输入相同的子目录结构和文件名）
    output_dir = input_path + "_0405"
    clip_size = 640
    overlap_size = 120
    # 正常参数
    conf_thresh = 0.3
    edge_reject_distance = 1
    edge_reject_conf_threshold = 0.5
    edge_reject_cls_conf_threshold = 0.66

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
            edge_reject_conf_threshold=edge_reject_conf_threshold,
            edge_reject_cls_conf_threshold=edge_reject_cls_conf_threshold,
            output=save_sub_dir,
            image_name=rel_path.name,
            debug=predict_debug,
            debug_clip=debug_clip,
            cls_pad_square=cls_pad_square
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
