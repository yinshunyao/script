#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
融合推理入口：大虫(orig) + 稻飞虱(daofeishi) + 玉米螟(yumiming)
引用各自实现，提供 predict 总入口
"""
import os
import sys
import cv2
import logging
from pathlib import Path

# 确保项目根目录在 path 中
_FILE = Path(__file__).resolve()
_ROOT = _FILE.parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from script.predict_orig import predict as predict_orig
from script.predict_size_daofeishi import PredictSize
from script.predict.model_detect import ior as _ior

# 模型根目录
"""
MODEL_ROOT = os.path.join(_ROOT, "models", "20260123")
if not os.path.isdir(MODEL_ROOT):
    MODEL_ROOT = os.path.join(_ROOT, "model", "20260123")
"""
# MODEL_ROOT = os.path.join(_ROOT, "model")
MODEL_ROOT = os.path.join(_ROOT, "models", "20260123")

SIZE_CONFIG_PATH = _FILE.parent / "size.json"

# daofeishi 需要与 orig/yumiming 做 merge 的类型（IoR > 阈值时滤除 daofeishi）
DAOFEISHI_MERGE_TYPES = frozenset({"baibeifeishi", "huifeishi", "hefeishi"})
IOR_THRESHOLD_MERGE = 0.8

# orig 与 yumiming 共同输出 yumiming 类型时，IoR > 阈值时滤除 orig
IOR_THRESHOLD_ORIG_YUMIMING = 0.8


def _ior_box(loc1, loc2):
    """IoR: 交集面积/最小框面积，引用 model_detect.ior"""
    return _ior(loc1, loc2)


def _merge_daofeishi_with_orig_yumiming(results):
    """
    daofeishi 的 baibeifeishi/huifeishi/hefeishi 与 orig、yumiming 做 merge。
    当 IoR(daofeishi_box, orig_or_yumiming_box) > 0.9 时，滤除 daofeishi 输出。
    """
    orig_yumiming = [r for r in results if r.get("source") in ("orig", "yumiming")]
    daofeishi_merge = [r for r in results if r.get("source") == "daofeishi"
                      and r.get("name") in DAOFEISHI_MERGE_TYPES]
    others = [r for r in results if r not in orig_yumiming and r not in daofeishi_merge]

    to_remove = set()
    for dr in daofeishi_merge:
        loc_d = dr["location"]
        for oy in orig_yumiming:
            loc_oy = oy["location"]
            if _ior_box(loc_d, loc_oy) > IOR_THRESHOLD_MERGE:
                to_remove.add(id(dr))
                break

    merged = orig_yumiming + [r for r in daofeishi_merge if id(r) not in to_remove] + others
    return merged


def _merge_orig_yumiming_with_yumiming(results):
    """
    orig 和 yumiming 模型都输出的 yumiming 类型做 merge。
    当 IoR(orig_yumiming_box, yumiming_box) > 0.8 时，滤除 orig 结果。
    """
    yumiming_results = [r for r in results if r.get("source") == "yumiming"]
    if not yumiming_results:
        return results
    orig_yumiming_type = [r for r in results if r.get("source") == "orig"
                         and r.get("name") == "yumiming"]
    others = [r for r in results if r not in orig_yumiming_type and r not in yumiming_results]

    to_remove = set()
    for or_ in orig_yumiming_type:
        loc_or = or_["location"]
        for yr in yumiming_results:
            loc_yr = yr["location"]
            if _ior_box(loc_or, loc_yr) > IOR_THRESHOLD_ORIG_YUMIMING:
                to_remove.add(id(or_))
                break

    merged = [r for r in orig_yumiming_type if id(r) not in to_remove] + yumiming_results + others
    return merged


def _merge_cls12_with_others(results):
    """
        12类大虫(cls_12) 与 大虫(orig)、玉米螟(yumiming) 以及其自身做 merge。
        采用“高分保留、低分滤除”策略，解决不同模型针对大型昆虫重复检出的问题。
        当 IoR(box_i, box_j) > 0.6 时，滤除置信度(score)较低的结果。
    """
    target_sources = ("cls_12", "orig", "yumiming")
    big_insects = [r for r in results if r.get("source") in target_sources]
    others = [r for r in results if r not in big_insects]

    big_insects.sort(key=lambda x: x.get("score", 0), reverse=True)

    to_remove = set()
    for i in range(len(big_insects)):
        res_i = big_insects[i]
        if id(res_i) in to_remove:
            continue

        for j in range(i + 1, len(big_insects)):
            res_j = big_insects[j]
            if id(res_j) in to_remove:
                continue

            if _ior_box(res_i["location"], res_j["location"]) > 0.8:
                to_remove.add(id(res_j))

    merged = [r for r in big_insects if id(r) not in to_remove] + others
    return merged

def _get_device(device=None):
    """获取推理设备"""
    if device is not None:
        return device
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# 延迟初始化的预测器
_predictor_daofeishi = None
_predictor_yumiming = None
_predictor_cls12 = None


def _get_predictor_daofeishi(device=None):
    """稻飞虱预测器"""
    global _predictor_daofeishi
    if _predictor_daofeishi is None:
        detect_path = os.path.join(MODEL_ROOT, "daofeishi-detect.pt")
        cls_path = os.path.join(MODEL_ROOT, "daofeishi-cls.pt")
        if not os.path.isfile(detect_path):
            detect_path = _FILE.parent / "daofeishi-detect.pt"
        if not os.path.isfile(cls_path):
            cls_path = _FILE.parent / "daofeishi-cls.pt"
        _predictor_daofeishi = PredictSize(
            detect_model_path=str(detect_path),
            size_config_path=str(SIZE_CONFIG_PATH),
            cls_list=["daofeishi"],
            cls_model_path=str(cls_path) if os.path.isfile(cls_path) else None,
            offset_rate=1.2,
            conf_thresh=0.4,
            device=device,
        )
    return _predictor_daofeishi


def _get_predictor_yumiming(device=None):
    """玉米螟预测器"""
    global _predictor_yumiming
    if _predictor_yumiming is None:
        detect_path = os.path.join(MODEL_ROOT, "yumiming-detect.pt")
        cls_path = os.path.join(MODEL_ROOT, "yumiming-cls.pt")
        if not os.path.isfile(detect_path):
            detect_path = _FILE.parent / "yumiming-detect.pt"
        _predictor_yumiming = PredictSize(
            detect_model_path=str(detect_path),
            size_config_path=str(SIZE_CONFIG_PATH),
            cls_list=["yumiming"],
            cls_model_path=str(cls_path) if os.path.isfile(cls_path) else None,
            offset_rate=1.2,
            conf_thresh=0.3,
            device=device,
        )
    return _predictor_yumiming

def _get_predictor_cls12(device=None):
    global _predictor_cls12
    if _predictor_cls12 is None:
        detect_path = os.path.join(MODEL_ROOT, "kuangxuan_0120.pt")
        cls_path = os.path.join(MODEL_ROOT, "cls_12.pt")
        cls_list = [
            "caoditanyee", "fayee", "erdianweiyee", "erhuaming", "daming",
            "mianlingchong", "kuanjingyee", "xiewenyee", "xiumuyee",
            "guajuanyeming", "pingshaoyingyee", "huangchizhuiye"
        ]

        _predictor_cls12 = PredictSize(
            detect_model_path=str(detect_path),
            size_config_path=str(SIZE_CONFIG_PATH),
            cls_list=cls_list,
            cls_model_path=str(cls_path),
            offset_rate=1.2,  # 稍微外扩
            conf_thresh=0.2,  # 置信度阈值
            device=device,
        )
    return _predictor_cls12

def _normalize_result(r, source):
    """统一结果格式: {name, score, location, msg, source}"""
    if "name" in r and "location" in r:
        return {
            "name": r["name"],
            "score": r.get("score", 0),
            "location": r["location"],
            "msg": r.get("msg", ""),
            "source": source,
        }
    # PredictSize 格式
    return {
        "name": r.get("cls_name", r.get("class_name", "unknown")),
        "score": r.get("cls_conf", r.get("conf", 0)),
        "location": [r["x1"], r["y1"], r["x2"], r["y2"]],
        "msg": "",
        "source": source,
    }


def predict(image, device=None, parts=None):
    """
    融合推理总入口

    :param image: 输入图像 (numpy BGR)
    :param device: 设备 'cuda'/'mps'/'cpu'，None 时自动检测
    :param parts: 启用的部分，None 表示默认流程。可选: ['orig', 'daofeishi', 'yumiming', 'cls_12']
    :return: 统一格式结果列表 [{name, score, location, msg, source}, ...]
    """
    if parts is None:
        # 默认去除 yumiming
        parts = ["orig", "daofeishi", "cls_12"]
    dev = _get_device(device)
    results = []

    if "orig" in parts:
        try:
            orig_res = predict_orig(image, device=dev)
            for r in orig_res:
                results.append(_normalize_result(r, "orig"))
        except Exception as e:
            logging.warning(f"大虫推理异常: {e}")

    if "daofeishi" in parts:
        try:
            predictor = _get_predictor_daofeishi(device=dev)
            daofeishi_res = predictor.predict(
                image, clip_size=640, overlap_size=100,
                output=None,
            )
            for r in daofeishi_res:
                nr = _normalize_result(r, "daofeishi")
                if nr["name"] == "other":
                    continue  # 过滤 daofeishi 的 other 类型
                results.append(nr)
        except Exception as e:
            logging.warning(f"稻飞虱推理异常: {e}")

    if "yumiming" in parts:
        try:
            predictor = _get_predictor_yumiming(device=dev)
            yumiming_res = predictor.predict(
                image, clip_size=None, overlap_size=200,
                output=None,
            )
            for r in yumiming_res:
                results.append(_normalize_result(r, "yumiming"))
        except Exception as e:
            logging.warning(f"玉米螟推理异常: {e}")

    if "cls_12" in parts:
        try:
            predictor = _get_predictor_cls12(device=dev)
            cls12_res = predictor.predict(image, clip_size=4000, overlap_size=1000)
            for r in cls12_res:
                score = r.get('cls_conf', 0)
                name = r.get('cls_name', 'unknown')
                if name == "other" and score < 0.05:
                    continue
                nr = _normalize_result(r, "cls_12")
                if nr:  # 配合上面的拦截逻辑
                    results.append(nr)
        except Exception as e:
            logging.warning(f"12类大虫推理异常: {e}")


    # daofeishi 的 baibeifeishi/huifeishi/hefeishi 与 orig、yumiming 做 merge
    results = _merge_daofeishi_with_orig_yumiming(results)
    # 仅在存在 yumiming 来源结果时执行该 merge，避免关闭 yumiming 后的无效处理
    if any(r.get("source") == "yumiming" for r in results):
        # orig 和 yumiming 都输出的 yumiming 类型做 merge，IoR > 0.8 时滤除 orig
        results = _merge_orig_yumiming_with_yumiming(results)

    results = _merge_cls12_with_others(results)

    return results


def draw_results(image, results, output_path=None):
    """
    将检测结果绘制到图片上

    :param image: 原始图像
    :param results: predict 返回的结果列表
    :param output_path: 保存路径，None 则不保存
    :return: 绘制后的图像
    """
    img_draw = image.copy()
    colors = {"orig": (0, 255, 0), "daofeishi": (255, 165, 0), "yumiming": (0, 191, 255), "cls_12": (255, 0, 255)}
    for r in results:
        x1, y1, x2, y2 = r["location"]
        name = r["name"]
        score = r.get("score", 0)
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
    if output_path:
        d = os.path.dirname(output_path)
        if d:
            os.makedirs(d, exist_ok=True)
        cv2.imwrite(output_path, img_draw)
    return img_draw


"""
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 示例：单张图片测试
    # input_path = os.path.join(_ROOT, "data", "稻飞虱和玉米螟", "稻飞虱 0209-测试")
    # input_path = os.path.join(_ROOT, "data", "稻飞虱和玉米螟", "玉米螟02")
    # input_path = "/Users/shunyaoyin/Documents/code/other/insect/data/稻飞虱和玉米螟/玉米螟02/玉米螟一批（腹部）.jpg"
    # input_path = "/Users/shunyaoyin/Documents/code/other/insect/data/稻飞虱和玉米螟/train-data/拍摄/大青叶蝉-1-侧面1_20260107190507466.jpg"
    input_path = "/Users/shunyaoyin/Documents/code/other/insect/data/稻飞虱和玉米螟/玉米螟01_原始/玉米螟 5侧面.jpg"
    # 找一张测试图
    PIC_EXT = {".jpg", ".jpeg", ".png"}
    input_p = Path(input_path)
    if input_p.is_dir():
        image_files = sorted(
            p for p in input_p.rglob("*")
            if p.is_file() and p.suffix.lower() in PIC_EXT
        )
    elif input_p.is_file() and input_p.suffix.lower() in PIC_EXT:
        image_files = [input_p]
    else:
        image_files = []

    if not image_files:
        print("未找到测试图片，请设置 input_path 或放置图片到 data 目录")
        print("使用空白图进行演示")
        import numpy as np
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:] = (240, 240, 240)
    else:
        img_path = image_files[0]
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"无法读取: {img_path}")
            sys.exit(1)
        print(f"测试图片: {img_path}")

    # 推理（可指定 parts 仅测试部分，如 parts=["orig"]）
    results = predict(img, device=None, parts=["orig", "daofeishi", "yumiming", "cls_12"])

    print(f"\n检测到 {len(results)} 个目标:")
    for i, r in enumerate(results, 1):
        print(f"  [{i}] {r['name']} score={r['score']:.2f} "
              f"box={r['location']} source={r['source']} {r.get('msg', '')}")

    # 绘制并保存
    output_path = os.path.join(_ROOT, "predict_merge_result.jpg")
    draw_results(img, results, output_path)
    print(f"\n结果图已保存: {output_path}")
"""

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # input_path = os.path.join(_ROOT, "image")  # 对应你左侧的 image 文件夹
    # output_dir = os.path.join(_ROOT, "our_test_merge")  # 结果保存目录
    # input_path = "/Users/shunyaoyin/Documents/code/other/insect/data/test-data/拼接图片"
    input_path = "/Users/shunyaoyin/Documents/code/other/insect/data/test-data/虫情4模型测试数据2/地老虎（尹）"
    output_dir = input_path + "-result"

    PIC_EXT = {".jpg", ".jpeg", ".png"}
    input_p = Path(input_path)
    image_files = sorted(p for p in input_p.rglob("*") if p.is_file() and p.suffix.lower() in PIC_EXT)

    if not image_files:
        print("未找到测试图片")
    else:
        # --- 修改点：开始循环处理 ---
        for img_p in image_files:
            print(f"\n正在推理: {img_p.name}")
            img = cv2.imread(str(img_p))
            if img is None: continue

            # 执行融合推理
            results = predict(img, device=None, parts=["orig", "daofeishi",])
            # results = predict(img, device=None, parts=["cls_12"])


            # 打印摘要
            print(f"检测到 {len(results)} 个目标")

            # 构造保存路径，保持文件名一致
            if not os.path.exists(output_dir): os.makedirs(output_dir)
            save_path = os.path.join(output_dir, img_p.name)

            draw_results(img, results, save_path)
            print(f"结果已保存: {save_path}")