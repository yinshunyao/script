# 比昂合并代码
# @author  yaojianbo
# @date  2026-03-09 16:03:01
import os
import sys
import numpy as np
from pathlib import Path

import cv2

# 确保项目根目录在 path 中
_FILE = Path(__file__).resolve()
_ROOT = _FILE.parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

MODEL_ROOT = os.path.join(_ROOT, "model")

SIZE_CONFIG_PATH = _FILE.parent / "size.json"

base_path = os.path.join(_ROOT, "models", "20260123")
if not os.path.isdir(base_path):
    base_path = os.path.join(_ROOT, "model", "20260123")

# from script.predict.model_detect_bejing_fixed import ModelDetector
from script.predict.model_detect import ModelDetector
from script.predict.model_cls import ModelCls

from script.predict_size_daofeishi import PredictSize

all_list = ['yumiming', 'xiaoguantouxishuai', 'dongfanglougu', 'huajinglvwenhuang', 'huangheyilijingui', 'tonglvyilijingui', 'anheisaijingui',
               'zhonghuaxiaobianxijingui', 'tubeibanhongchun', 'huangzuliechun', 'daozongjuanyeming', 'dongfangzhanchong', 'laoshizhanchong', 'daming',
               'meiguijinyee', 'zitiaochie', 'zhongjinhuyee', 'niaozuihuyee', 'taozhuming', 'yinwenyee', 'xuanqiyee','ganlanyee', 'mianlingchong',
               'bazidilaohu', 'dadilaohu', 'xiaodilaohu', 'huangdilaohu', 'baitiaoyee', 'xianweiyee', 'douyeming', 'meijinyee', 'zhongdaisanjiaoyee',
               'chaoqiaoyee', 'dingdianzuanyee', 'daomingling', 'huangbanqijiaomingduobanshuiming', 'yezhanxuyeming', 'sangming', 'baiyangzhuiyeyeming',
               'duowenkuyee', 'huishuangxiancie', 'daodue', 'yibeilue', 'sibanhongchun', 'huangjianqingchun', 'xiangjiaomuxijingui', 'hongjiaolvlijingui',
               'baowenbankoujia', 'xixiongkoujia', 'jiarongtianniu', 'huangxiongsanbaiyi', 'qingbeixiewentiane', 'baixingtiane', 'caoditanyee', 'erhuaming',
               'erdianweiyee', 'xiumuyee', 'fayee', 'pingshaoyingyee', 'kuanjingyee', 'xiewenyee', 'huangchizhuiyeyeming', 'guajuanming', 'ganlanyee',
               'sanhuaming', 'yancaoyee', 'zhonghualanyee', 'huangyeming', 'bailanijuanyeming', 'chenwudenge', 'badianhuidenge', 'shanguangmeiyee',
               'fendiedenge', 'chifuwudenge', 'mianchie', 'geshuchie', 'mixinkuyee', 'biancie', 'shuangchilvcie', 'tutaie', 'ditaie']

beyond_insect = {
        'yumiming', 'taozhuming', 'zhongjinhuyee', 'zhonghuaxiaobianxijingui', 'niaozuihuyee', 'dongfanglougu', 'dadilaohu', 'baitiaoyee',
        'anheisaijingui', 'yinwenyee', 'mianlingchong', 'daming', 'xianweiyee', 'huangzuliechun', 'huangdilaohu', 'huajinglvwenhuang', 'douyeming',
        'zitiaochie', 'meiguijinyee', 'bowen', 'laoshizhanchong', 'xiaoguantouxishuai', 'tubeibanhongchun', 'huangheyilijingui', 'xuanqiyee',
        'tonglvyilijingui', 'daozongjuanyeming', 'xiaoshie', 'dongfangzhanchong', 'xiaodilaohu_2', 'xiaodilaohu_1', 'bazidilaohu_2', 'bazidilaohu_1', 'zachong'
    }

xuexiao_list = [
        "caoditanyee", "fayee", "erdianweiyee", "erhuaming", "daming",
        "mianlingchong", "kuanjingyee", "xiewenyee", "xiumuyee",
        "guajuanyeming", "pingshaoyingyee", "huangchizhuiye"
    ]


_predictor_daofeishi = None
_predictor_cls12 = None
_model_big_clip = None
_model_second = None
_model_three = None
_model_three_bazidilaohu_xiaodilaohu = None

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
            # 阈值变更 0.4 > 0.65
            conf_thresh=0.65,
            device=device,
        )
    return _predictor_daofeishi

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

def predict_daofeishi(image, device=None):
    results = []
    predictor = _get_predictor_daofeishi(device=device)
    daofeishi_res = predictor.predict(
        image, clip_size=640,
        # 变更：100 -> 120
        overlap_size=120,
        # 新增：边缘过滤像素阈值
        edge_reject_distance=5,
        output=None,
    )
    for r in daofeishi_res:
        nr = _normalize_result(r, "daofeishi")
        if nr["name"] == "other":
            continue  # 过滤 daofeishi 的 other 类型
        results.append(nr)
    return results

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

def _get_beyond_models(device=None):
    """按需加载模型，支持指定 device"""
    global _model_big_clip, _model_second, _model_three, _model_three_bazidilaohu_xiaodilaohu
    dev = _get_device(device)
    if _model_big_clip is None:
        _model_big_clip = ModelDetector(
            model_path=os.path.join(base_path, "kuangxuan_0209.pt")
        )
    if _model_second is None:
        _model_second = ModelCls(os.path.join(base_path, "second_0213.pt"))
    if _model_three is None:
        _model_three = ModelCls(os.path.join(base_path, "three_0209.pt"))
    if _model_three_bazidilaohu_xiaodilaohu is None:
        _model_three_bazidilaohu_xiaodilaohu = ModelCls(
            os.path.join(base_path, "three_bazidilaohu_xiaodilaohu_0209.pt")
        )
    return _model_big_clip, _model_second, _model_three, _model_three_bazidilaohu_xiaodilaohu

def _get_cls_12_models():
    """按需加载模型，支持指定 device"""
    global _predictor_cls12
    if _predictor_cls12 is None:
        _predictor_cls12 = ModelCls(os.path.join(MODEL_ROOT, "cls_12.pt"))
    return _predictor_cls12

def class_insect(image, model, device=None):
    predict_ke = model.predict(image, device=device)
    name, conf = predict_ke["class_name"], predict_ke["conf"]
    return name, conf

def extract_clean_insect_contour(crop_image):
    # ===== 多策略融合分割 =====

    # 策略1: 灰度图阈值分割（捕捉深色部分）
    gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
    _, binary_gray = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)

    # 策略2: 自适应阈值（捕捉颜色变化部分）
    binary_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 11, 2)

    # 策略3: 边缘检测（捕捉轮廓边缘）
    edges = cv2.Canny(gray, 50, 150)
    edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    # 策略4: 颜色空间分割（HSV中的亮度/饱和度）
    hsv = cv2.cvtColor(crop_image, cv2.COLOR_BGR2HSV)
    # 提取饱和度较高的区域（虫体通常比背景饱和度高）
    saturation = hsv[:, :, 1]
    _, binary_saturation = cv2.threshold(saturation, 30, 255, cv2.THRESH_BINARY)

    # 融合多种分割结果
    # 加权融合：灰度阈值 + 自适应阈值 + 边缘 + 饱和度
    combined = cv2.addWeighted(binary_gray, 0.4, binary_adaptive, 0.3, 0)
    combined = cv2.addWeighted(combined, 0.7, edges_dilated, 0.3, 0)
    combined = cv2.addWeighted(combined, 0.8, binary_saturation, 0.2, 0)

    # 二值化融合结果
    _, combined_binary = cv2.threshold(combined, 127, 255, cv2.THRESH_BINARY)

    # ===== 形态学操作，连接断裂部分 =====
    kernel = np.ones((5, 5), np.uint8)

    # 先闭运算连接断裂
    closed = cv2.morphologyEx(combined_binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 再开运算去除小噪点
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    # 最后膨胀一点，确保轮廓完整
    final_binary = cv2.dilate(opened, np.ones((3, 3), np.uint8), iterations=1)

    # ===== 查找轮廓 =====
    contours, _ = cv2.findContours(final_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        # 取最大的轮廓（应该是虫体）
        main_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(main_contour)

        # 可选：如果最大轮廓面积太小，可能是漏检，尝试另一种方法
        if contour_area < 1000 and len(contours) > 1:
            # 合并相近的轮廓
            all_points = np.vstack([c.reshape(-1, 2) for c in contours])
            hull = cv2.convexHull(all_points)
            contour_area = cv2.contourArea(hull)
        return contour_area
    else:
        # 如果没找到轮廓，尝试用凸包
        all_points = np.column_stack(np.where(final_binary > 0))
        if len(all_points) > 0:
            hull = cv2.convexHull(all_points.astype(np.float32))
            contour_area = cv2.contourArea(hull)
            return contour_area
        else:
            return 0

def beyond_predict(crop_image, area, device=None):

    top2 = _model_second.predictTop2(crop_image, device=device)
    name, conf = top2["1"]["class_name"], top2["1"]["conf"]
    name2, conf2 = top2["2"]["class_name"], top2["2"]["conf"]

    first_name = name
    first_conf = conf
    second_name = name2
    second_conf = conf2

    # 二级模型识别过滤
    if name == "huangzuliechun" and conf < 0.8 and area < 200000:
        first_name = ""
    if name == "tubeibanhongchun" and conf < 0.8 and area < 120000:
        first_name = ""
    if name == "huangzuliechun" and area < 200000:
        first_name = ""
    if name == "tubeibanhongchun" and conf < 0.5:
        first_name = ""

    if name == "daming" and conf < 0.5 and area > 300000:
        first_name, first_conf = class_insect(crop_image, _model_three)

    if name == "dadilaohu" and conf < 0.7 and area < 700000:
        first_name, first_conf = class_insect(crop_image, _model_three_bazidilaohu_xiaodilaohu)

    if name == "baitiaoyee":
        if (area > 700000 and conf < 0.9) or (500000 <= area <= 700000 and first_conf < 0.75):
            first_name, first_conf = class_insect(crop_image, _model_three_bazidilaohu_xiaodilaohu)

    if name == "dongfangzhanchong" and conf < 0.5 and area > 500000:
        first_name, first_conf = class_insect(crop_image, _model_three_bazidilaohu_xiaodilaohu)

    if name == "laoshizhanchong" and conf < 0.75 and area > 600000:
        first_name, first_conf = class_insect(crop_image, _model_three_bazidilaohu_xiaodilaohu)

    if name == "xianweiyee" and conf < 0.9 and area > 400000:
        first_name, first_conf = class_insect(crop_image, _model_three_bazidilaohu_xiaodilaohu)

    if name in {"yinwenyee"} and conf < 0.5:
        first_name, first_conf = class_insect(crop_image, _model_three_bazidilaohu_xiaodilaohu)

    if "bazidilaohu" in name:
        first_name = "bazidilaohu"
    if "xiaodilaohu" in name:
        first_name = "xiaodilaohu"

    if "bazidilaohu" in second_name:
        second_name = "bazidilaohu"
    if "xiaodilaohu" in second_name:
        second_name = "xiaodilaohu"

    # 最后的过滤
    if area < 20000:
        first_name = ""

    if first_conf < 0.3:
        first_name = ""

    return first_name, first_conf, second_name, second_conf

def prdict_second(image, device=None, merage=False):
    results = []
    _get_beyond_models(device)
    model_cls_12 = _get_cls_12_models()

    results1 = _model_big_clip.predict(image, debug=False, clip_size=2500, overlap_size=800, device=device)
    print(results1)

    for d in results1:
        x1, y1, x2, y2, conf, name = d["x1"], d["y1"], d["x2"], d["y2"], d["conf"], d["class_name"]
        area = (x2 - x1) * (y2 - y1)
        crop_image = image[y1:y2, x1:x2]
        if merage:
            first_name, first_conf, second_name, second_conf = beyond_predict(crop_image, area, device=device)
            if first_name != "":
                results.append({"name": first_name, "score": first_conf, "location": [x1, y1, x2, y2], "msg": "", "source": "beyond"})
        else:
            name, conf = class_insect(crop_image, _model_second, device)
            if name in beyond_insect:
                first_name, first_conf, second_name, second_conf = beyond_predict(crop_image, area, device=device)
                if first_name != "":
                    results.append({"name": first_name, "score": first_conf, "location": [x1, y1, x2, y2], "msg": "",
                                    "source": "beyond"})
                else:
                    continue
            else:
                top2 = model_cls_12.predictTop2(crop_image, device=device)
                name, conf = top2["1"]["class_name"], top2["1"]["conf"]
                results.append({"name": name, "score": conf, "location": [x1, y1, x2, y2], "msg": "", "source": "cls-12"})
    return results

def ior(box1, box2):
    x1, y1, x2, y2 = box1[:4]
    x3, y3, x4, y4 = box2[:4]
    x_overlap = max(0, min(x2, x4) - max(x1, x3))
    y_overlap = max(0, min(y2, y4) - max(y1, y3))
    overlap_area = x_overlap * y_overlap
    box1_s = ( x2-x1 ) * (y2-y1)
    box2_s = ( x4-x3 ) * (y4-y3)
    if box1_s == 0 or box2_s == 0:
        return 0
    return overlap_area / min(box1_s, box2_s)

def _merge_big_with_small(big_insects, small_insects):

    to_remove = set()
    for i in range(len(big_insects)):
        res_i = big_insects[i]
        for j in range(len(small_insects)):
            res_j = small_insects[j]

            if ior(res_i["location"], res_j["location"]) > 0.8:
                to_remove.add(id(res_j))

    small_insects_result = [r for r in small_insects if id(r) not in to_remove]
    return big_insects + small_insects_result

def predict(image, merage=False):
    # device = _get_device()
    daofeishi_predict = predict_daofeishi(image)
    second = prdict_second(image, merage=merage)
    merge_big_with_small = _merge_big_with_small(second, daofeishi_predict)
    return merge_big_with_small
