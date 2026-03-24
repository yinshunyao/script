import os
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0].parents[0]

# 模型目录: models/20260123 或 model/20260123
base_path = os.path.join(ROOT, "models", "20260123")
if not os.path.isdir(base_path):
    base_path = os.path.join(ROOT, "model", "20260123")

# from script.predict.model_detect_bejing_fixed import ModelDetec
from script.predict.model_detect import ModelDetector
from script.predict.model_cls import ModelCls


def _get_device(device=None):
    """获取推理设备，支持 cuda/mps/cpu，None 时自动检测"""
    if device is not None:
        return device
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# 延迟初始化，支持 device 参数
_model_big_clip = None
_model_second = None
_model_three = None
_model_three_bazidilaohu_xiaodilaohu = None


def _get_models(device=None):
    """按需加载模型，支持指定 device"""
    global _model_big_clip, _model_second, _model_three, _model_three_bazidilaohu_xiaodilaohu
    dev = _get_device(device)
    if _model_big_clip is None:
        _model_big_clip = ModelDetector(
            model_path=os.path.join(base_path, "kuangxuan_0209.pt"),
            device=dev,
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


def class_insect(image, model, device=None):
    predict_ke = model.predict(image, device=device)
    name, conf = predict_ke["class_name"], predict_ke["conf"]
    return name, conf


def predict(orig_img, device=None):
    """
    大虫预测推理入口
    :param orig_img: 输入图像 (numpy BGR)
    :param device: 设备 'cuda'/'mps'/'cpu'，None 时自动检测
    :return: 结果列表 [{name, score, location, msg}, ...]
    """
    model_big_clip, model_second, model_three, model_three_bazidilaohu_xiaodilaohu = _get_models(device)
    dev = _get_device(device)

    results1 = model_big_clip.predict(orig_img, debug=False, clip_size=2500, overlap_size=800)

    results = []
    for d in results1:
        x1, y1, x2, y2, conf, name = d["x1"], d["y1"], d["x2"], d["y2"], d["conf"], d["class_name"]
        area = (x2 - x1) * (y2 - y1)
        crop_image = orig_img[y1:y2, x1:x2]

        message = ""
        name, conf = class_insect(crop_image, model_second, device=dev)

        if name == "daming" and conf < 0.5 and area > 300000:
            name, conf = class_insect(crop_image, model_three, device=dev)
            message = "大螟改成" + name

        if name == "dadilaohu" and conf < 0.7 and area < 700000:
            name, conf = class_insect(crop_image, model_three_bazidilaohu_xiaodilaohu, device=dev)
            message = "dadilaohu 改成"+name

        if name == "baitiaoyee":
            if (area > 700000 and conf < 0.9) or (500000 <= area <= 700000 and conf < 0.75):
                name, conf = class_insect(crop_image, model_three_bazidilaohu_xiaodilaohu, device=dev)
                message = "baitiaoyee 改成" + name

        if name == "dongfangzhanchong" and conf < 0.5 and area > 500000:
            name, conf = class_insect(crop_image, model_three_bazidilaohu_xiaodilaohu, device=dev)
            message = "dongfangzhanchong 改成"+name

        if name == "laoshizhanchong" and conf < 0.75 and area > 600000:
            name, conf = class_insect(crop_image, model_three_bazidilaohu_xiaodilaohu, device=dev)
            message = "laoshizhanchong 改成"+name

        if name == "ganlanyee" and conf < 0.7 and area < 700000:
            name, conf = class_insect(crop_image, model_three_bazidilaohu_xiaodilaohu, device=dev)
            message = "ganlanyee 改成"+name

        if name == "xianweiyee" and conf < 0.9 and area > 400000:
            name, conf = class_insect(crop_image, model_three_bazidilaohu_xiaodilaohu, device=dev)
            message = "xianweiyee 改成"+name

        if name in {"yinwenyee", "pingshaoyingyee"} and conf < 0.5:
            name, conf = class_insect(crop_image, model_three_bazidilaohu_xiaodilaohu, device=dev)


        if area < 20000:
            continue


        result_data = {
            'name': name,
            'score': conf,
            'location': [x1, y1, x2, y2],
            'msg': message
        }
        results.append(result_data)

    return results