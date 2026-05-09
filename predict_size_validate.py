#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Detail  : 在 predict_size 推理流程上，与源目录 Pascal VOC 标注比对验证；
#            几何匹配默认 IoU 阈值；可选 IoR（与 model_detect.ior 一致）。类别用 CLASS_MERGE_TO_GROUPS 归一后比较。
import logging
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
import unicodedata

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover
    Image = None
    ImageDraw = None
    ImageFont = None

_FILE = Path(__file__).resolve()
_ROOT = _FILE.parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from script.predict.model_detect import ior
from script.predict_size import PredictSize, write_pascal_voc_xml


def _dedup_by_cls_iou(rows: list[dict], *, iou_threshold: float) -> list[dict]:
    """
    后处理去重：按 cls_name 分组做贪心 NMS（基于本文件 box_iou）。
    用于解决“检测阶段不同 cls 都保留，但分类阶段映射到同一 cls_name”导致的重复框。
    """
    thr = float(iou_threshold)
    if thr <= 0 or not rows:
        return rows
    out: list[dict] = []
    by_cls: dict[str, list[dict]] = {}
    for r in rows:
        by_cls.setdefault(str(r.get("cls_name", "")), []).append(r)
    for _cls, rs in by_cls.items():
        rs = sorted(rs, key=lambda x: float(x.get("conf", 0.0) or 0.0), reverse=True)
        kept: list[dict] = []
        for r in rs:
            drop = False
            for k in kept:
                iou = float(
                    box_iou(
                        [r["x1"], r["y1"], r["x2"], r["y2"]],
                        [k["x1"], k["y1"], k["x2"], k["y2"]],
                    )
                )
                if iou >= thr:
                    drop = True
                    break
            if not drop:
                kept.append(r)
        out.extend(kept)
    return out


def _build_class_alias_map(
    merge: dict[str, list[str]] | None,
) -> dict[str, str]:
    """每个出现过的类别名 -> 合并后的代表名（dict 的 key）。"""
    if not merge:
        return {}
    m: dict[str, str] = {}
    for key, vals in merge.items():
        m[key] = key
        for v in vals:
            # '*' 作为通配符时不参与别名映射
            if str(v).strip() == "*":
                continue
            m[v] = key
    return m


def _get_wildcard_group_key(merge: dict[str, list[str]] | None) -> str | None:
    """
    返回配置了通配符 '*' 的 group key（如 'insect': ['*']）。
    约定：若存在多个，按 dict 迭代顺序取第一个。
    """
    if not merge:
        return None
    for k, aliases in merge.items():
        for a in (aliases or []):
            if str(a).strip() == "*":
                return str(k)
    return None


def normalize_class_name(raw: str, merge: dict[str, list[str]] | None) -> str:
    """
    配置内映射到组 key；否则原样。

    注意：'*' 仅用于「insect 粗分类可匹配任意具体昆虫」的 **匹配规则**（见 is_class_match），
    不应作为 normalize 的兜底归一规则，否则会把所有未显式列出的类别都压扁成同一个组，影响统计。
    """
    if not raw:
        return ""
    alias = _build_class_alias_map(merge)
    hit = alias.get(raw)
    if hit is not None:
        return hit
    return raw


def is_metric_ignored_other(raw: str, merge: dict[str, list[str]] | None) -> bool:
    """
    ``other`` 类在生产环境不输出；验证脚本中该类不参与 TP/FP/FN、按类统计、混淆矩阵等指标计算。
    判定基于 ``normalize_class_name`` 后与 ``other`` 等价（忽略大小写）。
    """
    # 通配符 '*' 的语义：匹配“包括 other 在内”的任意类别。
    # 一旦启用 wildcard 组（如 'insect': ['*']），则不应再把 other 从评估/匹配集合中剔除，
    # 否则会出现“miss gt=other 永远无法匹配”的现象。
    if _get_wildcard_group_key(merge):
        return False
    norm = normalize_class_name(str(raw or ""), merge)
    return str(norm).strip().lower() == "other"


def _build_class_groups(merge: dict[str, list[str]] | None) -> dict[str, set[str]]:
    """
    将 merge 配置转为等价类集合：group_key -> {key, values...}
    """
    groups: dict[str, set[str]] = {}
    if not merge:
        return groups
    for key, vals in merge.items():
        s = set([key])
        for v in vals or []:
            if str(v).strip() == "*":
                continue
            s.add(v)
        groups[key] = s
    return groups


def _build_super_groups(groups: dict[str, set[str]]) -> dict[str, set[str]]:
    """
    兼容粗分类（例如 xml/预测里出现 dilaohu、yee），即使配置里没显式写出来，也认为属于同一大类即算正确。
    规则基于本项目历史注释约定：
    - dilaohu = bazidilaohu / dadilaohu / huangdilaohu / xiaodilaohu 及其细分
    - yee = 各种 *yee 及其细分（若 tiancaiyee/huangyeming 被归到 other，也计入 yee 兼容）
    """
    super_groups: dict[str, set[str]] = {}

    # dilaohu 大类
    dilaohu_keys = {"bazidilaohu", "dadilaohu", "huangdilaohu", "xiaodilaohu"}
    dilaohu_set: set[str] = set()
    for k in dilaohu_keys:
        if k in groups:
            dilaohu_set |= set(groups[k])
    if dilaohu_set:
        dilaohu_set |= set(dilaohu_keys)
        super_groups["dilaohu"] = dilaohu_set

    # yee 大类：所有 group key 或 value 中包含 'yee' 的都收进来
    yee_set: set[str] = set()
    for k, s in groups.items():
        if "yee" in k:
            yee_set.add(k)
            yee_set |= set(s)
        else:
            for v in s:
                if "yee" in str(v):
                    yee_set.add(k)
                    yee_set |= set(s)
                    break
    if yee_set:
        yee_set.add("yee")
        super_groups["yee"] = yee_set

    return super_groups


def is_class_match(pred_raw: str, gt_raw: str, merge: dict[str, list[str]] | None) -> bool:
    """
    类别比对放宽：标注/预测都可能是粗分类或细分类。
    若二者在配置映射关系中可归为同一组（含 super group：dilaohu/yee），则算正确；
    不在配置中的类别仍按名称精确匹配。
    """
    pred_raw = str(pred_raw or "")
    gt_raw = str(gt_raw or "")
    if not pred_raw or not gt_raw:
        return False

    wildcard_key = _get_wildcard_group_key(merge)
    if wildcard_key:
        # 若任一侧归一后为 wildcard 组，则视为“昆虫粗分类”匹配任意具体昆虫
        if normalize_class_name(pred_raw, merge) == wildcard_key or normalize_class_name(gt_raw, merge) == wildcard_key:
            return True

    groups = _build_class_groups(merge)
    super_groups = _build_super_groups(groups)

    def _belongs(name: str) -> tuple[str | None, set[str] | None]:
        # super group 优先
        if name in super_groups:
            return name, super_groups[name]
        # 普通 group
        for gk, members in groups.items():
            if name == gk or name in members:
                return gk, members
        return None, None

    pg, pm = _belongs(pred_raw)
    gg, gm = _belongs(gt_raw)

    if pm is not None and gm is not None:
        # 同一组 / 或两边均落在同一个 super-group 成员集合
        if pg == gg:
            return True
        if pred_raw in gm and gt_raw in pm:
            return True
        return False

    # 只有一边在组内：要求另一边正好是该组 key 或该组成员
    if pm is not None and gm is None:
        return gt_raw in pm
    if gm is not None and pm is None:
        return pred_raw in gm

    # 都不在配置中：精确匹配
    return pred_raw == gt_raw


def parse_pascal_voc_objects(xml_path: str) -> list[dict[str, Any]]:
    """读取 VOC xml，返回 object 列表：name, x1,y1,x2,y2。"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    out: list[dict[str, Any]] = []
    for obj in root.findall("object"):
        name_el = obj.find("name")
        if name_el is None or not name_el.text:
            continue
        name = name_el.text.strip()
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        def _int(tag: str) -> int:
            el = bnd.find(tag)
            if el is None or el.text is None:
                raise ValueError(f"missing {tag}")
            return int(float(el.text.strip()))

        out.append(
            {
                "name": name,
                "x1": _int("xmin"),
                "y1": _int("ymin"),
                "x2": _int("xmax"),
                "y2": _int("ymax"),
            }
        )
    return out


def _box_tuple(d: dict) -> list[int]:
    return [int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"])]


def _rect_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = a_area + b_area - inter
    return float(inter) / float(denom) if denom > 0 else 0.0


def box_iou(box1, box2) -> float:
    """
    两框 IoU（交并比），与常见检测评估一致。
    `box*` 为 [x1,y1,x2,y2] 或与 `ior` 相同的前四元可索引序列。
    """
    a = (int(box1[0]), int(box1[1]), int(box1[2]), int(box1[3]))
    b = (int(box2[0]), int(box2[1]), int(box2[2]), int(box2[3]))
    return _rect_iou(a, b)


def match_pred_gt(
    preds: list[dict],
    gts: list[dict],
    threshold: float,
    metric: str = "iou",
) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    """
    贪心匹配 pred 与 gt。

    :param metric: ``"iou"``（默认，交并比）或 ``"ior"``（交集/最小框面积，与 model_detect.ior 一致）。
    :return: (matches (pred_idx, gt_idx, score), matched_pred_indices, matched_gt_indices)。
    """
    m = (metric or "iou").lower().strip()
    if m == "iou":

        def _score(bp: list[int], bg: list[int]) -> float:
            return box_iou(bp, bg)

    elif m == "ior":

        def _score(bp: list[int], bg: list[int]) -> float:
            return float(ior(bp, bg))

    else:
        raise ValueError("metric must be 'iou' or 'ior', got {0!r}".format(metric))

    pairs: list[tuple[float, int, int]] = []
    for i, p in enumerate(preds):
        bp = _box_tuple(p)
        for j, g in enumerate(gts):
            bg = _box_tuple(g)
            score = _score(bp, bg)
            if score >= threshold:
                pairs.append((score, i, j))
    pairs.sort(key=lambda x: x[0], reverse=True)
    used_p: set[int] = set()
    used_g: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for score, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        matches.append((i, j, score))
    return matches, used_p, used_g


def match_pred_gt_ior(
    preds: list[dict],
    gts: list[dict],
    ior_threshold: float,
) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    """向后兼容：等价于 ``match_pred_gt(..., metric='ior')``。"""
    return match_pred_gt(preds, gts, ior_threshold, metric="ior")


def _visible_index(r: dict, results_visible: list[dict]) -> int | None:
    for idx, x in enumerate(results_visible):
        if x is r:
            return idx
    return None


def _draw_cn_text(
    img_bgr: np.ndarray,
    text: str,
    org_xy: tuple[int, int],
    *,
    font_size: int = 20,
    color_bgr: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """
    OpenCV 的 putText 不支持中文，这里优先用 PIL 绘制；PIL 不可用时退化为英文/拼音也可读的文本。
    """
    x, y = int(org_xy[0]), int(org_xy[1])
    if Image is None:
        cv2.putText(
            img_bgr,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color_bgr,
            2,
            cv2.LINE_AA,
        )
        return img_bgr

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil)

    # 尝试常见中文字体；失败则退回默认字体（可能无法完整显示中文，但不影响主流程）
    font = None
    for fp in [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]:
        try:
            if os.path.isfile(fp):
                font = ImageFont.truetype(fp, font_size)
                break
        except Exception:
            font = None
    if font is None:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

    # PIL 用 RGB，这里做一次转换
    color_rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
    draw.text((x, y), text, fill=color_rgb, font=font)
    out = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return out


def _pick_caption_anchor(
    img_w: int,
    img_h: int,
    caption_w: int,
    caption_h: int,
    boxes_xyxy: list[tuple[int, int, int, int]],
    pad: int = 8,
) -> tuple[int, int]:
    """
    尝试将题注块放在“没有框的地方”：四角候选中选择与所有框重叠最小的位置。
    """
    cand = [
        (pad, pad),
        (img_w - caption_w - pad, pad),
        (pad, img_h - caption_h - pad),
        (img_w - caption_w - pad, img_h - caption_h - pad),
    ]
    cand = [(max(pad, x), max(pad, y)) for (x, y) in cand]
    best_xy = cand[0]
    best_score = float("inf")
    for x, y in cand:
        rect = (x, y, x + caption_w, y + caption_h)
        overlap = 0.0
        for b in boxes_xyxy:
            overlap += _rect_iou(rect, b)
        if overlap < best_score:
            best_score = overlap
            best_xy = (x, y)
    return best_xy


def _auto_draw_params(img_w: int, img_h: int) -> dict[str, int | float]:
    """
    按图像分辨率自适应绘制参数，避免高分辨率下文字/线条过细看不清。

    经验基准：在 1200px 级别的图上，cv2 font_scale≈0.8、rect thickness≈2、中文题注≈20px 比较合适。
    """
    base = float(max(int(img_w), int(img_h), 1))
    # 以 1200px 为 1.0 的缩放因子，做合理截断避免过大/过小
    k = base / 1200.0
    k = max(0.75, min(2.2, k))

    rect_thk = int(round(2 * k))
    rect_thk = max(2, min(8, rect_thk))

    font_scale = 0.8 * k
    font_scale = float(max(0.6, min(2.2, font_scale)))

    text_thk = int(round(2 * k))
    text_thk = max(1, min(6, text_thk))

    edge_font_scale = float(max(0.5, min(1.8, font_scale * 0.85)))
    edge_text_thk = max(1, min(5, int(round(text_thk * 0.8))))

    miss_font_scale = float(max(0.55, min(2.0, font_scale * 0.9)))
    miss_text_thk = max(1, min(6, int(round(text_thk * 1.0))))

    cap_font_size = int(round(20 * k))
    cap_font_size = max(18, min(54, cap_font_size))

    cap_pad = int(round(10 * k))
    cap_pad = max(8, min(28, cap_pad))

    return {
        "rect_thk": rect_thk,
        "font_scale": font_scale,
        "text_thk": text_thk,
        "edge_font_scale": edge_font_scale,
        "edge_text_thk": edge_text_thk,
        "miss_font_scale": miss_font_scale,
        "miss_text_thk": miss_text_thk,
        "cap_font_size": cap_font_size,
        "cap_pad": cap_pad,
    }


def draw_main_output_image(
    image,
    all_final_rows: list[dict],
    clip_size: int,
    overlap_size: int,
    predict_debug: bool,
    *,
    label_mode: str = "detailed",
    val_xml_mode: bool,
    results_visible: list[dict],
    gts: list[dict],
    matches: list[tuple[int, int, float]],
    matched_p: set[int],
    merge: dict[str, list[str]] | None,
) -> Any:
    """
    主输出图：无源 xml 时与 PredictSize._draw_results 一致；
    有源 xml 时：正确绿框、错误红框，other 与 filter 灰框（绘制含过滤框需 return_full_final）。
    """
    h_img, w_img = image.shape[:2]
    params = _auto_draw_params(w_img, h_img)
    draw_edge = PredictSize._uses_clip_predict_path(w_img, h_img, clip_size, overlap_size)

    mode = str(label_mode or "detailed").lower().strip()
    if mode not in ("minimal", "detailed"):
        mode = "detailed"

    def _put_text_outline(img: np.ndarray, text: str, org: tuple[int, int], *, fs: float) -> None:
        if not text:
            return
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, text, org, font, fs, (0, 0, 0), thickness=3, lineType=cv2.LINE_AA)
        cv2.putText(img, text, org, font, fs, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    def _draw_compact_labels(
        img: np.ndarray,
        *,
        x1: int,
        y1: int,
        w_img: int,
        h_img: int,
        lines: list[tuple[str, float]],
    ) -> None:
        lines = [(str(t), float(fs)) for (t, fs) in lines if str(t)]
        if not lines:
            return

        font = cv2.FONT_HERSHEY_SIMPLEX
        gap = 2
        # 字体上限：避免遮挡画面（与 dual 的“中心点标签”一致的小字号风格）
        max_fs = 0.45
        sizes: list[tuple[str, float, int, int]] = []
        for t, fs in lines:
            fs2 = min(max_fs, max(0.2, float(fs)))
            (tw, th), bl = cv2.getTextSize(t, font, fs2, 1)
            sizes.append((t, fs2, th, bl))

        total_h = sum(th + bl for _t, _fs, th, bl in sizes) + gap * (len(sizes) - 1)
        # 默认放在框上方；若超出顶部，则放到框下方
        place_above = (y1 - 6 - total_h) >= 0
        if place_above:
            baseline = y1 - 6
            for t, fs, th, bl in reversed(sizes):
                _put_text_outline(img, t, (max(0, min(w_img - 1, x1 + 2)), baseline), fs=fs)
                baseline -= th + bl + gap
        else:
            baseline = y1 + 6 + sizes[0][2] + sizes[0][3]
            baseline = max(0, min(h_img - 1, baseline))
            for idx, (t, fs, th, bl) in enumerate(sizes):
                by = baseline + idx * (th + bl + gap)
                if by >= h_img:
                    break
                _put_text_outline(img, t, (max(0, min(w_img - 1, x1 + 2)), int(by)), fs=fs)

    def _draw_non_val_with_det_class(img_bgr: np.ndarray, rows: list[dict]) -> np.ndarray:
        """
        非验证模式的输出图：沿用 PredictSize._draw_results 的视觉规则，
        但为了更易区分，这里将 other 设为黄色、过滤框仍为灰色。
        但额外在标签中显示 detect 模型的 class_name，便于对比 detect vs cls。
        """
        img_draw_local = img_bgr.copy()
        color_out = (0, 0, 255)
        color_other = (0, 255, 255)  # other：黄色（BGR）
        color_drop_debug = (180, 180, 180)

        for r in rows:
            x1, y1, x2, y2 = int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"])
            cls_name = r.get("cls_name", "unknown")
            det_name = r.get("class_name", "")
            cls_conf = float(r.get("cls_conf", 0.0) or 0.0)
            det_conf = float(r.get("conf", 0.0) or 0.0)

            if predict_debug:
                color = color_drop_debug if r.get("filter") else color_out
            else:
                color = color_other if cls_name == "other" else color_out

            cv2.rectangle(img_draw_local, (x1, y1), (x2, y2), color, int(params["rect_thk"]))

            if cls_name == "other":
                label = "" if mode == "minimal" else "other"
            else:
                if mode == "minimal":
                    # 简略：类名-检测框置信度-分类置信度（各两位小数）
                    label = f"{cls_name}-{det_conf:.2f}-{cls_conf:.2f}"
                else:
                    label = f"det={det_name} cls={cls_name} det:{det_conf:.2f} cls:{cls_conf:.2f}"
            edge_label = None
            if draw_edge and clip_size:
                m = r.get("edge_min_dist")
                if m is None:
                    origin = PredictSize._parse_detect_clip_origin(r.get("detect_id", ""))
                    if origin is not None:
                        cx1, cy1 = origin
                        m = PredictSize._min_dist_to_clip_edge(
                            x1, y1, x2, y2, cx1, cy1, clip_size, w_img, h_img
                        )
                if m is not None:
                    edge_label = f"edge-{int(round(float(m)))}"

            # minimal 模式下 other 不显示任何文字；label 为空则直接跳过文字绘制
            if not label:
                continue

            font_scale = float(params["font_scale"])
            line_specs: list[tuple[str, float]] = [(label, font_scale)]
            if edge_label:
                line_specs.append((edge_label, float(params["edge_font_scale"])))
            _draw_compact_labels(
                img_draw_local,
                x1=x1,
                y1=y1,
                w_img=w_img,
                h_img=h_img,
                lines=line_specs,
            )

        return img_draw_local

    if not val_xml_mode:
        draw_rows = (
            all_final_rows
            if predict_debug
            else [r for r in all_final_rows if not r.get("filter")]
        )
        return _draw_non_val_with_det_class(image, draw_rows)

    color_gray = (180, 180, 180)  # 灰色（BGR）
    color_ok = (0, 255, 0)  # 正确：绿色
    color_fp = (0, 0, 255)  # 误报：红色
    color_fn = (255, 0, 255)  # 漏报：粉色（BGR）
    color_other = (0, 255, 255)  # other：黄色（BGR）
    color_cls_err = color_gray  # 类型错误：灰色（BGR）
    pred_to_gt = {i: j for i, j, _ in matches}
    matched_g = {j for _i, j, _s in matches}

    img_draw = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 题注统计（仅验证模式下有意义）；other 不参与题注中的指标计数（与 dual 验证脚本一致）
    n_tp = n_fp = n_fn = n_cls_err = 0
    for i, r in enumerate(results_visible):
        if is_metric_ignored_other(str(r.get("cls_name", "") or ""), merge):
            continue
        if i not in matched_p:
            n_fp += 1
        else:
            j = pred_to_gt[i]
            gt_name = gts[j]["name"]
            pred_cls = r.get("cls_name", "")
            if is_class_match(pred_cls, gt_name, merge):
                n_tp += 1
            else:
                n_cls_err += 1
                n_fp += 1
    for j, _g in enumerate(gts):
        if is_metric_ignored_other(str(_g.get("name", "") or ""), merge):
            continue
        if j not in matched_g:
            n_fn += 1

    for r in all_final_rows:
        if r.get("filter"):
            color = color_gray
        elif r.get("cls_name") == "other":
            color = color_other
        else:
            vi = _visible_index(r, results_visible)
            if vi is None:
                color = color_gray
            elif vi not in matched_p:
                color = color_fp
            else:
                j = pred_to_gt[vi]
                gt_name = gts[j]["name"]
                pred_cls = r.get("cls_name", "")
                if is_class_match(pred_cls, gt_name, merge):
                    color = color_ok
                else:
                    color = color_cls_err

        x1, y1, x2, y2 = r["x1"], r["y1"], r["x2"], r["y2"]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cls_name = r.get("cls_name", "unknown")
        det_name = r.get("class_name", "")
        cls_conf = float(r.get("cls_conf", 0.0) or 0.0)
        det_conf = float(r.get("conf", 0.0) or 0.0)

        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, int(params["rect_thk"]))

        gt_note = None
        if cls_name == "other":
            label = "" if mode == "minimal" else "other"
        else:
            if mode == "minimal":
                # 简略：类名-检测框置信度-分类置信度（各两位小数）
                label = f"{cls_name}-{det_conf:.2f}-{cls_conf:.2f}"
            else:
                label = f"det={det_name} pred={cls_name} det:{det_conf:.2f} cls:{cls_conf:.2f}"
        vi = _visible_index(r, results_visible)
        if vi is not None and vi in matched_p:
            j = pred_to_gt[vi]
            gt_name = gts[j]["name"]
            if not is_class_match(cls_name, gt_name, merge):
                if mode == "detailed":
                    # 类型预测错误：预测与标注分两行展示，第二行备注正确类别
                    gt_note = f"gt={gt_name}"
        edge_label = None
        if draw_edge and clip_size:
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

        # minimal 模式下 other 不显示任何文字；label 为空则直接跳过文字绘制
        if not label:
            continue

        font_scale = float(params["font_scale"])
        line_specs: list[tuple[str, float]] = [(label, font_scale)]
        if gt_note:
            line_specs.append((gt_note, font_scale))
        if edge_label:
            line_specs.append((edge_label, float(params["edge_font_scale"])))
        _draw_compact_labels(
            img_draw,
            x1=x1,
            y1=y1,
            w_img=w_img,
            h_img=h_img,
            lines=line_specs,
        )

    # 漏检：依据源 xml 标注画红框
    for j, g in enumerate(gts):
        if j in matched_g:
            continue
        gx1, gy1, gx2, gy2 = int(g["x1"]), int(g["y1"]), int(g["x2"]), int(g["y2"])
        cv2.rectangle(img_draw, (gx1, gy1), (gx2, gy2), color_fn, int(params["rect_thk"]))
        glabel = f"miss gt={g['name']}"
        (tw, th), _bl = cv2.getTextSize(
            glabel, font, float(params["miss_font_scale"]), int(params["miss_text_thk"])
        )
        ty = min(h_img - 2, gy2 + th + 8)
        cv2.rectangle(img_draw, (gx1, ty - th - 6), (gx1 + tw + 4, ty), color_fn, -1)
        cv2.putText(
            img_draw,
            glabel,
            (gx1 + 2, ty - 4),
            font,
            float(params["miss_font_scale"]),
            (255, 255, 255),
            int(params["miss_text_thk"]),
        )

    # 在没有框的地方增加中文题注说明（图例 + 本图统计）
    legend_lines = [
        f"正确(绿): {n_tp}",
        f"误报(红): {n_fp}",
        f"漏报(粉): {n_fn}",
        f"类型错误(灰): {n_cls_err}",
    ]
    caption = "；".join(legend_lines)

    # 估算题注块大小：按字符数粗略估计，避免引入复杂的字体测量依赖
    cap_font_size = int(params["cap_font_size"])
    cap_pad = int(params["cap_pad"])
    approx_char_w = int(cap_font_size * 0.9)
    caption_w = min(w_img - 2 * cap_pad, max(220, approx_char_w * max(10, len(caption) // 2)))
    caption_h = cap_font_size + 2 * cap_pad

    boxes_xyxy: list[tuple[int, int, int, int]] = []
    for r in all_final_rows:
        try:
            boxes_xyxy.append((int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"])))
        except Exception:
            continue
    for g in gts:
        try:
            boxes_xyxy.append((int(g["x1"]), int(g["y1"]), int(g["x2"]), int(g["y2"])))
        except Exception:
            continue

    cap_x, cap_y = _pick_caption_anchor(w_img, h_img, caption_w, caption_h, boxes_xyxy, pad=8)
    cv2.rectangle(
        img_draw,
        (cap_x, cap_y),
        (min(w_img - 1, cap_x + caption_w), min(h_img - 1, cap_y + caption_h)),
        (0, 0, 0),
        -1,
    )
    img_draw = _draw_cn_text(
        img_draw,
        caption,
        (cap_x + cap_pad, cap_y + cap_pad),
        font_size=cap_font_size,
        color_bgr=(255, 255, 255),
    )

    return img_draw


def save_prediction_image_and_xml(
    image,
    all_final_rows: list[dict],
    results_visible: list[dict],
    result_output_dir: str,
    rel_path: Path,
    clip_size: int,
    overlap_size: int,
    predict_debug: bool,
    *,
    draw_boxes_text: bool = True,
    label_mode: str = "detailed",
    val_xml_mode: bool = False,
    gts: list[dict] | None = None,
    matches: list[tuple[int, int, float]] | None = None,
    matched_p: set[int] | None = None,
) -> None:
    """
    写主结果图 + VOC xml。xml 仅含未过滤框（与 yumiming 一致）。
    有源 xml 时主图用绿/红/灰策略（见 draw_main_output_image）。
    """
    os.makedirs(result_output_dir, exist_ok=True)
    h_img, w_img = image.shape[:2]

    if draw_boxes_text:
        if val_xml_mode:
            assert gts is not None and matches is not None and matched_p is not None
            img_draw = draw_main_output_image(
                image,
                all_final_rows,
                clip_size,
                overlap_size,
                predict_debug,
                label_mode=label_mode,
                val_xml_mode=True,
                results_visible=results_visible,
                gts=gts,
                matches=matches,
                matched_p=matched_p,
                merge=CLASS_MERGE_TO_GROUPS,
            )
        else:
            img_draw = draw_main_output_image(
                image,
                all_final_rows,
                clip_size,
                overlap_size,
                predict_debug,
                label_mode=label_mode,
                val_xml_mode=False,
                results_visible=results_visible,
                gts=[],
                matches=[],
                matched_p=set(),
                merge=CLASS_MERGE_TO_GROUPS,
            )
    else:
        # 关闭可视化：直接写原图（不画框/不写文字）
        img_draw = image
    save_path = os.path.join(result_output_dir, rel_path.name)
    cv2.imwrite(save_path, img_draw)
    logging.info(f"结果图片已保存: {save_path}")

    depth = 3 if image.ndim >= 3 else 1
    xml_name = Path(rel_path.name).stem + ".xml"
    xml_path = os.path.join(result_output_dir, xml_name)
    write_pascal_voc_xml(
        xml_path,
        folder_name=os.path.basename(os.path.normpath(result_output_dir)) or "",
        image_filename=rel_path.name,
        width=w_img,
        height=h_img,
        depth=depth,
        results=results_visible,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from script.insect_info import c1 as C1_KEYS, c2 as C2_MAP
    from script.insect_info import INSECTS as INSECTS_CFG

    # 推理类别合并：比对时源 xml 的 name 可能是 key，也可能是 value 列表中的某个；
    # 不在此表中的类别按字符串精确匹配（归一后等于自身）。
    CLASS_MERGE_TO_GROUPS: dict[str, list[str]] | None = {
        # a(1) b(4) c(2)
        "anheisaijingui": ["anheisaijingui"],
        "badianhuidenge": ["badianhuidenge"],
        "baitiaoyee": ["baitiaoyee"],
        # 地老虎组
        "bazidilaohu": ["bazidilaohu-bei", "bazidilaohu-fu"],
        "caoditanyee": ["caoditanyee"],
        "chunlue": ["chunlue"],
        # d(7)
        "dadilaohu": ["dadilaohu"],
        # "dilaohu": ["bazidilaohu-bei", "bazidilaohu-fu", "dadilaohu", "huangdilaohu", "xiaodilaohu-bei", "xiaodilaohu-fu"],
        "daheisaijingui": ["daheisaijingui"],
        "daming": ["daming"],
        "daozongjuanyeming": ["daozongjuanyeming"],
        "daqingyechan": ["daqingyechan"],
        "dongfanglougu": ["dongfanglougu", ],
        "dongfangnianchong": ["dongfangnianchong", ],
        # "nianchong": ["dongfangnianchong", "laoshinianchong"],
        "douyeming": ["douyeming"],
        # e(2) f(2) g(2)
        "erdianweiyee": ["erdianweiyee"],
        "erhuaming": ["erhuaming"],
        "fayee": ["fayee"],
        "fendiedenge": ["fendiedenge"],
        "ganlanyee": ["ganlanyee"],
        "guajuanming": ["guajuanming"],
        # h(9) k(1) l（1）
        "hongjiaolvlijingui": ["hongjiaolvlijingui"],
        "huajinglvwenhuang": ["huajinglvwenhuang"],
        "huangchizhuiyeyeming": ["huangchizhuiyeyeming"],
        "huangdilaohu": ["huangdilaohu"],
        "huangheyilijingui": ["huangheyilijingui"],
        "huangtutaie": ["huangtutaie"],
        "huangyee": ["huangyee"],
        "huangyeming": ["huangyeming"],
        "huangzuliechun": ["huangzuliechun"],
        "kuanjingyee": ["kuanjingyee"],
        "laoshinianchong": ["laoshinianchong"],
        # m（3） n（1）p（1） q（2）
        "maimuyeming": ["maimuyeming"],
        "meiguijinyee": ["meiguijinyee"],
        "mianlingchong": ["mianlingchong"],
        "niaozuihuyee": ["niaozuihuyee"],
        "pingshaoyingyee": ["pingshaoyingyee"],
        "qijiaoming": ["qijiaoming"],
        "qiweiyee": ["qiweiyee"],
        # s（2）t（5）
        "shanguangmeidenge": ["shanguangmeidenge"],
        "sibanjuanyeming": ["sibanjuanyeming"],
        "taozhuming": ["taozhuming"],
        "tiancaibaidaiyeming": ["tiancaibaidaiyeming"],
        "tiancaiyee": ["tiancaiyee"],
        "tonglvyilijingui": ["tonglvyilijingui"],
        "tubeibanhongchun": ["tubeibanhongchun"],
        # x(9)
        "xianbanhongchun": ["xianbanhongchun"],
        "xianweiyee": ["xianweiyee"],
        "xiaocaie": ["xiaocaie-bei", "xiaocaie-fu"],
        "xiaodilaohu": ["xiaodilaohu-bei", "xiaodilaohu-fu"],
        "xiaoshie": ["xiaoshie"],
        "xiaoyunbansaijingui": ["xiaoyunbansaijingui"],
        "xiewenyee": ["xiewenyee"],
        "xiumuyee": ["xiumuyee"],
        "xuanqiyee": ["xuanqiyee"],
        # y(6)
        "yanqingchong": ["yanqingchong"],
        "yeesoujifeng": ["yeesoujifeng"],
        "yibeilue": ["yibeilue"],
        "yindingyee": ["yindingyee"],
        "yinwenyee": ["yinwenyee"],
        "yumiming": ["yumiming"],
        # z(4)
        "zhanmoyee": ["zhanmoyee"],
        "zhonghuaxiaobianxijingui": ["zhonghuaxiaobianxijingui"],
        "zhongjinhuyee": ["zhongjinhuyee"],
        "zijiaochie": ["zijiaochie"],
}

    PIC_EXT = {".jpg", ".jpeg", ".png"}

    cls_list = None
    # 大虫
    # detect_model_path = "/Users/shunyaoyin/Documents/code/models/kuangxuan_0209.pt"
    detect_model_path = "/Users/shunyaoyin/Documents/code/ai-company/insect/doc/测试结果/大虫框选/20260425-all/weights/best.pt"
    # detect_model_path = "/Users/shunyaoyin/Documents/code/ai-company/insect/doc/测试结果/大虫框选/20260426-large-01/best.pt"
    detect_model_path = "/Users/shunyaoyin/Documents/code/ai-company/insect/doc/测试结果/大虫框选/20260426-large-02/temp.pt"
    cls_model_path = "/Users/shunyaoyin/Documents/code/ai-company/insect/doc/测试结果/大虫训练总结/20260424-all-large/best.pt"
    cls_model_path = None
    # 稻飞虱
    # detect_model_path = "/Users/shunyaoyin/Documents/code/models/daofeishi-detect-0405.pt"
    # cls_model_path = "/Users/shunyaoyin/Documents/code/models/daofeishi-cls.pt"
    # input_path = '/Users/shunyaoyin/Documents/code/datasets/insect-data/test-data/daofeishi-chatgpt'

    # cls_model_path = "/Users/shunyaoyin/Documents/code/ai-company/insect/doc/测试结果/大虫训练总结/20260423-all-small/epoch5.pt"
    # cls_model_path = "/Users/shunyaoyin/Documents/code/ai-company/insect/doc/测试结果/大虫训练总结/20260423-all-small/best.pt"
    input_path = "/Users/shunyaoyin/Documents/code/datasets/insect-data/test-data/dachong-测试数据集"
    # input_path = "/Users/shunyaoyin/Documents/code/datasets/insect-data/test-data/dachong-honghe-temp"
    # input_path = "/Users/shunyaoyin/Documents/code/datasets/insect-data/test-data/虫情3模型测试数据/玉米螟（四川，广东）"
    # input_path = "/Users/shunyaoyin/Documents/code/datasets/insect-data/test-data/比赛-北京"
    input_path = "/Users/shunyaoyin/Documents/code/datasets/insect-data/test-data/福建大赛"
    # input_path = "/Users/shunyaoyin/Documents/code/datasets/insect-data/test-data/虫情4设备现场测试数据"
    # input_path = "/Users/shunyaoyin/Documents/code/datasets/insect-data/test-data/线上收集0423"
    # input_path = "/Users/shunyaoyin/Documents/code/datasets/insect-data/test-data/duankou_1"
    output_dir = input_path + "_0426_validate_02"
    clip_size = 0
    overlap_size = 600
    conf_thresh = 0.2
    detect_nms_iou = 0.5
    detect_max_det = 1000
    # Ultralytics class-agnostic NMS：跨类别抑制重复框（不同 cls_id 也会互相抑制）
    detect_nms_agnostic: bool | None = None
    # conf_thresh = 0.55
    edge_reject_distance = 5
    edge_reject_conf_threshold = 1
    edge_reject_cls_conf_threshold = 0.3
    cls_top1_conf_threshold = 0.3
    predict_debug = False
    # 可视化输出开关：False 时保存原图（不画框/不写文字），xml 仍照常输出
    draw_boxes_text = True
    # 可视化标签模式：
    # - "minimal"：简略展示「最终分类名-检测置信度-分类置信度」（各两位小数，如 daming-0.50-0.69）
    # - "detailed"(默认)：详细排错信息（det/cls/conf，验证模式下错误还会带 gt）
    label_mode = "detailed"
    debug_clip = False
    cls_pad_square = True
    # 新增：检测 padding 开关（整图检测时也补成正方形再检测）
    # - False(默认)：保持历史行为：只有切片边缘窗才会 padding；整图检测不额外 padding
    # - True：即使 clip_size >= 图像边长（整图检测），也先将整图白底补成正方形再检测，并将框映射回原图坐标
    detect_pad_square_full_image = True
    size_config_path = None

    # 验证：pred 与 gt 同一框的几何匹配。默认 IoU；可选 "ior"（与 model_detect.ior 一致）
    val_box_match_metric = "iou"  # "iou" | "ior"
    val_iou_threshold = 0.5
    val_ior_threshold = 0.8
    val_geom_threshold = (
        val_iou_threshold if val_box_match_metric.lower().strip() == "iou" else val_ior_threshold
    )
    # 最后“按类别统计”的排序开关：
    # - True(默认)：按每类正确率(=tp/(tp+cls_err))降序
    # - False：按标注类别名(归一后的类名)升序
    sort_stat_by_acc: bool = True
    # 分类后去重：按 cls_name 做一次 IoU 去重；None 表示不去重（保持历史行为）
    post_dedup_iou_threshold: float | None = None

    predictor = PredictSize(
        detect_model_path=detect_model_path,
        size_config_path=size_config_path,
        cls_list=cls_list,
        cls_model_path=cls_model_path,
        offset_rate=1.2,
        conf_thresh=conf_thresh,
        device=None,
        augment=False,
    )

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

    sum_tp = sum_fp = sum_fn = sum_cls_err = 0
    sum_geom_match = 0
    stat_by_cls: dict[str, dict[str, int]] = {}

    def _inc(cls_name: str, key: str, n: int = 1) -> None:
        cls_name = str(cls_name or "")
        if cls_name not in stat_by_cls:
            stat_by_cls[cls_name] = {"tp": 0, "fp": 0, "fn": 0, "cls_err": 0, "gt": 0, "pred": 0}
        stat_by_cls[cls_name][key] = int(stat_by_cls[cls_name].get(key, 0)) + int(n)

    for idx, img_path in enumerate(image_files, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[{idx}/{len(image_files)}] 无法读取图片，跳过: {img_path}")
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
        result_output_dir = save_sub_dir
        src_xml = img_path.with_suffix(".xml")

        all_rows = predictor.predict(
            img,
            clip_size=clip_size,
            overlap_size=overlap_size,
            edge_reject_distance=edge_reject_distance,
            edge_reject_conf_threshold=edge_reject_conf_threshold,
            edge_reject_cls_conf_threshold=edge_reject_cls_conf_threshold,
            cls_top1_conf_threshold=cls_top1_conf_threshold,
            output=None,
            image_name=rel_path.name,
            debug=predict_debug,
            debug_clip=debug_clip,
            cls_pad_square=cls_pad_square,
            detect_pad_square_full_image=detect_pad_square_full_image,
            detect_nms_iou=detect_nms_iou,
            detect_max_det=detect_max_det,
            detect_nms_agnostic=detect_nms_agnostic,
            return_full_final=True,
        )
        results = [r for r in all_rows if not r.get("filter")]
        if post_dedup_iou_threshold is not None:
            results = _dedup_by_cls_iou(results, iou_threshold=float(post_dedup_iou_threshold))

        tp = fp = fn = cls_err = 0
        gts: list[dict] | None = None
        matches: list[tuple[int, int, float]] | None = None
        matched_p: set[int] | None = None
        matched_g: set[int] = set()
        val_xml_mode = False

        if src_xml.is_file():
            try:
                gts = parse_pascal_voc_objects(str(src_xml))
            except (ET.ParseError, OSError, ValueError) as e:
                logging.warning("读取源 xml 失败 %s: %s", src_xml, e)
                gts = None
            if gts is not None:
                merge = CLASS_MERGE_TO_GROUPS
                p_eval_idx = [
                    i
                    for i in range(len(results))
                    if not is_metric_ignored_other(str(results[i].get("cls_name", "") or ""), merge)
                ]
                g_eval_idx = [
                    j
                    for j in range(len(gts))
                    if not is_metric_ignored_other(str(gts[j].get("name", "") or ""), merge)
                ]
                results_ev = [results[i] for i in p_eval_idx]
                gts_ev = [gts[j] for j in g_eval_idx]

                if not gts_ev:
                    matches, matched_p, matched_g = [], set(), set()
                    val_xml_mode = True
                    pred_to_gt = {}
                    for i in range(len(results)):
                        if is_metric_ignored_other(str(results[i].get("cls_name", "") or ""), merge):
                            continue
                        fp += 1
                        pn = normalize_class_name(results[i].get("cls_name", ""), merge)
                        _inc(pn, "pred", 1)
                        _inc(pn, "fp", 1)
                    fn = 0
                else:
                    matches_ev, matched_p_ev, matched_g_ev = match_pred_gt(
                        results_ev,
                        gts_ev,
                        val_geom_threshold,
                        metric=val_box_match_metric,
                    )
                    matches = [(p_eval_idx[i], g_eval_idx[j], sc) for i, j, sc in matches_ev]
                    matched_p = {p_eval_idx[i] for i in matched_p_ev}
                    matched_g = {g_eval_idx[j] for j in matched_g_ev}
                    val_xml_mode = True
                    sum_geom_match += len(matches)

                    pred_to_gt = {i: j for i, j, _ in matches}
                    for j in g_eval_idx:
                        _inc(normalize_class_name(gts[j].get("name", ""), merge), "gt", 1)
                    for i in p_eval_idx:
                        _inc(normalize_class_name(results[i].get("cls_name", ""), merge), "pred", 1)

                    for i in range(len(results)):
                        if is_metric_ignored_other(str(results[i].get("cls_name", "") or ""), merge):
                            continue
                        if i not in matched_p:
                            fp += 1
                            _inc(
                                normalize_class_name(results[i].get("cls_name", ""), merge),
                                "fp",
                                1,
                            )
                        else:
                            j = pred_to_gt[i]
                            pred_norm = normalize_class_name(results[i].get("cls_name", ""), merge)
                            gt_norm = normalize_class_name(gts[j].get("name", ""), merge)
                            if is_class_match(
                                results[i].get("cls_name", ""),
                                gts[j].get("name", ""),
                                merge,
                            ):
                                tp += 1
                                _inc(gt_norm, "tp", 1)
                            else:
                                cls_err += 1
                                fp += 1
                                _inc(gt_norm, "cls_err", 1)
                                _inc(pred_norm, "fp", 1)

                    fn = len(gts_ev) - len(matched_g_ev)
                    for j in g_eval_idx:
                        if j not in matched_g:
                            _inc(normalize_class_name(gts[j].get("name", ""), merge), "fn", 1)

        else:
            logging.info(f"无源标注 xml，跳过比对: {src_xml}")

        save_prediction_image_and_xml(
            img,
            all_rows,
            results,
            result_output_dir,
            rel_path,
            clip_size,
            overlap_size,
            predict_debug,
            draw_boxes_text=draw_boxes_text,
            label_mode=label_mode,
            val_xml_mode=val_xml_mode,
            gts=gts,
            matches=matches,
            matched_p=matched_p,
        )

        sum_tp += tp
        sum_fp += fp
        sum_fn += fn
        sum_cls_err += cls_err

        print(
            f"[{idx}/{len(image_files)}] {rel_path}  pred={len(results)}  "
            f"tp={tp} fp={fp} fn={fn} cls_err={cls_err}  (xml={src_xml.is_file()})"
        )
        for r in results:
            print(
                f"    [{r['cls_name']}] conf={r.get('cls_conf', 0):.2f}  "
                f"det_conf={r['conf']:.2f}  "
                f"box=({r['x1']},{r['y1']},{r['x2']},{r['y2']})"
            )

    predictor.release()
    # 汇总百分比指标（仅对有 xml 的图片统计）
    # - 报出率：标注框中被“报出/匹配到”的比例（不论类别对错）= (tp + cls_err) / (tp + cls_err + fn)
    # - 正确率：预测框中类别正确的比例（精确率）= tp / (tp + fp)
    # - 错误率：预测框中不正确的比例 = fp / (tp + fp) = 1 - 正确率
    denom_gt = float(sum_tp + sum_cls_err + sum_fn)
    denom_pred = float(sum_tp + sum_fp)
    report_rate = (float(sum_tp + sum_cls_err) / denom_gt) if denom_gt > 0 else 0.0
    acc_rate = (float(sum_tp) / denom_pred) if denom_pred > 0 else 0.0
    err_rate = (float(sum_fp) / denom_pred) if denom_pred > 0 else 0.0
    miss_rate = (float(sum_fn) / denom_gt) if denom_gt > 0 else 0.0
    total_dev_rate = miss_rate + err_rate

    print(
        f"汇总(有 xml 的图片参与 tp/fp/fn): tp={sum_tp} fp={sum_fp} fn={sum_fn} "
        f"cls_err={sum_cls_err} geom_pairs={sum_geom_match}  |  "
        f"报出率={report_rate*100:.2f}% 正确率={acc_rate*100:.2f}% 错误率={err_rate*100:.2f}%  "
        f"漏检率={miss_rate*100:.2f}% 总偏差率={total_dev_rate*100:.2f}%"
    )
    if stat_by_cls:
        # 中文表格打印（等宽对齐，提升可读性）
        headers = [
            "类别(归一)",
            "标注数",
            "预测数",
            "正确TP",
            "报出率",
            "类型错",
            "正确率",
            "漏检FN",
            "漏检率",
            "多检FP",
            "误报率",
            "总偏差率",
        ]
        rows_with_sort: list[tuple[str, float, list[Any]]] = []
        total = {"gt": 0, "pred": 0, "tp": 0, "cls_err": 0, "fn": 0, "fp": 0}
        for cls_name in stat_by_cls.keys():
            s = stat_by_cls[cls_name]
            gt_n = int(s.get("gt", 0))
            pred_n = int(s.get("pred", 0))
            tp_n = int(s.get("tp", 0))
            ce_n = int(s.get("cls_err", 0))
            fn_n = int(s.get("fn", 0))
            fp_n = int(s.get("fp", 0))
            # 每类百分比
            # - 报出率：该类标注中“正确报出”的比例（召回率）= tp / gt
            # - 正确率：该类被匹配到的样本里，类别正确的比例 = tp / (tp + cls_err)
            # - 误报率：预测为该类的框中，最终判为误报的比例 = fp / pred
            denom_gt = float(gt_n)
            denom_matched = float(tp_n + ce_n)
            denom_pred = float(pred_n)
            report_rate = (float(tp_n) / denom_gt) if denom_gt > 0 else 0.0
            acc_rate = (float(tp_n) / denom_matched) if denom_matched > 0 else 0.0
            fp_rate = (float(fp_n) / denom_pred) if denom_pred > 0 else 0.0
            miss_rate = (float(fn_n) / denom_gt) if denom_gt > 0 else 0.0
            total_dev_rate = miss_rate + fp_rate

            row = [
                str(cls_name),
                gt_n,
                pred_n,
                tp_n,
                f"{report_rate*100:.2f}%",
                ce_n,
                f"{acc_rate*100:.2f}%",
                fn_n,
                f"{miss_rate*100:.2f}%",
                fp_n,
                f"{fp_rate*100:.2f}%",
                f"{total_dev_rate*100:.2f}%",
            ]
            rows_with_sort.append((str(cls_name), float(acc_rate), row))
            total["gt"] += gt_n
            total["pred"] += pred_n
            total["tp"] += tp_n
            total["cls_err"] += ce_n
            total["fn"] += fn_n
            total["fp"] += fp_n

        # 排序：默认按正确率降序；关闭后按标注类别名升序
        if sort_stat_by_acc:
            rows_with_sort.sort(
                key=lambda x: (-x[1], -int(stat_by_cls.get(x[0], {}).get("gt", 0)), x[0])
            )
        else:
            rows_with_sort.sort(key=lambda x: x[0])
        rows = [r for _cls, _acc, r in rows_with_sort]

        def _disp_w(s: str) -> int:
            # 终端显示宽度：中日韩宽字符按 2 计
            w = 0
            for ch in s:
                if unicodedata.east_asian_width(ch) in ("W", "F"):
                    w += 2
                else:
                    w += 1
            return w

        def _ljust_disp(s: str, width: int) -> str:
            pad = max(0, width - _disp_w(s))
            return s + (" " * pad)

        def _rjust_disp(s: str, width: int) -> str:
            pad = max(0, width - _disp_w(s))
            return (" " * pad) + s

        # 计算列宽（按显示宽度）
        all_lines = [headers] + [[str(x) for x in r] for r in rows] + [
            [
                "合计",
                str(total["gt"]),
                str(total["pred"]),
                str(total["tp"]),
                "",
                str(total["cls_err"]),
                "",
                str(total["fn"]),
                "",
                str(total["fp"]),
                "",
                "",
            ]
        ]
        widths = [0] * len(headers)
        for line in all_lines:
            for i, cell in enumerate(line):
                widths[i] = max(widths[i], _disp_w(str(cell)))

        def _fmt_line(items: list[str]) -> str:
            out = []
            for i, it in enumerate(items):
                it = str(it)
                if i == 0:
                    out.append(_ljust_disp(it, widths[i]))
                else:
                    out.append(_rjust_disp(it, widths[i]))
            return " | ".join(out)

        sort_hint = "正确率降序" if sort_stat_by_acc else "标注类别名升序"
        print(f"按类别统计（归一后，按{sort_hint}）:")
        print(_fmt_line(headers))
        print("-+-".join("-" * w for w in widths))
        for r in rows:
            print(_fmt_line([str(x) for x in r]))
        print("-+-".join("-" * w for w in widths))
        print(
            _fmt_line(
                [
                    "合计",
                    str(total["gt"]),
                    str(total["pred"]),
                    str(total["tp"]),
                    "",
                    str(total["cls_err"]),
                    "",
                    str(total["fn"]),
                    "",
                    str(total["fp"]),
                    "",
                    "",
                ]
            )
        )

        # -------- 一类/二类/其他 汇总统计（基于 script/insect_info.py 的 c1/c2）--------
        c1_set = set(str(x) for x in (C1_KEYS or []))
        c2_set = set(str(x) for x in (C2_MAP or {}).keys())

        def _bucket(cls_norm: str) -> str:
            if cls_norm in c1_set:
                return "一类害虫"
            if cls_norm in c2_set:
                return "二类害虫"
            return "其他虫子"

        group_total: dict[str, dict[str, int]] = {
            "一类害虫": {"gt": 0, "pred": 0, "tp": 0, "cls_err": 0, "fn": 0, "fp": 0},
            "二类害虫": {"gt": 0, "pred": 0, "tp": 0, "cls_err": 0, "fn": 0, "fp": 0},
            "其他虫子": {"gt": 0, "pred": 0, "tp": 0, "cls_err": 0, "fn": 0, "fp": 0},
        }
        for cls_norm, s in stat_by_cls.items():
            b = _bucket(str(cls_norm))
            for k in ("gt", "pred", "tp", "cls_err", "fn", "fp"):
                group_total[b][k] += int(s.get(k, 0))

        def _fmt_group_line(name: str, s: dict[str, int]) -> str:
            gt_n = int(s.get("gt", 0))
            pred_n = int(s.get("pred", 0))
            tp_n = int(s.get("tp", 0))
            ce_n = int(s.get("cls_err", 0))
            fn_n = int(s.get("fn", 0))
            fp_n = int(s.get("fp", 0))
            denom_gt = float(tp_n + ce_n + fn_n)  # 与整体汇总口径一致：只对有 xml 的框计
            denom_pred = float(tp_n + fp_n)
            miss_rate = (float(fn_n) / denom_gt) if denom_gt > 0 else 0.0
            fp_rate = (float(fp_n) / denom_pred) if denom_pred > 0 else 0.0
            total_dev = miss_rate + fp_rate
            return (
                f"{name}: gt={gt_n} pred={pred_n} tp={tp_n} fp={fp_n} fn={fn_n} cls_err={ce_n} | "
                f"漏检率={miss_rate*100:.2f}% 误报率={fp_rate*100:.2f}% 总偏差率={total_dev*100:.2f}%"
            )

        print("分组统计（按 insect_info 的一类/二类/其他）：")
        print(_fmt_group_line("一类害虫", group_total["一类害虫"]))
        print(_fmt_group_line("二类害虫", group_total["二类害虫"]))
        print(_fmt_group_line("其他虫子", group_total["其他虫子"]))

        # -------- 湖南统计方法：按“识别数量 vs 鉴定数量”计算每种准确率，并加权汇总 --------
        # 说明：
        # - 识别数量：该类预测框数 pred
        # - 鉴定数量：该类标注框数 gt（仅有 xml 才会计入）
        # - 每种准确率(%) = (1 - abs(pred-gt)/gt) * 100
        #   - gt==0 且 pred>=1 => 0%
        #   - 计算结果为负 => 0%
        #   - gt==0 且 pred==0 => 0%（当晚诱集未出现该虫，避免“空类=100%”拉高均值）
        def _hn_acc(pred_n: int, gt_n: int) -> float:
            pred_n = int(pred_n)
            gt_n = int(gt_n)
            if gt_n <= 0:
                return 0.0 if pred_n >= 0 else 0.0
            acc = (1.0 - abs(float(pred_n - gt_n)) / float(gt_n)) * 100.0
            if acc < 0:
                return 0.0
            return float(acc)

        # 湖南省二类：从 insect_info 的 level=2 中筛出“华中”或备注里显式包含“湖南”的条目
        # 注：现有表格里有少量条目只在备注写“湖南”，未在 zones 标为华中，这里兼容两者。
        hn_c2_set: set[str] = set()
        for k, cfg in (INSECTS_CFG or {}).items():
            try:
                if int(getattr(cfg, "level", 0)) != 2:
                    continue
                zones = set(getattr(cfg, "zones", []) or [])
                notes = str(getattr(cfg, "notes", "") or "")
                if ("华中" in zones) or ("湖南" in notes):
                    hn_c2_set.add(str(k))
            except Exception:
                continue

        def _is_c1(k: str) -> bool:
            return k in c1_set

        def _is_hn_c2(k: str) -> bool:
            return k in hn_c2_set

        # 仅统计“当晚诱集”出现的虫（gt>0 或 pred>0），并输出每种准确率
        hn_rows: list[tuple[str, str, int, int, float]] = []
        for cls_norm, s in stat_by_cls.items():
            cls_norm = str(cls_norm)
            gt_n = int(s.get("gt", 0))
            pred_n = int(s.get("pred", 0))
            if gt_n <= 0 and pred_n <= 0:
                continue
            bucket = "一类害虫" if _is_c1(cls_norm) else ("湖南二类害虫" if _is_hn_c2(cls_norm) else "其他")
            acc = _hn_acc(pred_n, gt_n)
            hn_rows.append((bucket, cls_norm, gt_n, pred_n, acc))

        # 打印：按一类/湖南二类优先，其它置后
        bucket_rank = {"一类害虫": 0, "湖南二类害虫": 1, "其他": 9}
        hn_rows.sort(key=lambda x: (bucket_rank.get(x[0], 9), x[0], x[1]))

        print("湖南统计方法（每种虫体的自动识别与计数准确率）：")
        print("分组 | 类别(归一) | 鉴定数量(gt) | 识别数量(pred) | 每种准确率")
        for b, cls_norm, gt_n, pred_n, acc in hn_rows:
            print(f"{b} | {cls_norm} | {gt_n} | {pred_n} | {acc:.2f}%")

        def _avg_acc(rows: list[tuple[str, str, int, int, float]], bucket: str) -> float:
            vals = [r[4] for r in rows if r[0] == bucket]
            if not vals:
                return 0.0
            return float(sum(vals) / float(len(vals)))

        avg_c1 = _avg_acc(hn_rows, "一类害虫")
        avg_hn_c2 = _avg_acc(hn_rows, "湖南二类害虫")
        day_acc = 0.6 * avg_c1 + 0.4 * avg_hn_c2
        day_score_30 = day_acc * 0.30  # 30 分制

        print(
            "湖南统计方法汇总："
            f"一类均值={avg_c1:.2f}%  湖南二类均值={avg_hn_c2:.2f}%  "
            f"加权当天准确率={day_acc:.2f}%  折算(30分制)={day_score_30:.2f}/30"
        )
    print("处理完成")
