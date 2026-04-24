#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Detail  : 在 predict_size_yumiming 推理流程上，与源目录 Pascal VOC 标注比对验证；
#            几何匹配用 IoR（与 model_detect.ior 一致），类别用 CLASS_MERGE_TO_GROUPS 归一后比较。
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
from predict_size_daofeishi import PredictSize
from predict_size_yumiming import _write_pascal_voc_xml


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
            m[v] = key
    return m


def normalize_class_name(raw: str, merge: dict[str, list[str]] | None) -> str:
    """配置内映射到组 key；否则原样（精确匹配语义）。"""
    if not raw:
        return ""
    alias = _build_class_alias_map(merge)
    return alias.get(raw, raw)


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


def match_pred_gt_ior(
    preds: list[dict],
    gts: list[dict],
    ior_threshold: float,
) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    """
    按 IoR 贪心匹配 pred 与 gt（与检测合并同类思路一致，阈值默认 0.8）。
    返回 (matches (pred_idx, gt_idx, ior), matched_pred_indices, matched_gt_indices)。
    """
    pairs: list[tuple[float, int, int]] = []
    for i, p in enumerate(preds):
        bp = _box_tuple(p)
        for j, g in enumerate(gts):
            bg = _box_tuple(g)
            score = ior(bp, bg)
            if score >= ior_threshold:
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

    if not val_xml_mode:
        draw_rows = (
            all_final_rows
            if predict_debug
            else [r for r in all_final_rows if not r.get("filter")]
        )
        return PredictSize._draw_results(
            image,
            draw_rows,
            draw_edge_px=draw_edge,
            clip_size=clip_size,
            debug_filter_palette=predict_debug,
        )

    color_gray = (180, 180, 180)
    color_ok = (0, 255, 0)  # 正确：绿色
    color_fp = (0, 0, 255)  # 误报：红色
    color_fn = (255, 0, 255)  # 漏报：粉色（BGR）
    color_cls_err = (0, 255, 255)  # 类型错误：黄色（BGR）
    pred_to_gt = {i: j for i, j, _ in matches}
    matched_g = {j for _i, j, _s in matches}

    img_draw = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 题注统计（仅验证模式下有意义）
    n_tp = n_fp = n_fn = n_cls_err = 0
    for i, r in enumerate(results_visible):
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
        if j not in matched_g:
            n_fn += 1

    for r in all_final_rows:
        if r.get("filter") or r.get("cls_name") == "other":
            color = color_gray
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
        cls_conf = r.get("cls_conf", 0.0)
        det_conf = r.get("conf", 0.0)

        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, int(params["rect_thk"]))

        label = f"{cls_name} det:{det_conf:.2f} cls:{cls_conf:.2f}"
        vi = _visible_index(r, results_visible)
        if vi is not None and vi in matched_p:
            j = pred_to_gt[vi]
            gt_name = gts[j]["name"]
            if not is_class_match(cls_name, gt_name, merge):
                label = f"pred={cls_name} gt={gt_name} det:{det_conf:.2f} cls:{cls_conf:.2f}"
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

        font_scale = float(params["font_scale"])
        thickness = int(params["text_thk"])
        line_specs = [(label, font_scale, thickness)]
        if edge_label:
            line_specs.append(
                (
                    edge_label,
                    float(params["edge_font_scale"]),
                    int(params["edge_text_thk"]),
                )
            )

        gap = 2
        rows = []
        baseline_y = y1 - 4
        for line, fs, thk in reversed(line_specs):
            (tw, th), bl = cv2.getTextSize(line, font, fs, thk)
            rows.append((line, fs, thk, tw, th, bl, baseline_y))
            baseline_y -= th + bl + gap

        max_tw = max(row[3] for row in rows)
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

        cv2.rectangle(img_draw, (bg_left, bg_top), (bg_right, bg_bottom), color, -1)
        for line, fs, thk, _tw, _th, _bl, by in rows:
            cv2.putText(
                img_draw,
                line,
                (x1 + 2, by),
                font,
                fs,
                (255, 255, 255),
                thk,
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
        f"类型错误(黄): {n_cls_err}",
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

    if val_xml_mode:
        assert gts is not None and matches is not None and matched_p is not None
        img_draw = draw_main_output_image(
            image,
            all_final_rows,
            clip_size,
            overlap_size,
            predict_debug,
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
            val_xml_mode=False,
            results_visible=results_visible,
            gts=[],
            matches=[],
            matched_p=set(),
            merge=CLASS_MERGE_TO_GROUPS,
        )
    save_path = os.path.join(result_output_dir, rel_path.name)
    cv2.imwrite(save_path, img_draw)
    logging.info(f"结果图片已保存: {save_path}")

    depth = 3 if image.ndim >= 3 else 1
    xml_name = Path(rel_path.name).stem + ".xml"
    xml_path = os.path.join(result_output_dir, xml_name)
    _write_pascal_voc_xml(
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
    detect_model_path = "/Users/shunyaoyin/Documents/code/models/kuangxuan_0209.pt"
    # cls_model_path = "/Users/shunyaoyin/Downloads/best.pt"
    cls_model_path = "/Users/shunyaoyin/Documents/code/ai-company/insect/doc/测试结果/大虫训练总结/20260423-all-small/epoch5.pt"
    cls_model_path = "/Users/shunyaoyin/Documents/code/ai-company/insect/doc/测试结果/大虫训练总结/20260423-all-small/best.pt"
    input_path = "/Users/shunyaoyin/Documents/code/datasets/insect-data/test-data/dachong-测试数据集"
    # input_path = "/Users/shunyaoyin/Documents/code/datasets/insect-data/test-data/dachong-honghe-temp"
    # input_path = "/Users/shunyaoyin/Documents/code/datasets/insect-data/test-data/虫情3模型测试数据/玉米螟（四川，广东）"
    input_path = "/Users/shunyaoyin/Documents/code/datasets/insect-data/test-data/比赛-北京"
    input_path = "/Users/shunyaoyin/Documents/code/datasets/insect-data/test-data/福建大赛"
    input_path = "/Users/shunyaoyin/Documents/code/datasets/insect-data/test-data/虫情4设备现场测试数据"
    input_path = "/Users/shunyaoyin/Documents/code/datasets/insect-data/test-data/线上收集0423"
    output_dir = input_path + "_042301_validate"
    clip_size = 0
    overlap_size = 600
    conf_thresh = 0.3
    edge_reject_distance = 5
    edge_reject_conf_threshold = 1
    edge_reject_cls_conf_threshold = 0.66
    cls_top1_conf_threshold = 0.3
    predict_debug = False
    debug_clip = False
    cls_pad_square = True
    size_config_path = None

    # 验证：与标注同一框的 IoR 阈值（与 model_detect.ior 定义一致）
    val_ior_threshold = 0.8
    # 最后“按类别统计”的排序开关：
    # - True(默认)：按每类正确率(=tp/(tp+cls_err))降序
    # - False：按标注类别名(归一后的类名)升序
    sort_stat_by_acc: bool = True

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
            return_full_final=True,
        )
        results = [r for r in all_rows if not r.get("filter")]

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
                matches, matched_p, matched_g = match_pred_gt_ior(
                    results, gts, val_ior_threshold
                )
                val_xml_mode = True
                sum_geom_match += len(matches)

                pred_to_gt = {i: j for i, j, _ in matches}
                # gt 计数（按归一后的组名）
                for g in gts:
                    _inc(normalize_class_name(g.get("name", ""), CLASS_MERGE_TO_GROUPS), "gt", 1)
                # pred 计数（排除 filter=True；other 也计入 pred 方便看分布）
                for r in results:
                    _inc(normalize_class_name(r.get("cls_name", ""), CLASS_MERGE_TO_GROUPS), "pred", 1)

                for i in range(len(results)):
                    if i not in matched_p:
                        fp += 1
                        _inc(normalize_class_name(results[i].get("cls_name", ""), CLASS_MERGE_TO_GROUPS), "fp", 1)
                    else:
                        j = pred_to_gt[i]
                        pred_norm = normalize_class_name(results[i].get("cls_name", ""), CLASS_MERGE_TO_GROUPS)
                        gt_norm = normalize_class_name(gts[j].get("name", ""), CLASS_MERGE_TO_GROUPS)
                        if is_class_match(
                            results[i].get("cls_name", ""),
                            gts[j].get("name", ""),
                            CLASS_MERGE_TO_GROUPS,
                        ):
                            tp += 1
                            _inc(gt_norm, "tp", 1)
                        else:
                            cls_err += 1
                            fp += 1
                            _inc(gt_norm, "cls_err", 1)
                            _inc(pred_norm, "fp", 1)

                fn = len(gts) - len(matched_g)
                for j, g in enumerate(gts):
                    if j not in matched_g:
                        _inc(normalize_class_name(g.get("name", ""), CLASS_MERGE_TO_GROUPS), "fn", 1)

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

    print(
        f"汇总(有 xml 的图片参与 tp/fp/fn): tp={sum_tp} fp={sum_fp} fn={sum_fn} "
        f"cls_err={sum_cls_err} geom_pairs={sum_geom_match}  |  "
        f"报出率={report_rate*100:.2f}% 正确率={acc_rate*100:.2f}% 错误率={err_rate*100:.2f}%"
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
            "多检FP",
            "误报率",
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

            row = [
                str(cls_name),
                gt_n,
                pred_n,
                tp_n,
                f"{report_rate*100:.2f}%",
                ce_n,
                f"{acc_rate*100:.2f}%",
                fn_n,
                fp_n,
                f"{fp_rate*100:.2f}%",
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
                str(total["fp"]),
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
                    str(total["fp"]),
                    "",
                ]
            )
        )
    print("处理完成")
