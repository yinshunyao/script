#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Detail  : 只调用分类模型做 VOC(xml) 裁剪验证：
#            - 输入目录包含图片与同名 Pascal VOC xml
#            - 解析 xml 的 bndbox，从图片裁剪目标
#            - 调用分类模型（YOLO classification）
#            - 比对预测类别与标签是否一致
#            - 按类别统计，并导出混淆矩阵与「易混淆类别对」排行表（按行内混淆比例排序）
#            - 可选落盘：误分类裁剪、分类正确裁剪（均含置信度于文件名）
#            - 默认运行前清空 output_dir，避免与上次结果混合
import csv
import logging
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2

_FILE = Path(__file__).resolve()
_ROOT = _FILE.parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from script.predict.model_cls import ModelCls
from script.insect_info import c1 as DELIVERY_C1, c2 as DELIVERY_C2


def _normalize_by_map(raw: str, mapping: dict[str, str] | None) -> str:
    raw = str(raw or "").strip()
    if not raw or not mapping:
        return raw
    return str(mapping.get(raw, raw))


def normalize_class_name(
    raw: str,
    merge: dict[str, list[str]] | None,
    *,
    mapping: dict[str, str] | None = None,
) -> str:
    """
    将 raw 映射到合并组 key；若不在 merge 中则返回原始名称。
    merge 形态：{group_key: [alias1, alias2, ...]}
    """
    raw = _normalize_by_map(raw, mapping)
    raw = str(raw or "").strip()
    if not raw or not merge:
        return raw
    if raw in merge:
        return raw
    for k, aliases in merge.items():
        if raw in (aliases or []):
            return str(k)
    return raw


def is_class_match(
    pred_raw: str,
    gt_raw: str,
    merge: dict[str, list[str]] | None,
    *,
    pred_name_map: dict[str, str] | None = None,
    xml_name_map: dict[str, str] | None = None,
) -> bool:
    """
    仅用于“分类裁剪验证”的标签一致性判断：
    - 允许分别对预测名 / xml 标注名做等价映射（映射到 canonical）
    - 两边再做 normalize_class_name
    - normalize 后完全一致则算正确
    """
    pred_raw = str(pred_raw or "").strip()
    gt_raw = str(gt_raw or "").strip()
    if not pred_raw or not gt_raw:
        return False
    pred_norm = normalize_class_name(pred_raw, merge, mapping=pred_name_map)
    gt_norm = normalize_class_name(gt_raw, merge, mapping=xml_name_map)
    return pred_norm == gt_norm


def parse_pascal_voc_objects(xml_path: str) -> list[dict]:
    """读取 VOC xml，返回 object 列表：name, x1,y1,x2,y2（int）。"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    out: list[dict] = []
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


def _collect_images(input_path: str) -> tuple[Path, list[Path]]:
    pic_ext = {".jpg", ".jpeg", ".png"}
    input_p = Path(input_path)
    if input_p.is_file():
        image_files = [input_p] if input_p.suffix.lower() in pic_ext else []
    elif input_p.is_dir():
        image_files = sorted(
            p for p in input_p.rglob("*") if p.is_file() and p.suffix.lower() in pic_ext
        )
    else:
        image_files = []
    return input_p, image_files


def _safe_crop_xyxy(img_bgr, x1: int, y1: int, x2: int, y2: int):
    h, w = img_bgr.shape[:2]
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w))
    y2 = max(0, min(int(y2), h))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    crop = img_bgr[y1:y2, x1:x2]
    if crop is None or crop.size == 0:
        return None
    return crop


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _clear_run_output_dir(out_dir: str) -> None:
    """若输出目录已存在则整目录删除，避免本次结果与上次裁剪图、CSV 混合。"""
    p = Path(out_dir).expanduser().resolve()
    if not p.exists():
        return
    if not p.is_dir():
        logging.warning("输出路径已存在且不是目录，跳过清理: %s", p)
        return
    shutil.rmtree(p)
    logging.info("已清理上次输出目录: %s", p)


def _crop_export_stem(rel_path: Path, obj_index: int, pred: dict | None) -> str:
    """导出裁剪小图文件名：原图 stem + 目标序号 + 模型置信度。"""
    conf = float((pred or {}).get("conf", 0.0) or 0.0)
    return f"{rel_path.stem}__obj{obj_index:03d}__conf{conf:.3f}.jpg"


def _export_stat_by_cls_csv(out_dir: str, stat_by_cls: dict[str, dict[str, int]]) -> None:
    _ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "stat_by_class.csv")
    headers = [
        "class_norm",
        "gt",
        "pred",
        "tp",
        "report_rate",
        "fn",
        "miss_rate",
        "fp",
        "fp_rate",
        "combined_dev_rate",
    ]
    rows = []
    for cls_name, s in stat_by_cls.items():
        gt_n = int(s.get("gt", 0))
        pred_n = int(s.get("pred", 0))
        tp_n = int(s.get("tp", 0))
        fn_n = int(s.get("fn", 0))
        fp_n = int(s.get("fp", 0))
        report_rate = (float(tp_n) / float(gt_n)) if gt_n > 0 else 0.0
        miss_rate = (float(fn_n) / float(gt_n)) if gt_n > 0 else 0.0
        fp_rate = (float(fp_n) / float(pred_n)) if pred_n > 0 else 0.0
        combined_dev_rate = miss_rate + fp_rate
        rows.append(
            {
                "class_norm": str(cls_name),
                "gt": gt_n,
                "pred": pred_n,
                "tp": tp_n,
                "report_rate": round(report_rate, 6),
                "fn": fn_n,
                "miss_rate": round(miss_rate, 6),
                "fp": fp_n,
                "fp_rate": round(fp_rate, 6),
                "combined_dev_rate": round(combined_dev_rate, 6),
            }
        )
    # 排序优先级：
    # 1) 综合偏差率 combined_dev_rate（漏检率+误报率）：低 -> 高
    # 2) 漏报率 miss_rate：低 -> 高
    # 3) 误报率 fp_rate：低 -> 高
    # 再按 support(gt) 高 -> 低，最后按类别名升序，保证稳定
    rows.sort(
        key=lambda r: (
            float(r["combined_dev_rate"]),
            float(r["miss_rate"]),
            float(r["fp_rate"]),
            -int(r["gt"]),
            str(r["class_norm"]),
        )
    )
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(rows)


def _export_confusion_csv(out_dir: str, cm: dict[tuple[str, str], int]) -> None:
    _ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "confusion_matrix.csv")
    labels = sorted({a for a, _ in cm.keys()} | {b for _, b in cm.keys()})
    idx = {lab: i for i, lab in enumerate(labels)}
    mat = [[0] * len(labels) for _ in range(len(labels))]
    for (a, b), c in cm.items():
        if a not in idx or b not in idx:
            continue
        mat[idx[a]][idx[b]] += int(c)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gt\\pred"] + labels)
        for i, lab in enumerate(labels):
            w.writerow([lab] + [str(mat[i][j]) for j in range(len(labels))])


def _confusion_row_totals(cm: dict[tuple[str, str], int]) -> dict[str, int]:
    """每个 gt 类别在混淆矩阵中的行合计（该 gt 下的样本总数）。"""
    row_sum: dict[str, int] = defaultdict(int)
    for (gt, _pred), c in cm.items():
        row_sum[str(gt)] += int(c)
    return dict(row_sum)


def _export_confusion_pairs_ranked_csv(out_dir: str, cm: dict[tuple[str, str], int]) -> str | None:
    """
    从混淆矩阵提取 gt!=pred 的误分类对，按「行内混淆比例」从高到低排序后落盘。

    行内混淆比例 confusion_rate_in_gt_row = count(gt,pred) / sum_pred count(gt,*)，
    即：在真实为 gt 的样本中，被判成 pred 的比例。
    """
    _ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "confusion_pairs_ranked.csv")
    if not cm:
        return None
    row_totals = _confusion_row_totals(cm)
    total_samples = sum(row_totals.values())
    rows_out: list[dict[str, object]] = []
    for (gt, pred), c in cm.items():
        gt_s, pred_s = str(gt), str(pred)
        if gt_s == pred_s:
            continue
        cnt = int(c)
        if cnt <= 0:
            continue
        rs = int(row_totals.get(gt_s, 0))
        rate_row = (float(cnt) / float(rs)) if rs > 0 else 0.0
        rate_all = (float(cnt) / float(total_samples)) if total_samples > 0 else 0.0
        rows_out.append(
            {
                "gt": gt_s,
                "pred": pred_s,
                "count": cnt,
                "gt_row_total": rs,
                "confusion_rate_in_gt_row": round(rate_row, 6),
                "share_of_all_objects": round(rate_all, 6),
            }
        )
    rows_out.sort(
        key=lambda r: (
            -float(r["confusion_rate_in_gt_row"]),  # 行内混淆比例：高 -> 低
            -int(r["count"]),
            str(r["gt"]),
            str(r["pred"]),
        )
    )
    headers = [
        "gt",
        "pred",
        "count",
        "gt_row_total",
        "confusion_rate_in_gt_row",
        "share_of_all_objects",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(rows_out)
    return out_path


def _print_confusion_pairs_top(
    cm: dict[tuple[str, str], int],
    *,
    title: str,
    top_n: int = 20,
) -> None:
    """终端打印易混淆类别对（与 confusion_pairs_ranked.csv 同序，仅前若干行）。"""
    if not cm or top_n <= 0:
        return
    row_totals = _confusion_row_totals(cm)
    pairs: list[tuple[float, int, str, str]] = []
    for (gt, pred), c in cm.items():
        gt_s, pred_s = str(gt), str(pred)
        if gt_s == pred_s:
            continue
        cnt = int(c)
        if cnt <= 0:
            continue
        rs = int(row_totals.get(gt_s, 0))
        rate_row = (float(cnt) / float(rs)) if rs > 0 else 0.0
        pairs.append((rate_row, cnt, gt_s, pred_s))
    pairs.sort(key=lambda x: (-x[0], -x[1], x[2], x[3]))
    print(f"{title}（前 {min(top_n, len(pairs))} 条，按行内混淆比例从高到低）:")
    print(f"{'gt':<28} {'pred':<28} {'个数':>8} {'行合计':>8} {'行内比例':>10}")
    for rate_row, cnt, gt_s, pred_s in pairs[:top_n]:
        rs = int(row_totals.get(gt_s, 0))
        print(
            f"{gt_s:<28} {pred_s:<28} {cnt:>8} {rs:>8} {rate_row*100:>9.2f}%"
        )


def _disp_w(s: str) -> int:
    # 终端显示宽度：中日韩宽字符按 2 计
    import unicodedata

    w = 0
    for ch in str(s):
        if unicodedata.east_asian_width(ch) in ("W", "F"):
            w += 2
        else:
            w += 1
    return w


def _ljust_disp(s: str, width: int) -> str:
    pad = max(0, width - _disp_w(s))
    return str(s) + (" " * pad)


def _rjust_disp(s: str, width: int) -> str:
    pad = max(0, width - _disp_w(s))
    return (" " * pad) + str(s)


def _print_stat_by_cls(
    title: str,
    stat_by_cls: dict[str, dict[str, int]],
    *,
    focus: frozenset[str] | None = None,
) -> None:
    """
    只影响“按类别统计”的打印过滤，不影响统计累积/CSV/混淆矩阵等其它流程。

    :param focus: 若为 None 或空集合，则全量类别打印；否则仅打印 focus 内的类别（类别名为归一后的 class_norm）。
    """
    if not stat_by_cls:
        return
    headers = [
        "类别(归一)",
        "标签数",
        "预测数",
        "正确TP",
        "报出率",
        "漏报FN",
        "漏报率",
        "误报FP",
        "误报率",
        "综合偏差率",
    ]

    focus_set = frozenset(str(x).strip() for x in (focus or []) if str(x).strip())

    rows_with_sort: list[tuple[float, float, float, int, str, list[str]]] = []  # combined_dev, miss, fp, gt, name, row
    total = {"gt": 0, "pred": 0, "tp": 0, "fn": 0, "fp": 0}
    for cls_name in stat_by_cls.keys():
        if focus_set and str(cls_name) not in focus_set:
            continue
        s = stat_by_cls[cls_name]
        gt_n = int(s.get("gt", 0))
        pred_n = int(s.get("pred", 0))
        tp_n = int(s.get("tp", 0))
        fn_n = int(s.get("fn", 0))
        fp_n = int(s.get("fp", 0))

        report_rate = (float(tp_n) / float(gt_n)) if gt_n > 0 else 0.0
        miss_rate = (float(fn_n) / float(gt_n)) if gt_n > 0 else 0.0
        fp_rate = (float(fp_n) / float(pred_n)) if pred_n > 0 else 0.0
        combined_dev_rate = miss_rate + fp_rate

        row = (
            [
                str(cls_name),
                str(gt_n),
                str(pred_n),
                str(tp_n),
                f"{report_rate*100:.2f}%",
                str(fn_n),
                f"{miss_rate*100:.2f}%",
                str(fp_n),
                f"{fp_rate*100:.2f}%",
                f"{combined_dev_rate*100:.2f}%",
            ]
        )
        rows_with_sort.append((float(combined_dev_rate), float(miss_rate), float(fp_rate), int(gt_n), str(cls_name), row))
        total["gt"] += gt_n
        total["pred"] += pred_n
        total["tp"] += tp_n
        total["fn"] += fn_n
        total["fp"] += fp_n

    rows_with_sort.sort(
        key=lambda x: (
            x[0],  # combined_dev_rate asc（越低越好）
            x[1],  # miss_rate asc
            x[2],  # fp_rate asc
            -x[3],  # support desc
            x[4],  # class name asc
        )
    )
    rows = [r for _acc, _miss, _fp, _gt, _name, r in rows_with_sort]

    total_gt = int(total["gt"])
    total_pred = int(total["pred"])
    total_fn = int(total["fn"])
    total_fp = int(total["fp"])
    total_miss_r = (float(total_fn) / float(total_gt)) if total_gt > 0 else 0.0
    total_fp_r = (float(total_fp) / float(total_pred)) if total_pred > 0 else 0.0
    total_combined_dev = total_miss_r + total_fp_r
    all_lines = [headers] + rows + [
        [
            "合计",
            str(total_gt),
            str(total_pred),
            str(total["tp"]),
            "",
            str(total_fn),
            "",
            str(total_fp),
            "",
            f"{total_combined_dev*100:.2f}%",
        ]
    ]
    widths = [0] * len(headers)
    for line in all_lines:
        for i, cell in enumerate(line):
            widths[i] = max(widths[i], _disp_w(str(cell)))

    def _fmt_line(items: list[str]) -> str:
        out = []
        for i, it in enumerate(items):
            if i == 0:
                out.append(_ljust_disp(it, widths[i]))
            else:
                out.append(_rjust_disp(it, widths[i]))
        return " | ".join(out)

    print(f"{title}:")
    print(_fmt_line(headers))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(_fmt_line([str(x) for x in r]))
    print("-+-".join("-" * w for w in widths))
    print(_fmt_line(all_lines[-1]))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # ----------------------- 需要你改的参数 -----------------------
    # 输入：图片 + 同名 xml（Pascal VOC）
    input_path = "/Users/shunyaoyin/Documents/code/datasets/insect-data/test-data/比赛-北京"
    # input_path = "/Users/shunyaoyin/Documents/code/datasets/insect-data/test-data/dachong-标准测试集"
    # input_path = "/Users/shunyaoyin/Documents/code/datasets/insect-data/test-data/dachong-测试数据集"
    # 分类模型（Ultralytics YOLO classification）
    # cls_model_path = "/Users/shunyaoyin/Documents/code/ai-company/insect/doc/测试结果/大虫训练总结/20260424-all-large/best.pt"
    cls_model_path = "/Users/shunyaoyin/Documents/code/ai-company/insect/doc/测试结果/分类测试/v2-20260428-all-large/best.pt"
    # v4
    # cls_model_path = "/Users/shunyaoyin/Documents/code/ai-company/insect/doc/测试结果/分类测试/v4-cls/temp.pt"
    # v5
    cls_model_path = "/Users/shunyaoyin/Documents/code/ai-company/insect/doc/测试结果/分类测试/v5-cls/best.pt"

    # 输出
    output_dir = input_path + "-v5"
    # 只打印关注类别（可选）：仅影响“按类别统计”的打印，不影响统计与 CSV/混淆矩阵落盘
    # 例：FOCUS_CLASS_NAMES = ("bazidilaohu", "caodiming")
    # FOCUS_CLASS_NAMES: tuple[str, ...] | None = (
    #     # 一级
    #     "caodiming", "daozongjuanyeming", "yumiming", "caoditanyee",
    #     "erhuaming",  "laoshinianchong", "laoshinianchong", "feihuang",
    #     "daofeishi", "hefeishi"
    # )
    FOCUS_CLASS_NAMES = None


    # 类别归一（可选）：用于“标签一致性判断”和最终统计的类名归一
    # 说明：和 `script/predict_size_validate.py` 一样，key 为归一名，values 为别名列表。
    CLASS_MERGE_TO_GROUPS: dict[str, list[str]] | None = None
    # 名称等价映射（可选）：用于解决“xml 标注名”和“模型预测名”命名不一致的问题。
    # 这两个 dict 的 value 应该是同一个 canonical（标准名）。
    #
    # 例：
    # XML_NAME_TO_CANON = {"灰飞虱": "huifeishi", "白背飞虱": "baibeifeishi"}
    # PRED_NAME_TO_CANON = {"baifeifeishi": "baibeifeishi"}  # 模型输出别名 -> 标准名
    XML_NAME_TO_CANON: dict[str, str] | None = {
        "dongfangzhanchong": "dongfangnianchong",
        "laoshizhanchong": "laoshinianchong",
        "baibeifeishi": "daofeishi",
        # "daofeishi": "daofeishi",
        "hefeishi": "daofeishi",
        "dawen": "wen",
    }
    PRED_NAME_TO_CANON: dict[str, str] | None = None

    # 分类预处理：是否将裁剪补成白底正方形（训练若是白边正方形，建议 True）
    cls_pad_square: bool = True
    # 可选：灰度+CLAHE+Otsu，再扩三通道（只有你训练时用了类似流程才建议打开）
    cls_gray_binarize: bool = False
    # 是否保存误分类裁剪小图（会按 gt/pred 分目录存，便于快速排查）
    save_misclassified_crops: bool = True
    # 是否保存分类正确的裁剪小图（按归一化类别 class= 分目录，文件名含置信度）
    save_correct_crops: bool = True
    # 运行前是否清空整个 output_dir（删除上次验证产物，避免混合）
    clean_output_before_run: bool = True

    # ------------------------------------------------------------

    input_p, image_files = _collect_images(input_path)
    print(f"共找到 {len(image_files)} 张图片")
    if clean_output_before_run:
        _clear_run_output_dir(output_dir)
    _ensure_dir(output_dir)

    classifier = ModelCls(
        model_path=cls_model_path,
        device=None,
        pad_square=cls_pad_square,
        gray_binarize=cls_gray_binarize,
    )

    stat_by_cls: dict[str, dict[str, int]] = {}

    def _inc(cls_norm: str, key: str, n: int = 1) -> None:
        cls_norm = str(cls_norm or "")
        if cls_norm not in stat_by_cls:
            stat_by_cls[cls_norm] = {"gt": 0, "pred": 0, "tp": 0, "fn": 0, "fp": 0}
        stat_by_cls[cls_norm][key] = int(stat_by_cls[cls_norm].get(key, 0)) + int(n)

    cm: defaultdict[tuple[str, str], int] = defaultdict(int)  # (gt_norm, pred_norm) -> count
    sum_gt = sum_tp = sum_fn = sum_fp = 0
    img_with_xml = 0
    obj_total = 0
    obj_skipped = 0

    for idx, img_path in enumerate(image_files, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            logging.warning("[%s/%s] 无法读取图片，跳过: %s", idx, len(image_files), img_path)
            continue

        rel_path = img_path.relative_to(input_p) if input_p.is_dir() else Path(img_path.name)
        src_xml = img_path.with_suffix(".xml")
        if not src_xml.is_file():
            logging.info("[%s/%s] 无 xml，跳过: %s", idx, len(image_files), rel_path)
            continue

        try:
            gts = parse_pascal_voc_objects(str(src_xml))
        except (ET.ParseError, OSError, ValueError) as e:
            logging.warning("[%s/%s] 读取 xml 失败 %s: %s", idx, len(image_files), src_xml, e)
            continue

        img_with_xml += 1
        per_img_correct = per_img_wrong = per_img_total = 0

        for oi, g in enumerate(gts):
            gt_raw = str(g.get("name", "") or "").strip()
            if not gt_raw:
                obj_skipped += 1
                continue

            crop = _safe_crop_xyxy(img, g["x1"], g["y1"], g["x2"], g["y2"])
            if crop is None:
                obj_skipped += 1
                continue

            obj_total += 1
            per_img_total += 1

            pred = classifier.predict(crop)
            pred_raw = str((pred or {}).get("class_name", "") or "").strip() if pred else ""
            if not pred_raw:
                pred_raw = "unknown"

            gt_norm = (
                normalize_class_name(
                    gt_raw, CLASS_MERGE_TO_GROUPS, mapping=XML_NAME_TO_CANON
                )
                or gt_raw
            )
            pred_norm = (
                normalize_class_name(
                    pred_raw, CLASS_MERGE_TO_GROUPS, mapping=PRED_NAME_TO_CANON
                )
                or pred_raw
            )

            _inc(gt_norm, "gt", 1)
            _inc(pred_norm, "pred", 1)
            ok = bool(
                is_class_match(
                    pred_raw,
                    gt_raw,
                    CLASS_MERGE_TO_GROUPS,
                    pred_name_map=PRED_NAME_TO_CANON,
                    xml_name_map=XML_NAME_TO_CANON,
                )
            )
            if ok:
                _inc(gt_norm, "tp", 1)
                sum_tp += 1
                per_img_correct += 1
                if save_correct_crops:
                    stem = _crop_export_stem(rel_path, oi, pred)
                    out_ok = os.path.join(
                        output_dir,
                        "classified_crops",
                        f"class={gt_norm}",
                    )
                    _ensure_dir(out_ok)
                    cv2.imwrite(os.path.join(out_ok, stem), crop)
            else:
                # 分类模型每个 GT 都会给出一个预测：
                # - 对 GT 类别来说是 FN（漏报/没报对）
                # - 对 Pred 类别来说是 FP（误报/报成了别的类）
                _inc(gt_norm, "fn", 1)
                _inc(pred_norm, "fp", 1)
                sum_fn += 1
                sum_fp += 1
                per_img_wrong += 1

            cm[(gt_norm, pred_norm)] += 1
            sum_gt += 1

            if save_misclassified_crops and (not ok):
                stem = _crop_export_stem(rel_path, oi, pred)

                # 原有索引：先按 GT，再按 Pred
                out_sub_gt_first = os.path.join(
                    output_dir,
                    "misclassified_crops",
                    f"gt={gt_norm}",
                    f"pred={pred_norm}",
                )
                _ensure_dir(out_sub_gt_first)
                cv2.imwrite(os.path.join(out_sub_gt_first, stem), crop)

                # 新增反向索引：先按 Pred，再按 GT（便于从预测类别反查）
                out_sub_pred_first = os.path.join(
                    output_dir,
                    "misclassified_crops_by_pred",
                    f"pred={pred_norm}",
                    f"gt={gt_norm}",
                )
                _ensure_dir(out_sub_pred_first)
                cv2.imwrite(os.path.join(out_sub_pred_first, stem), crop)

        print(
            f"[{idx}/{len(image_files)}] {rel_path}  objs={per_img_total}  "
            f"correct={per_img_correct}  wrong={per_img_wrong}"
        )

    overall_report_rate = (float(sum_tp) / float(sum_gt)) if sum_gt > 0 else 0.0
    overall_miss_rate = (float(sum_fn) / float(sum_gt)) if sum_gt > 0 else 0.0
    overall_fp_rate = (float(sum_fp) / float(sum_gt)) if sum_gt > 0 else 0.0
    print("======== 分类模型 VOC 裁剪验证汇总 ========")
    print(
        f"images_with_xml={img_with_xml}  obj_total={obj_total}  obj_skipped={obj_skipped}  "
        f"标签(gt)={sum_gt}  正确TP={sum_tp}  漏报FN={sum_fn}  误报FP={sum_fp}  "
        f"报出率={overall_report_rate*100:.2f}%  漏报率={overall_miss_rate*100:.2f}%  "
        f"误报率={overall_fp_rate*100:.2f}%"
    )

    _print_stat_by_cls("按类别统计", stat_by_cls, focus=frozenset(FOCUS_CLASS_NAMES or []))

    def _normalize_delivery_focus(names: set[str]) -> frozenset[str]:
        out: set[str] = set()
        for n in names:
            raw = str(n or "").strip()
            if not raw:
                continue
            out.add(raw)
            out.add(normalize_class_name(raw, CLASS_MERGE_TO_GROUPS, mapping=XML_NAME_TO_CANON))
            out.add(normalize_class_name(raw, CLASS_MERGE_TO_GROUPS, mapping=PRED_NAME_TO_CANON))
        return frozenset(x for x in out if str(x).strip())

    delivery_c1_focus = _normalize_delivery_focus(set(DELIVERY_C1))
    delivery_c2_focus = _normalize_delivery_focus(set(DELIVERY_C2.keys()))

    _print_stat_by_cls("一类交付害虫统计", stat_by_cls, focus=delivery_c1_focus)
    _print_stat_by_cls("二类交付害虫统计", stat_by_cls, focus=delivery_c2_focus)
    _export_stat_by_cls_csv(output_dir, stat_by_cls)
    _export_confusion_csv(output_dir, dict(cm))
    pairs_path = _export_confusion_pairs_ranked_csv(output_dir, dict(cm))
    _print_confusion_pairs_top(dict(cm), title="易混淆类别对")
    print(f"输出目录: {output_dir}")
    if pairs_path:
        print(f"易混淆类别对排行表: {pairs_path}")

