#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对已检出 VOC（含 cls_topn）做 top1/top2/top3 场景分析：正确 / 近种 / 其他。

设计见 insect/doc/02-dr/【分类评估】检出结果topn近种场景分析.md
直接改 ``if __name__ == "__main__"`` 变量后运行。
"""
from __future__ import annotations

import csv
import json
import logging
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

_INSECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _INSECT_ROOT / "script" / "config"
if str(_INSECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_INSECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("analyze_cls_topn_scenes")

DUMP_KEYS = frozenset(
    {
        "black_classes",
        "other",
        "other_small",
        "other_part",
        "weizhi",
        "unknown",
        "insect",
        "background",
    }
)
COARSE_DUMP_KEYS = frozenset({"chie", "minge"})

# 长后缀优先。jingui 放在 saijingui / yilijingui 之后。
_SUFFIX_GROUP: tuple[tuple[str, str], ...] = (
    ("saijingui", "jingui"),
    ("yilijingui", "jingui"),
    ("lijingui", "jingui"),
    ("yechan", "yechan"),
    ("feishi", "feishi"),
    ("dilaohu", "yee"),
    ("nianchong", "yee"),
    ("zhanchong", "yee"),
    ("tiane", "tiane"),
    ("yeming", "ming"),
    ("juane", "juane"),
    ("denge", "denge"),
    ("lougu", "lougu"),
    ("piaochong", "piaochong"),
    ("bujia", "bujia"),
    ("hujia", "hujia"),
    ("mangchun", "chun"),
    ("yee", "yee"),
    ("ming", "ming"),
    ("chie", "chie"),
    ("chun", "chun"),
    ("jingui", "jingui"),
)

_SKIP_XML_PARENTS = frozenset({"eval_metrics"})


@dataclass(frozen=True)
class VocBox:
    name: str
    x1: int
    y1: int
    x2: int
    y2: int
    filtered: bool
    topn: tuple[tuple[str, float], ...]
    det_conf: float
    cls_conf: float


@dataclass(frozen=True)
class SceneHit:
    bucket: str
    detail: str
    gt_key: str
    top1: str
    top2: str
    top3: str
    top1_conf: float
    top2_conf: float
    top3_conf: float
    group_top1: str
    group_top2: str


def _text(el: ET.Element | None) -> str:
    if el is None or el.text is None:
        return ""
    return str(el.text).strip()


def _bbox_from_obj(obj: ET.Element) -> tuple[int, int, int, int] | None:
    box = obj.find("bndbox")
    if box is None:
        return None
    try:
        x1, y1 = int(float(_text(box.find("xmin")))), int(float(_text(box.find("ymin"))))
        x2, y2 = int(float(_text(box.find("xmax")))), int(float(_text(box.find("ymax"))))
    except ValueError:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _parse_topn(obj: ET.Element) -> tuple[tuple[str, float], ...]:
    block = obj.find("cls_topn")
    items: list[tuple[str, float]] = []
    if block is not None:
        for item in block.findall("item"):
            nm = _text(item.find("name"))
            if not nm:
                continue
            conf = 0.0
            raw = _text(item.find("conf"))
            if raw:
                try:
                    conf = float(raw)
                except ValueError:
                    conf = 0.0
            items.append((nm, conf))
    if items:
        return tuple(items)
    nm = _text(obj.find("name"))
    conf = 0.0
    raw = _text(obj.find("cls_conf"))
    if raw:
        try:
            conf = float(raw)
        except ValueError:
            conf = 0.0
    if nm:
        return ((nm, conf),)
    return ()


def _is_filtered(obj: ET.Element) -> bool:
    if _text(obj.find("filter_reason")):
        return True
    flag = _text(obj.find("filtered")).lower()
    return flag in {"1", "true", "yes"}


def parse_voc_xml(path: Path) -> list[VocBox]:
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as e:
        log.warning("XML 解析失败，跳过: %s (%s)", path, e)
        return []
    out: list[VocBox] = []
    for obj in root.findall("object"):
        bb = _bbox_from_obj(obj)
        if bb is None:
            continue
        name = _text(obj.find("name"))
        det = 0.0
        cls_c = 0.0
        d_raw, c_raw = _text(obj.find("det_conf")), _text(obj.find("cls_conf"))
        if d_raw:
            try:
                det = float(d_raw)
            except ValueError:
                det = 0.0
        if c_raw:
            try:
                cls_c = float(c_raw)
            except ValueError:
                cls_c = 0.0
        out.append(
            VocBox(
                name=name,
                x1=bb[0],
                y1=bb[1],
                x2=bb[2],
                y2=bb[3],
                filtered=_is_filtered(obj),
                topn=_parse_topn(obj),
                det_conf=det,
                cls_conf=cls_c,
            )
        )
    return out


def box_iou(a: VocBox, b: VocBox) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0, a.x2 - a.x1) * max(0, a.y2 - a.y1)
    ba = max(0, b.x2 - b.x1) * max(0, b.y2 - b.y1)
    den = aa + ba - inter
    return float(inter) / float(den) if den > 0 else 0.0


def _walk_alg_class_names(obj: Any, mapping: dict[str, str]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, dict) and (
                "cn_name" in v or "infer_name" in v or "enable" in v
            ):
                key = str(k).strip()
                if key:
                    mapping.setdefault(key, key)
                    cn = str(v.get("cn_name") or "").strip()
                    infer = str(v.get("infer_name") or "").strip()
                    if cn:
                        mapping[cn] = key
                    if infer:
                        mapping.setdefault(infer, key)
            _walk_alg_class_names(v, mapping)
    elif isinstance(obj, list):
        for x in obj:
            _walk_alg_class_names(x, mapping)


def load_name_catalog(
    *,
    insect_info_path: Path | None = None,
    alg_json_path: Path | None = None,
) -> dict[str, str]:
    """中文 / 拼音 / 异名 → 拼音 key。"""
    mapping: dict[str, str] = {
        "其他": "other",
        "未知": "unknown",
        "昆虫": "insect",
        "black_classes": "black_classes",
    }
    info_path = insect_info_path or (_CONFIG_DIR / "insect_info.json")
    if info_path.is_file():
        raw = json.loads(info_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            for pinyin, rec in raw.items():
                key = str(pinyin).strip()
                if not key:
                    continue
                mapping.setdefault(key, key)
                if isinstance(rec, dict):
                    zh = str(rec.get("name_zh") or "").strip()
                    if zh:
                        mapping.setdefault(zh, key)
    alg_path = alg_json_path or (_CONFIG_DIR / "insect_alg_all.json")
    if alg_path.is_file():
        try:
            alg = json.loads(alg_path.read_text(encoding="utf-8"))
            _walk_alg_class_names(alg, mapping)
        except json.JSONDecodeError as e:
            log.warning("insect_alg_all.json 无法解析: %s", e)
    try:
        from script.config.competition_cn_aliases import (  # noqa: PLC0415
            COMPETITION_CANONICAL_CN,
            COMPETITION_CN_ALIASES,
        )
    except ImportError:
        COMPETITION_CANONICAL_CN = {}
        COMPETITION_CN_ALIASES = {}
    for pinyin, zh in COMPETITION_CANONICAL_CN.items():
        if zh:
            mapping[str(zh)] = str(pinyin)
        mapping.setdefault(str(pinyin), str(pinyin))
    for zh, pinyin in COMPETITION_CN_ALIASES.items():
        if zh and pinyin:
            mapping[str(zh)] = str(pinyin)
    return mapping


def canon_name(raw: str, catalog: dict[str, str]) -> str:
    s = (raw or "").strip()
    if not s:
        return ""
    if s in catalog:
        return catalog[s]
    low = s.lower()
    if low in catalog:
        return catalog[low]
    return low if s.isascii() else s


def near_group(key: str) -> str:
    k = (key or "").strip().lower()
    if not k or k in DUMP_KEYS:
        return ""
    for suffix, group in _SUFFIX_GROUP:
        if k.endswith(suffix):
            return group
    return ""


def is_dump(key: str) -> bool:
    k = (key or "").strip().lower()
    if k in DUMP_KEYS:
        return True
    return k.startswith("other")


def is_coarse_dump(key: str) -> bool:
    return (key or "").strip().lower() in COARSE_DUMP_KEYS


def is_near(a: str, b: str) -> bool:
    ga, gb = near_group(a), near_group(b)
    return bool(ga and gb and ga == gb)


def _top_at(topn: tuple[tuple[str, float], ...], i: int) -> tuple[str, float]:
    if i < len(topn):
        return topn[i]
    return "", 0.0


def classify_scene(
    topn_keys: tuple[str, ...],
    topn_confs: tuple[float, ...],
    *,
    gt_key: str = "",
) -> SceneHit:
    """按 DR 表判定大类/细类。``topn_keys`` 已是拼音 key。"""
    t1 = topn_keys[0] if topn_keys else ""
    t2 = topn_keys[1] if len(topn_keys) > 1 else ""
    t3 = topn_keys[2] if len(topn_keys) > 2 else ""
    c1 = topn_confs[0] if topn_confs else 0.0
    c2 = topn_confs[1] if len(topn_confs) > 1 else 0.0
    c3 = topn_confs[2] if len(topn_confs) > 2 else 0.0
    g1, g2 = near_group(t1), near_group(t2)
    gt = (gt_key or "").strip()

    def hit(bucket: str, detail: str) -> SceneHit:
        return SceneHit(
            bucket=bucket,
            detail=detail,
            gt_key=gt,
            top1=t1,
            top2=t2,
            top3=t3,
            top1_conf=c1,
            top2_conf=c2,
            top3_conf=c3,
            group_top1=g1,
            group_top2=g2,
        )

    if len(topn_keys) < 2:
        return hit("其他", "no_topn")

    if gt:
        if t1 == gt:
            if is_near(t1, t2):
                return hit("正确", "correct_near_top2")
            if is_dump(t2):
                return hit("正确", "correct_dump_top2")
            if is_coarse_dump(t2):
                return hit("正确", "correct_coarse_top2")
            return hit("正确", "correct_far_top2")
        in_t2, in_t3 = t2 == gt, t3 == gt
        if in_t2 and is_near(t1, gt):
            return hit("近种", "err_gt_in_top2_near")
        if in_t3 and is_near(t1, gt):
            return hit("近种", "err_gt_in_top3_near")
        if in_t2:
            return hit("近种", "err_gt_in_top2")
        if in_t3:
            return hit("近种", "err_gt_in_top3")
        if is_dump(t2):
            return hit("其他", "dump_ood")
        if is_near(t1, t2):
            return hit("近种", "err_top12_near")
        if is_coarse_dump(t2):
            return hit("其他", "coarse_dump")
        return hit("其他", "far_error")

    if is_dump(t2):
        return hit("其他", "no_gt_dump")
    if is_coarse_dump(t2):
        return hit("其他", "no_gt_coarse")
    if is_near(t1, t2):
        return hit("近种", "no_gt_top12_near")
    return hit("其他", "no_gt_far")


def match_gt(
    pred: VocBox, gts: list[VocBox], used: set[int], iou_thr: float
) -> tuple[VocBox | None, float]:
    best_i = -1
    best_iou = 0.0
    for i, g in enumerate(gts):
        if i in used:
            continue
        iou = box_iou(pred, g)
        if iou > best_iou:
            best_iou = iou
            best_i = i
    if best_i < 0 or best_iou < iou_thr:
        return None, best_iou
    used.add(best_i)
    return gts[best_i], best_iou


def iter_pred_xmls(pred_dir: Path) -> Iterable[Path]:
    for p in sorted(pred_dir.rglob("*.xml")):
        if p.name.startswith("._"):
            continue
        if any(part in _SKIP_XML_PARENTS for part in p.parts):
            continue
        yield p


def gt_xml_for_pred(pred_xml: Path, gt_dir: Path | None) -> Path | None:
    if gt_dir is None or not gt_dir.is_dir():
        return None
    cand = gt_dir / pred_xml.name
    if cand.is_file():
        return cand
    hits = list(gt_dir.rglob(pred_xml.name))
    hits = [h for h in hits if not any(part in _SKIP_XML_PARENTS for part in h.parts)]
    return hits[0] if hits else None


def analyze_directory(
    pred_dir: Path,
    *,
    gt_dir: Path | None,
    out_dir: Path,
    iou_thr: float,
    include_filtered: bool,
    catalog: dict[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    xmls = list(iter_pred_xmls(pred_dir))
    log.info("扫描预测 XML: n=%d dir=%s", len(xmls), pred_dir)
    for xml_path in xmls:
        preds = parse_voc_xml(xml_path)
        gt_path = gt_xml_for_pred(xml_path, gt_dir)
        gts = parse_voc_xml(gt_path) if gt_path else []
        used: set[int] = set()
        for pred in preds:
            if pred.filtered and not include_filtered:
                continue
            keys: list[str] = []
            confs: list[float] = []
            raw_names: list[str] = []
            for nm, cf in pred.topn:
                keys.append(canon_name(nm, catalog))
                confs.append(cf)
                raw_names.append(nm)
            if not keys and pred.name:
                keys.append(canon_name(pred.name, catalog))
                confs.append(pred.cls_conf)
                raw_names.append(pred.name)
            gt_box, iou = match_gt(pred, gts, used, iou_thr) if gts else (None, 0.0)
            gt_raw = gt_box.name if gt_box else ""
            gt_key = canon_name(gt_raw, catalog) if gt_raw else ""
            scene = classify_scene(tuple(keys), tuple(confs), gt_key=gt_key)
            rows.append(
                {
                    "xml": str(xml_path),
                    "filtered": int(pred.filtered),
                    "iou": round(iou, 4),
                    "gt_raw": gt_raw,
                    "gt": scene.gt_key,
                    "top1_raw": raw_names[0] if raw_names else pred.name,
                    "top1": scene.top1,
                    "top1_conf": round(scene.top1_conf, 6),
                    "top2_raw": raw_names[1] if len(raw_names) > 1 else "",
                    "top2": scene.top2,
                    "top2_conf": round(scene.top2_conf, 6),
                    "top3_raw": raw_names[2] if len(raw_names) > 2 else "",
                    "top3": scene.top3,
                    "top3_conf": round(scene.top3_conf, 6),
                    "margin_12": round(scene.top1_conf - scene.top2_conf, 6),
                    "group_top1": scene.group_top1,
                    "group_top2": scene.group_top2,
                    "bucket": scene.bucket,
                    "detail": scene.detail,
                    "det_conf": round(pred.det_conf, 6),
                    "box": f"{pred.x1},{pred.y1},{pred.x2},{pred.y2}",
                }
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "cls_topn_scenes.csv"
    if rows:
        with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")
    log.info("写出 %s 行=%d", csv_path, len(rows))
    return rows


def print_summary(rows: list[dict[str, Any]], *, high_conf: float = 0.8) -> None:
    n = len(rows)
    print(f"\n=== 检出 topn 场景汇总  n={n} ===")
    if not n:
        print("无记录")
        return
    buckets = Counter(r["bucket"] for r in rows)
    details = Counter(r["detail"] for r in rows)
    print("大类:")
    for k in ("正确", "近种", "其他"):
        print(f"  {k:4s}  {buckets.get(k, 0):6d}  {buckets.get(k, 0) / n:.1%}")
    print("细类:")
    for k, c in details.most_common():
        print(f"  {k:24s}  {c:6d}  {c / n:.1%}")

    correct = [r for r in rows if r["bucket"] == "正确"]
    if correct:
        near_top2 = sum(1 for r in correct if r["detail"] == "correct_near_top2")
        print(
            f"正确样本中 top2 为近种: {near_top2}/{len(correct)} = {near_top2 / len(correct):.1%}"
        )

    has_gt_err = [
        r for r in rows if r.get("gt") and r["bucket"] != "正确" and r["detail"] != "no_topn"
    ]
    if has_gt_err:
        dump_n = sum(1 for r in has_gt_err if r["detail"] == "dump_ood")
        near_n = sum(1 for r in has_gt_err if r["bucket"] == "近种")
        print(
            f"有 GT 的错误: n={len(has_gt_err)}  近种={near_n} ({near_n / len(has_gt_err):.1%})"
            f"  垃圾桶OOD={dump_n} ({dump_n / len(has_gt_err):.1%})"
        )

    high = [
        r
        for r in rows
        if float(r["top1_conf"]) >= high_conf
        and r["bucket"] != "正确"
        and r["detail"] != "no_topn"
    ]
    if high:
        dump_h = sum(
            1 for r in high if r["detail"] in {"dump_ood", "no_gt_dump"}
        )
        print(
            f"top1_conf>={high_conf} 且非正确: n={len(high)}  "
            f"其中垃圾桶结构={dump_h} ({dump_h / len(high):.1%})"
        )


def run_self_check() -> None:
    """TC01–TC04，失败则抛 AssertionError。"""
    s1 = classify_scene(
        ("chirongsaijingui", "heirongsaijingui", "anheisaijingui"),
        (0.9, 0.08, 0.02),
        gt_key="chirongsaijingui",
    )
    assert s1.bucket == "正确" and s1.detail == "correct_near_top2", s1
    s2 = classify_scene(
        ("chirongsaijingui", "huangheyilijingui", "anheisaijingui"),
        (0.7, 0.2, 0.1),
        gt_key="huangheyilijingui",
    )
    assert s2.bucket == "近种" and s2.detail == "err_gt_in_top2_near", s2
    s3 = classify_scene(
        ("chirongsaijingui", "black_classes", "chie"),
        (0.96, 0.03, 0.001),
        gt_key="",
    )
    assert s3.bucket == "其他" and s3.detail == "no_gt_dump", s3
    s4 = classify_scene(
        ("chirongsaijingui", "black_classes", "chie"),
        (0.96, 0.03, 0.001),
        gt_key="yechan",
    )
    assert s4.bucket == "其他" and s4.detail == "dump_ood", s4
    assert near_group("chirongsaijingui") == "jingui"
    assert near_group("huangheyilijingui") == "jingui"
    assert is_near("chirongsaijingui", "daheisaijingui")
    assert not is_near("chirongsaijingui", "chie")
    log.info("self-check OK (TC01–TC04)")


def main(
    *,
    pred_xml_dir: str,
    gt_xml_dir: str,
    out_dir: str,
    iou_thr: float,
    include_filtered: bool,
    run_self_check_flag: bool,
) -> None:
    if run_self_check_flag:
        run_self_check()
    catalog = load_name_catalog()
    log.info("名称映射条数=%d", len(catalog))
    pred_dir = Path(pred_xml_dir).expanduser()
    if not pred_dir.is_dir():
        raise FileNotFoundError(f"PRED_XML_DIR 不存在: {pred_dir}")
    gt_path = Path(gt_xml_dir).expanduser() if gt_xml_dir else None
    if gt_path is not None and not gt_path.is_dir():
        log.warning("GT_XML_DIR 不存在，按无标注分析: %s", gt_path)
        gt_path = None
    rows = analyze_directory(
        pred_dir,
        gt_dir=gt_path,
        out_dir=Path(out_dir).expanduser(),
        iou_thr=float(iou_thr),
        include_filtered=bool(include_filtered),
        catalog=catalog,
    )
    print_summary(rows)


if __name__ == "__main__":
    RUN_SELF_CHECK = True
    # 检出 XML 目录（predict_all 输出根，递归）
    PRED_XML_DIR = (
        "/Volumes/shunyao-h1/训练数据/北京比赛/北京设备全标注早期-2.0.2-3.11.7-ori"
    )
    # 标注目录；空字符串则只看 top1–top2 结构
    GT_XML_DIR = "/Volumes/shunyao-h1/训练数据/北京比赛/北京设备全标注早期"
    OUT_DIR = str(Path(PRED_XML_DIR) / "cls_topn_scenes")
    IOU_THR = 0.25
    INCLUDE_FILTERED = True
    main(
        pred_xml_dir=PRED_XML_DIR,
        gt_xml_dir=GT_XML_DIR,
        out_dir=OUT_DIR,
        iou_thr=IOU_THR,
        include_filtered=INCLUDE_FILTERED,
        run_self_check_flag=RUN_SELF_CHECK,
    )
