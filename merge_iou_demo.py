#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/05/06
# @Author  : ysy
# @Email   : xxx@qq.com 
# @Detail  : 
# @Software: PyCharm

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

from script.merge_iou_sources import merge_two_sources_by_iou


DetectItem = Dict[str, Any]


@dataclass(frozen=True)
class VocObjects:
    tree: ET.ElementTree
    objects: List[Tuple[str, int, int, int, int]]  # (name, x1, y1, x2, y2)


def _safe_int(s: Optional[str], default: int = 0) -> int:
    if s is None:
        return default
    try:
        return int(float(str(s).strip()))
    except Exception:
        return default


def parse_voc_xml(xml_path: Path) -> VocObjects:
    """
    读取 VOC xml，返回 (tree, objects)。
    objects 元素： (name, x1, y1, x2, y2)。
    """
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    out: List[Tuple[str, int, int, int, int]] = []
    for obj in root.findall("object"):
        name_el = obj.find("name")
        if name_el is None or not name_el.text:
            continue
        name = name_el.text.strip()
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        x1 = _safe_int(getattr(bnd.find("xmin"), "text", None))
        y1 = _safe_int(getattr(bnd.find("ymin"), "text", None))
        x2 = _safe_int(getattr(bnd.find("xmax"), "text", None))
        y2 = _safe_int(getattr(bnd.find("ymax"), "text", None))
        out.append((name, x1, y1, x2, y2))
    return VocObjects(tree=tree, objects=out)


def _collect_xmls_by_stem(root_dir: Path) -> Tuple[Dict[str, Path], int]:
    """
    收集目录下所有 xml（递归），以 **xml 文件名 stem** 作为 key，用于跨目录配对 merge。

    若 stem 重名（冲突），会选择一条路径并返回冲突计数（用于打印告警）。
    """
    root_dir = root_dir.expanduser().resolve()
    m: DefaultDict[str, List[Path]] = DefaultDict(list)
    for p in root_dir.rglob("*.xml"):
        if p.is_file():
            m[p.stem].append(p)

    conflict = 0
    out: Dict[str, Path] = {}

    def _pick_one(paths: List[Path]) -> Path:
        rels = [(len(str(p.relative_to(root_dir))), str(p.relative_to(root_dir)).replace("\\", "/"), p) for p in paths]
        rels.sort(key=lambda x: (x[0], x[1]))
        return rels[0][2]

    for stem, paths in m.items():
        if len(paths) > 1:
            conflict += 1
        out[stem] = _pick_one(paths)

    return out, conflict


def _objects_to_detect_items(objects: Sequence[Tuple[str, int, int, int, int]], *, source: str) -> List[DetectItem]:
    items: List[DetectItem] = []
    for name, x1, y1, x2, y2 in objects:
        items.append(
            {
                "name": str(name),
                "score": 1.0,
                "location": [int(x1), int(y1), int(x2), int(y2)],
                "msg": "",
                "source": source,
            }
        )
    return items


def _replace_voc_objects(
    base_tree: ET.ElementTree,
    *,
    merged_objects: Sequence[Tuple[str, int, int, int, int]],
) -> ET.ElementTree:
    """
    以 base_tree 为模板，清空原 object 节点并写入 merged_objects。
    """
    tree = deepcopy(base_tree)
    root = tree.getroot()
    for obj in list(root.findall("object")):
        root.remove(obj)

    for name, x1, y1, x2, y2 in merged_objects:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = str(name)
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        bnd = ET.SubElement(obj, "bndbox")
        ET.SubElement(bnd, "xmin").text = str(int(x1))
        ET.SubElement(bnd, "ymin").text = str(int(y1))
        ET.SubElement(bnd, "xmax").text = str(int(x2))
        ET.SubElement(bnd, "ymax").text = str(int(y2))

    # python 3.9+ 支持缩进
    try:
        ET.indent(tree, space="\t")  # type: ignore[attr-defined]
    except Exception:
        pass
    return tree


def merge_two_voc_dirs(
    *,
    dir_a: str,
    dir_b: str,
    out_dir: str,
    iou_threshold: float,
    prefer_dir: str = "b",
    keep_unpaired: bool = True,
) -> None:
    """
    合并两个 VOC XML 目录（递归扫描 xml）。

    - **dir_b 优先**：`prefer_dir="b"` 时，合并后保留 dir_b 的框与类别名（与需求一致）。
    - 输出写入 out_dir，保持相对路径结构（xml key）。
    - 未配对 xml：keep_unpaired=True 时直接拷贝（以 prefer_dir 为准）。
    """
    a_root = Path(dir_a).expanduser().resolve()
    b_root = Path(dir_b).expanduser().resolve()
    out_root = Path(out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    a_map, conflict_a = _collect_xmls_by_stem(a_root)
    b_map, conflict_b = _collect_xmls_by_stem(b_root)
    keys = sorted(set(a_map.keys()) | set(b_map.keys()))

    prefer_source = "B" if str(prefer_dir).lower().strip() == "b" else "A"

    total = 0
    merged_count = 0
    only_a = 0
    only_b = 0
    warn_skipped_loc = 0

    for key in keys:
        pa = a_map.get(key)
        pb = b_map.get(key)

        if pa is None and pb is None:
            continue

        if pa is None or pb is None:
            if not keep_unpaired:
                continue
            src = pb if pa is None else pa
            if src is None:
                continue
            if pa is None:
                only_b += 1
            else:
                only_a += 1

            dst = out_root / (key + ".xml")
            dst.parent.mkdir(parents=True, exist_ok=True)
            # 直接拷贝原 xml（不改内容）
            dst.write_bytes(Path(src).read_bytes())
            total += 1
            continue

        # 两边都有：做 merge
        voc_a = parse_voc_xml(pa)
        voc_b = parse_voc_xml(pb)

        left_items = _objects_to_detect_items(voc_a.objects, source="A")
        right_items = _objects_to_detect_items(voc_b.objects, source="B")

        merged_items = merge_two_sources_by_iou(
            left_results=left_items,
            right_results=right_items,
            iou_threshold=float(iou_threshold),
            prefer_source=prefer_source,
        )

        merged_objects: List[Tuple[str, int, int, int, int]] = []
        for it in merged_items:
            loc = it.get("location")
            if not isinstance(loc, (list, tuple)) or len(loc) != 4:
                warn_skipped_loc += 1
                continue
            merged_objects.append(
                (str(it.get("name", "")), int(loc[0]), int(loc[1]), int(loc[2]), int(loc[3]))
            )

        base_tree = voc_b.tree if prefer_source == "B" else voc_a.tree
        out_tree = _replace_voc_objects(base_tree, merged_objects=merged_objects)

        out_path = out_root / (key + ".xml")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_tree.write(str(out_path), encoding="utf-8", xml_declaration=True)

        total += 1
        merged_count += 1

    print("==== merge_two_voc_dirs done ====")
    print("dir_a:", str(a_root))
    print("dir_b:", str(b_root))
    print("out_dir:", str(out_root))
    print("iou_threshold:", float(iou_threshold))
    print("prefer_dir:", prefer_dir, "(prefer_source:", prefer_source, ")")
    print("xml_total_written:", total)
    print("xml_merged:", merged_count)
    print("xml_only_a_copied:", only_a)
    print("xml_only_b_copied:", only_b)
    if conflict_a:
        print("warn_dir_a_stem_conflicts:", conflict_a, "(picked one path per stem)")
    if conflict_b:
        print("warn_dir_b_stem_conflicts:", conflict_b, "(picked one path per stem)")
    if warn_skipped_loc:
        print("warn_skipped_items_due_to_bad_location:", warn_skipped_loc)


if __name__ == "__main__":
    # ======= 直接在这里改参数运行（不使用命令行参数） =======
    DIR_A = "/Users/shunyaoyin/Documents/code/datasets/insect-data/小虫/daofeishi-0330-0331-1280-d1.1"
    DIR_B = "/Users/shunyaoyin/Documents/code/datasets/insect-data/小虫/daofeishi-0330-0331-1280"
    OUT_DIR = "/Users/shunyaoyin/Documents/code/datasets/insect-data/小虫/daofeishi-0330-0331-1280__merged_iou"

    IOU_THRESHOLD = 0.65
    PREFER_DIR = "b"  # "b" 表示后者（DIR_B）优先：保留其框与类别名
    KEEP_UNPAIRED = True
    CLEAR_OUT_DIR = False  # 输出目录已存在时是否清空 xml（避免残留影响观察）
    # ======================================================

    if not os.path.isdir(DIR_A):
        raise FileNotFoundError(f"DIR_A not found: {DIR_A}")
    if not os.path.isdir(DIR_B):
        raise FileNotFoundError(f"DIR_B not found: {DIR_B}")
    if CLEAR_OUT_DIR and os.path.isdir(OUT_DIR):
        for p in Path(OUT_DIR).rglob("*.xml"):
            try:
                p.unlink()
            except Exception:
                pass

    merge_two_voc_dirs(
        dir_a=DIR_A,
        dir_b=DIR_B,
        out_dir=OUT_DIR,
        iou_threshold=IOU_THRESHOLD,
        prefer_dir=PREFER_DIR,
        keep_unpaired=KEEP_UNPAIRED,
    )
