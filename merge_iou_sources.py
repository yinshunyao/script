#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DetectItem = Dict[str, Any]
Box = Tuple[float, float, float, float]  # (x1, y1, x2, y2) in xyxy


def _to_box_xyxy(item: DetectItem) -> Optional[Box]:
    loc = item.get("location")
    if not isinstance(loc, (list, tuple)) or len(loc) != 4:
        return None
    try:
        x1, y1, x2, y2 = float(loc[0]), float(loc[1]), float(loc[2]), float(loc[3])
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _box_iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = a_area + b_area - inter
    return float(inter / denom) if denom > 0 else 0.0


def _score(item: DetectItem) -> float:
    try:
        return float(item.get("score", 0) or 0)
    except Exception:
        return 0.0


def merge_two_sources_by_iou(
    left_results: Sequence[DetectItem],
    right_results: Sequence[DetectItem],
    *,
    iou_threshold: float,
    prefer_source: Optional[str] = None,
) -> List[DetectItem]:
    """
    合并同一张图片上「两个检测源」的检测框。

    - 使用 IoU >= iou_threshold 作为“同一目标”的判定。
    - 对满足阈值的候选对，按 IoU 从高到低贪心一对一匹配（每个框最多参与一次合并）。
    - 合并后输出 1 个框：优先保留 `prefer_source` 对应结果的 `location` 和 `name`。

    输入/输出 item 建议是 `script/predict_merge.py` 的统一格式：
    `{name, score, location=[x1,y1,x2,y2], msg, source}`。
    但本函数只强依赖 `location`，其余字段尽量保留/合并。
    """
    iou_thr = float(iou_threshold)
    if iou_thr <= 0:
        # 阈值 <= 0 视为不做合并（避免误吞框）
        return [deepcopy(x) for x in left_results] + [deepcopy(x) for x in right_results]

    left_boxes: List[Optional[Box]] = [_to_box_xyxy(x) for x in left_results]
    right_boxes: List[Optional[Box]] = [_to_box_xyxy(x) for x in right_results]

    pairs: List[Tuple[float, int, int]] = []
    for i, a in enumerate(left_boxes):
        if a is None:
            continue
        for j, b in enumerate(right_boxes):
            if b is None:
                continue
            v = _box_iou(a, b)
            if v >= iou_thr:
                pairs.append((v, i, j))
    pairs.sort(key=lambda x: x[0], reverse=True)

    used_left: set[int] = set()
    used_right: set[int] = set()

    merged: List[DetectItem] = []
    for iou, i, j in pairs:
        if i in used_left or j in used_right:
            continue
        used_left.add(i)
        used_right.add(j)

        li = left_results[i]
        rj = right_results[j]
        lsrc = li.get("source")
        rsrc = rj.get("source")

        # 选择保留项（框和类别名按优先源）
        keep = li
        drop = rj
        if prefer_source is not None:
            if rsrc == prefer_source and lsrc != prefer_source:
                keep, drop = rj, li
        else:
            if _score(rj) > _score(li):
                keep, drop = rj, li

        out = deepcopy(keep)

        # 其余字段尽量融合：score 取 max，msg/source/name/location 以 keep 为准
        out["score"] = max(_score(li), _score(rj))
        out["is_merged"] = True
        out["merge_iou"] = float(iou)
        out["merged_from_sources"] = [str(lsrc or ""), str(rsrc or "")]
        out["merged_from_names"] = [str(li.get("name", "")), str(rj.get("name", ""))]
        out["merged_from_locations"] = [deepcopy(li.get("location")), deepcopy(rj.get("location"))]

        msg_keep = str(keep.get("msg", "") or "")
        msg_drop = str(drop.get("msg", "") or "")
        if msg_keep and msg_drop and msg_drop not in msg_keep:
            out["msg"] = msg_keep + " | " + msg_drop
        elif not msg_keep and msg_drop:
            out["msg"] = msg_drop

        merged.append(out)

    # 未匹配到的框原样保留
    for i, item in enumerate(left_results):
        if i not in used_left:
            merged.append(deepcopy(item))
    for j, item in enumerate(right_results):
        if j not in used_right:
            merged.append(deepcopy(item))

    # 稳定排序：按 y1, x1
    def _key(x: DetectItem) -> Tuple[int, int]:
        loc = x.get("location")
        if isinstance(loc, (list, tuple)) and len(loc) == 4:
            try:
                return (int(loc[1]), int(loc[0]))
            except Exception:
                return (0, 0)
        return (0, 0)

    merged.sort(key=_key)
    return merged


def merge_results_by_iou_prefer_source(
    results: Sequence[DetectItem],
    *,
    source_left: str,
    source_right: str,
    iou_threshold: float,
    prefer_source: str,
) -> List[DetectItem]:
    """
    针对 `results`（单张图的混合来源输出），只在 `source_left` 与 `source_right` 两类之间做 IoU 合并。
    其余来源结果不参与合并，直接拼回返回。
    """
    left = [x for x in results if x.get("source") == source_left]
    right = [x for x in results if x.get("source") == source_right]
    others = [x for x in results if x.get("source") not in (source_left, source_right)]

    merged_lr = merge_two_sources_by_iou(
        left_results=left,
        right_results=right,
        iou_threshold=iou_threshold,
        prefer_source=prefer_source,
    )
    return merged_lr + [deepcopy(x) for x in others]


if __name__ == "__main__":
    # 直接运行的小验证（不依赖图片/模型）
    r1 = {"name": "a", "score": 0.6, "location": [10, 10, 50, 50], "source": "s1", "msg": ""}
    r2 = {"name": "b", "score": 0.9, "location": [12, 12, 52, 52], "source": "s2", "msg": "from s2"}
    r3 = {"name": "c", "score": 0.5, "location": [200, 200, 240, 240], "source": "s2", "msg": ""}

    out = merge_results_by_iou_prefer_source(
        [r1, r2, r3],
        source_left="s1",
        source_right="s2",
        iou_threshold=0.5,
        prefer_source="s1",
    )
    for x in out:
        print(x)
