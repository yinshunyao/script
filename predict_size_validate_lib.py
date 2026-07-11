#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

# @Detail  : VOC 验证共用库：几何匹配（IoU/IoR）、类别合并、分级父子一致（可选 hierarchy_type_index）、
#            ``draw_main_output_image`` / ``parse_pascal_voc_objects`` 等。
#
# 评估统计/混淆矩阵导出见本文件末尾（``_print_stat_by_cls``、``_std_eval_*`` 等）。
import csv
import json
import logging
import os
import re
import sys
import warnings
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
import unicodedata

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_OTHER_ZH_PREFIX = "其他"

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


def resolve_eval_label(
    raw: str,
    label_alias_map: dict[str, str] | None = None,
) -> str:
    """
    评估用类名解析：将 xml/标注中的中文名映射为 canonical（拼音/infer_name）。

    - ``label_alias_map`` 命中则返回映射值；
    - 以「其他」开头的中文视为 ``other``；
    - 非中文或未命中映射时原样返回（trim 后）。
    """
    s = str(raw or "").strip()
    if not s:
        return ""
    if _CJK_RE.search(s) and s.startswith(_OTHER_ZH_PREFIX):
        return "other"
    if label_alias_map:
        hit = label_alias_map.get(s)
        if hit:
            return str(hit).strip()
    return s


_FILENAME_GT_TOKEN_SEP_RE = re.compile(r"[_\-\s\.]+")


def _build_pinyin_gt_lookup(label_alias_map: dict[str, str] | None) -> frozenset[str]:
    """评估用：从别名表收集可作为文件名 GT 的拼音/canonical 键（小写）。"""
    keys: set[str] = set()
    if not label_alias_map:
        return frozenset()
    for zh, py in label_alias_map.items():
        py_s = str(py or "").strip().lower()
        if py_s:
            keys.add(py_s)
        zh_s = str(zh or "").strip()
        if zh_s and not _CJK_RE.search(zh_s):
            keys.add(zh_s.lower())
    return frozenset(keys)


def infer_gt_label_from_filename_stem(
    stem: str,
    label_alias_map: dict[str, str] | None = None,
) -> str | None:
    """
    从图片文件名 stem 推断 GT 类别（无 VOC xml 时的分类/单目标测试集）。

    支持整段中文名（如 ``灰飞虱``）、拼音键（如 ``huifeishi``），以及
    ``灰飞虱_001``、``test-huifeishi-01`` 等含分隔符文件名中的中文/拼音片段。
    命中 ``label_alias_map`` 后返回原始标签串，后续由 ``resolve_eval_label`` 归一化。
    """
    s = str(stem or "").strip()
    if not s:
        return None

    mapping = label_alias_map or {}

    if s in mapping:
        return s

    cjk_keys = [k for k in mapping if _CJK_RE.search(str(k))]
    for k in sorted(cjk_keys, key=len, reverse=True):
        if k in s:
            return str(k)

    pinyin_set = _build_pinyin_gt_lookup(mapping)
    s_lower = s.lower()
    if s_lower in pinyin_set:
        return s_lower

    for tok in _FILENAME_GT_TOKEN_SEP_RE.split(s):
        t = str(tok or "").strip()
        if not t:
            continue
        if t in mapping:
            return t
        tl = t.lower()
        if tl in pinyin_set:
            return tl

    return None


def build_filename_pseudo_gt_objects(
    stem: str,
    *,
    label_alias_map: dict[str, str] | None = None,
) -> list[dict[str, Any]] | None:
    """
    无 xml 时按文件名推断图级 GT 类别（仅 ``name``，不含几何框）。

    校验口径：文件名不代表整图都是该虫，而是图内**报出的检测框**均应判为该种类。
    """
    label = infer_gt_label_from_filename_stem(stem, label_alias_map)
    if not label:
        return None
    return [{"name": label}]


def _canonical_from_alg_out_entry(cls_key: str, entry: dict[str, Any]) -> str:
    """评估 canonical 用 ``out`` 路由 key；``infer_name`` 另登记为别名指向 key。"""
    key = str(cls_key or "").strip()
    if key == "*":
        return "insect"
    return key


def _is_eval_out_route_key(key: str) -> bool:
    """路由占位键（通配、直径区间等）不参与评估类名别名映射。"""
    k = str(key or "").strip()
    return not k or k == "*" or k.startswith("[")


def _register_out_entry_eval_aliases(
    cls_key: str,
    entry: dict[str, Any],
    mapping: dict[str, str],
    *,
    info_catalog: dict[str, Any] | None = None,
) -> None:
    """
    将 ``out`` 条目的 key / ``cn_name`` / ``infer_name`` 及 ``insect_info`` 中文名
    统一登记到同一 canonical（**路由 key**；``infer_name`` 作为外部名别名指向 key）。
    """
    canonical = _canonical_from_alg_out_entry(str(cls_key), entry)
    if not canonical:
        return
    cn = str(entry.get("cn_name") or "").strip()
    infer = str(entry.get("infer_name") or "").strip()
    key = str(cls_key or "").strip()
    if cn:
        mapping[cn] = canonical
    if infer:
        mapping[infer] = canonical
    if key and not _is_eval_out_route_key(key):
        mapping[key] = canonical
    if info_catalog is not None and key in info_catalog:
        rec = info_catalog[key]
        zh = str(getattr(rec, "name_zh", "") or "").strip()
        if zh:
            mapping[zh] = canonical


def _collect_cn_aliases_from_alg_node(
    node: Any,
    out: dict[str, str],
    *,
    info_catalog: dict[str, Any] | None = None,
) -> None:
    """递归收集 ``insect_alg_all`` 配置树中各类名别名 → canonical 映射。"""
    if isinstance(node, dict):
        out_table = node.get("out")
        if isinstance(out_table, dict):
            for cls_key, entry in out_table.items():
                if not isinstance(entry, dict):
                    continue
                _register_out_entry_eval_aliases(
                    str(cls_key),
                    entry,
                    out,
                    info_catalog=info_catalog,
                )
                _collect_cn_aliases_from_alg_node(
                    entry, out, info_catalog=info_catalog
                )
        for key in ("models", "cls"):
            sub = node.get(key)
            if isinstance(sub, dict):
                _collect_cn_aliases_from_alg_node(
                    sub, out, info_catalog=info_catalog
                )
        # 顶层 models.{detect_big|...} 等根节点容器（本身无 out 键名时亦须下钻）
        for k, v in node.items():
            if k in ("out", "models", "cls"):
                continue
            if isinstance(v, dict):
                _collect_cn_aliases_from_alg_node(
                    v, out, info_catalog=info_catalog
                )
    elif isinstance(node, list):
        for item in node:
            _collect_cn_aliases_from_alg_node(
                item, out, info_catalog=info_catalog
            )


@dataclass(frozen=True)
class ClassTierEquivalence:
    """
    ``postprocess.class_tier_aliases`` 的无向等价类索引。

    每条 ``key: value`` 仅表示两类可互换等价（不做 canonical 覆盖）；多组 kv 经并查集传递
    （如 ``A~B``、``B~C`` ⇒ ``A~C``）。统计合并时 ``value`` 侧为 preferred canonical 名。
    """

    _parent: dict[str, str]
    _stat_canonical: dict[str, str]

    @classmethod
    def empty(cls) -> ClassTierEquivalence:
        return cls(_parent={}, _stat_canonical={})

    @classmethod
    def from_tier_aliases(cls, aliases: dict[str, Any] | None) -> ClassTierEquivalence:
        parent: dict[str, str] = {}
        stat_canonical: dict[str, str] = {}

        def _find(x: str) -> str:
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(a: str, b: str) -> None:
            ra, rb = _find(a), _find(b)
            if ra != rb:
                parent[rb] = ra

        if isinstance(aliases, dict):
            for key, value in aliases.items():
                k = str(key or "").strip()
                v = str(value or "").strip()
                if k and v:
                    _union(k, v)
                    root = _find(v)
                    stat_canonical[root] = v
        return cls(_parent=parent, _stat_canonical=stat_canonical)

    def _find(self, name: str) -> str:
        x = str(name or "").strip()
        if not x:
            return ""
        parent = self._parent
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def are_equivalent(self, a: str, b: str) -> bool:
        a = str(a or "").strip()
        b = str(b or "").strip()
        if not a or not b:
            return False
        if a == b:
            return True
        if not self._parent:
            return False
        ra = self._find(a)
        rb = self._find(b)
        if not ra or not rb:
            return False
        return ra == rb

    def canonical_stat_name(self, name: str) -> str:
        """
        统计/比赛计数用 canonical 类键：优先 ``class_tier_aliases`` 的 value 侧，否则取最短名。
        """
        x = str(name or "").strip()
        if not x or not self._parent:
            return x
        root = self._find(x)
        if not root:
            return x
        pref = self._stat_canonical.get(root)
        if pref:
            return pref
        members = {x, root}
        for node in self._parent:
            if self._find(node) == root:
                members.add(node)
        return min(members, key=lambda n: (len(n), n))

    def expand_names(self, names: tuple[str, ...] | list[str]) -> frozenset[str]:
        """将白名单类名扩展为与 ``class_tier_aliases`` 等价闭包内的全部成员。"""
        allowed = {str(name or "").strip() for name in names if str(name or "").strip()}
        if not allowed or not self._parent:
            return frozenset(allowed)
        by_root: dict[str, set[str]] = {}
        for node in self._parent:
            by_root.setdefault(self._find(node), set()).add(node)
        expanded = set(allowed)
        for members in by_root.values():
            if members & allowed:
                expanded |= members
        return frozenset(expanded)


def build_class_tier_equivalence(
    tier_aliases: dict[str, Any] | None,
) -> ClassTierEquivalence:
    """由 ``postprocess.class_tier_aliases`` 构建可传递的类型等价索引。"""
    return ClassTierEquivalence.from_tier_aliases(tier_aliases)


def load_class_tier_equivalence(
    *,
    alg_config: dict[str, Any] | None = None,
) -> ClassTierEquivalence:
    """从 alg 配置的 ``postprocess.class_tier_aliases`` 加载类型等价索引。"""
    if not isinstance(alg_config, dict):
        return ClassTierEquivalence.empty()
    postprocess = alg_config.get("postprocess")
    if not isinstance(postprocess, dict):
        return ClassTierEquivalence.empty()
    return build_class_tier_equivalence(postprocess.get("class_tier_aliases"))


def build_eval_label_alias_map(
    *,
    alg_config: dict[str, Any] | None = None,
) -> dict[str, str]:
    """
    构建评估用「展示名/中文标注 → canonical 类名」表。

    来源（后者覆盖前者）：
    1. ``insect_info.json`` 的 ``name_zh`` → 拼音 key；
    2. ``insect_alg_all``（或传入的 ``alg_config``）各 ``out`` 条目的 key / ``cn_name`` /
       ``infer_name`` 及对应 ``insect_info`` 中文名 → 同一 canonical（**路由 key**）；
    3. 比赛清单 ``competition_cn_aliases``（覆盖 alg 中同 cn 多 key 冲突，如
       ``zhonghuaxingbujia`` / ``zhonghualanbujia`` 的 **单向** 拼音别名）；
    4. 常用中文兜底（如「灰飞虱」→ ``huifeishi``，「其他」→ ``other``）。

    ``postprocess.class_tier_aliases`` **不**写入本表；类型等价见 ``ClassTierEquivalence`` /
    ``is_class_match(..., tier_equivalence=...)``。
    """
    mapping: dict[str, str] = {"其他": "other", "未知": "unknown", "昆虫": "insect"}

    info_catalog: dict[str, Any] | None = None
    try:
        from script.config.insect_info import load_json_catalog  # noqa: PLC0415
    except ImportError:
        load_json_catalog = None  # type: ignore[misc, assignment]
    if load_json_catalog is not None:
        try:
            info_catalog = load_json_catalog()
            for pinyin, rec in info_catalog.items():
                zh = str(getattr(rec, "name_zh", "") or "").strip()
                if zh:
                    mapping.setdefault(zh, str(pinyin))
        except Exception as e:
            logging.warning("构建评估中文映射：加载 insect_info 失败: %s", e)
            info_catalog = None

    # insect_info 未收录时的少量兜底（与 ls_classification_ingest 一致，反向映射）
    for pinyin, zh in (
        ("huifeishi", "灰飞虱"),
        ("other", "其他"),
        ("unknown", "未知"),
    ):
        mapping.setdefault(zh, pinyin)

    # 比赛清单 canonical 与历史异名（见 script/config/competition_cn_aliases.py）
    try:
        from script.config.competition_cn_aliases import (  # noqa: PLC0415
            COMPETITION_CANONICAL_CN,
            COMPETITION_CN_ALIASES,
            COMPETITION_PINYIN_ALIASES,
        )
    except ImportError:
        COMPETITION_CANONICAL_CN = {}  # type: ignore[misc, assignment]
        COMPETITION_CN_ALIASES = {}  # type: ignore[misc, assignment]
        COMPETITION_PINYIN_ALIASES = {}  # type: ignore[misc, assignment]

    for pinyin, zh in COMPETITION_CANONICAL_CN.items():
        if zh:
            mapping[zh] = str(pinyin)

    for legacy_zh, pinyin in COMPETITION_CN_ALIASES.items():
        mapping.setdefault(legacy_zh, pinyin)

    for src_py, dst_py in COMPETITION_PINYIN_ALIASES.items():
        mapping[src_py] = dst_py

    # 历史简称标注兼容
    for legacy_zh, pinyin in (
        ("国槐尺蠖", "huaichie"),  # 与槐尺蛾同种，标注名与模型 cn_name 不一致
        ("guohuaichihuo", "huaichie"),
    ):
        mapping.setdefault(legacy_zh, pinyin)

    if isinstance(alg_config, dict):
        _collect_cn_aliases_from_alg_node(
            alg_config.get("models"),
            mapping,
            info_catalog=info_catalog,
        )
        # alg ``out`` 可能为同一 cn_name 注册多个路由 key（如清单 zhonghuaxingbujia 与训练
        # zhonghualanbujia）；比赛拼音别名须在 alg 之后再次覆盖。
        for legacy_zh, pinyin in COMPETITION_CN_ALIASES.items():
            mapping[str(legacy_zh)] = str(pinyin)
        for src_py, dst_py in COMPETITION_PINYIN_ALIASES.items():
            mapping[str(src_py)] = str(dst_py)

    return mapping


@lru_cache(maxsize=4)
def _cached_eval_label_alias_map_from_alg_path(alg_path: str) -> dict[str, str]:
    p = Path(alg_path)
    alg_config: dict[str, Any] | None = None
    if p.is_file():
        try:
            with open(p, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                alg_config = raw
        except Exception as e:
            logging.warning("构建评估中文映射：读取 %s 失败: %s", p, e)
    return build_eval_label_alias_map(alg_config=alg_config)


def load_eval_label_alias_map(
    *,
    alg_config: dict[str, Any] | None = None,
    alg_config_path: str | Path | None = None,
) -> dict[str, str]:
    """
    加载评估用中文/别名 → canonical 映射。

    优先使用内存中的 ``alg_config``；否则按 ``alg_config_path`` 读取并缓存。
    """
    if isinstance(alg_config, dict):
        return build_eval_label_alias_map(alg_config=alg_config)
    if alg_config_path is not None:
        return _cached_eval_label_alias_map_from_alg_path(str(alg_config_path))
    return build_eval_label_alias_map()


def normalize_class_name(
    raw: str,
    merge: dict[str, list[str]] | None,
    *,
    label_alias_map: dict[str, str] | None = None,
) -> str:
    """
    配置内映射到组 key；否则原样。

    注意：'*' 仅用于「insect 粗分类可匹配任意具体昆虫」的 **匹配规则**（见 is_class_match），
    不应作为 normalize 的兜底归一规则，否则会把所有未显式列出的类别都压扁成同一个组，影响统计。
    """
    raw = resolve_eval_label(raw, label_alias_map)
    if not raw:
        return ""
    alias = _build_class_alias_map(merge)
    hit = alias.get(raw)
    if hit is not None:
        return hit
    return raw


def is_other_prefix_classname(raw: str) -> bool:
    """类别名以 ``other`` 开头（不区分大小写），用于灰色绘制与指标忽略（含 ``other_part`` 等）。"""
    s = str(raw or "").strip().lower()
    return bool(s) and s.startswith("other")


_OTHER_CLS_CONF_REASON_RE = re.compile(
    r"^other_cls_conf:([^(]+)\(([\d.]+)<=([\d.]+)\)\s*$"
)


def _parse_other_cls_conf_reason(
    reason: str | None,
) -> tuple[str, float, float] | None:
    """从 ``other_cls_conf:类名(conf<=thr)`` 解析 demote 前的 top1 与门限。"""
    r = str(reason or "").strip()
    m = _OTHER_CLS_CONF_REASON_RE.match(r)
    if not m:
        return None
    try:
        return m.group(1).strip(), float(m.group(2)), float(m.group(3))
    except (TypeError, ValueError):
        return None


def _topk_item_class_name(item: Any) -> str:
    if isinstance(item, dict):
        return str(
            item.get("class_name") or item.get("name") or ""
        ).strip()
    if isinstance(item, str):
        return item.strip()
    return ""


def eval_pred_row_display_class(row: dict[str, Any]) -> str:
    """
    混淆矩阵列名 / 类型混淆切图 ``pred=`` 目录名。

    对外报出为 ``other`` / ``other_seg`` 等后置兜底类时，回溯模型原始 top1 类名；
    否则使用 ``cls_name``（校验 pred dict）或 ``name``。
    """
    cls_name = str(row.get("cls_name") or "").strip()
    name = str(row.get("name") or "").strip()
    if cls_name and not is_other_prefix_classname(cls_name):
        return cls_name
    if not is_other_prefix_classname(cls_name) and not is_other_prefix_classname(name):
        return cls_name or name or "?"

    parsed = _parse_other_cls_conf_reason(
        str(row.get("cls_other_reason") or row.get("filter_reason") or "")
    )
    if parsed is not None:
        nm = parsed[0]
        if nm and not is_other_prefix_classname(nm):
            return nm
    for key in ("cls_name_top1", "viz_cls_name"):
        v = str(row.get(key) or "").strip()
        if v and not is_other_prefix_classname(v):
            return v
    for field in ("cls_topk", "cls_top3", "cls_topn"):
        for item in row.get(field) or []:
            v = _topk_item_class_name(item)
            if v and not is_other_prefix_classname(v):
                return v
    seg = str(row.get("seg_cls_name") or row.get("class_name") or "").strip()
    if seg and not is_other_prefix_classname(seg):
        return seg
    return cls_name or name or "?"


def is_fuzzy_eval_class(
    raw: str,
    *,
    label_alias_map: dict[str, str] | None = None,
) -> bool:
    """
    评估用「粗/兜底」类名：``insect`` 与 ``other`` 前缀族。

    在 ``fuzzy_only_wildcard`` 模式下，仅此类之间可通过 ``insect:['*']`` 通配或分级树
    ``insect`` 字面量互匹；具体物种（如 ``xiaocaie``）不与粗类互匹为 TP。
    """
    s = resolve_eval_label(raw, label_alias_map)
    if not s:
        return False
    if s.lower() == "insect" or s == "昆虫":
        return True
    return is_other_prefix_classname(s)


def is_metric_ignored_other(raw: str, merge: dict[str, list[str]] | None) -> bool:
    """
    ``other`` 类在生产环境不输出；验证脚本中该类不参与 TP/FP/FN、按类统计、混淆矩阵等指标计算。
    判定：归一化后与 ``other`` 等价，或类名以 ``other`` 开头（如 ``other_part``）。
    """
    # 通配符 '*' 的语义：匹配“包括 other 在内”的任意类别。
    # 一旦启用 wildcard 组（如 'insect': ['*']），则不应再把 other 从评估/匹配集合中剔除，
    # 否则会出现“miss gt=other 永远无法匹配”的现象。
    if _get_wildcard_group_key(merge):
        return False
    raw_s = str(raw or "").strip()
    if is_other_prefix_classname(raw_s):
        return True
    norm = normalize_class_name(raw_s, merge)
    if is_other_prefix_classname(norm):
        return True
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
    - dilaohu = 地老虎/二点委/三叉等 + 粘虫类（与 similar_detect 中 ``dilaohunian`` 外观组一致；兼容别名 ``dilaohu_nian``）及其细分
    - yee = 各种 *yee 及其细分（若 tiancaiyee/huangyeming 被归到 other，也计入 yee 兼容）
    """
    super_groups: dict[str, set[str]] = {}

    # dilaohu 大类（含粘虫：成虫多视角易与地老虎类混淆）
    dilaohu_keys = {
        "bazidilaohu",
        "chaseidilaohu",
        "dadilaohu",
        "huangdilaohu",
        "xiaodilaohu",
        "erdianweiyee",
        "sanchadilaohu",
        "danmainianchong",
        "dongfangnianchong",
        "laoshinianchong",
    }
    dilaohu_set: set[str] = set()
    for k in dilaohu_keys:
        if k in groups:
            dilaohu_set |= set(groups[k])
    if dilaohu_set:
        dilaohu_set |= set(dilaohu_keys)
        dilaohu_set.update(("dilaohu", "dilaohu_nian", "dilaohunian", "nian"))
        super_groups["dilaohu"] = dilaohu_set
        super_groups["dilaohu_nian"] = dilaohu_set
        super_groups["dilaohunian"] = dilaohu_set
        super_groups["nian"] = dilaohu_set

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
        yee_set.add("ye_e")
        super_groups["yee"] = yee_set

    return super_groups


def _is_hierarchy_leaf_number(x: object) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _walk_cls_hierarchy_edges(
    sub: dict[str, Any],
    parent: str,
    edges: list[tuple[str, str]],
    nodes: set[str],
    node_to_top: dict[str, str],
    top_root: str,
) -> None:
    """从 cls_merge 分级对象收集 (父 key -> 子 key) 边；结构与检测训练校验一致。"""
    if not sub:
        raise ValueError(f"分级配置在 {parent!r} 下存在空对象")
    vals = list(sub.values())
    all_leaf = all(_is_hierarchy_leaf_number(v) for v in vals)
    all_nested = all(isinstance(v, dict) for v in vals)
    if all_leaf:
        for folder, n_raw in sub.items():
            if not isinstance(folder, str) or not str(folder).strip():
                raise ValueError(f"分级配置在 {parent!r} 下存在非法叶 key")
            if isinstance(n_raw, bool) or not isinstance(n_raw, (int, float)):
                raise ValueError(
                    f"分级配置叶 value 须为数值（父={parent!r} key={folder!r}）"
                )
            key = str(folder).strip()
            if key:
                nodes.add(key)
                edges.append((parent, key))
                node_to_top[key] = top_root
        return
    if all_nested:
        for k, v in sub.items():
            if not isinstance(k, str) or not str(k).strip():
                raise ValueError(f"分级配置在 {parent!r} 下存在非法分组 key")
            seg = str(k).strip()
            if not seg:
                continue
            nodes.add(seg)
            edges.append((parent, seg))
            node_to_top[seg] = top_root
            if not isinstance(v, dict):
                raise ValueError(f"分级配置在 {parent!r}/{seg!r} 内部结构异常")
            _walk_cls_hierarchy_edges(v, seg, edges, nodes, node_to_top, top_root)
        return
    raise ValueError(
        f"分级配置在 {parent!r} 处结构非法：须为「全叶(目录名->数值)」或「全嵌套子对象」"
    )


@dataclass(frozen=True)
class ClsHierarchyIndex:
    """``cls_merge`` 分级树上的类型关系：同一链路上的祖先/后代视为类型一致。"""

    all_nodes: frozenset[str]
    descendants: dict[str, frozenset[str]]
    # 每个节点（含中间层、叶目录名）→ 所属 JSON 顶层类 key；混淆矩阵按顶层父类归并统计
    node_to_top: dict[str, str]

    def rollup_to_top(self, normalized_name: str) -> str:
        """将 ``normalize_class_name`` 后的类名归并到 cls_merge 顶层父类；不在树内则原样返回。"""
        n = str(normalized_name or "").strip()
        if not n:
            return "?"
        top = self.node_to_top.get(n)
        if top is not None:
            return top
        return n

    def is_hierarchy_consistent(
        self,
        pred_raw: str,
        gt_raw: str,
        *,
        fuzzy_only_wildcard: bool = False,
        label_alias_map: dict[str, str] | None = None,
    ) -> bool:
        """
        标注类名与检出/分类类名在分级树中成父子（含隔代）关系则一致。
        另：字面量 ``insect``（大小写不敏感）在启用本索引时视为通配符；``fuzzy_only_wildcard``
        为真时仅与同为粗/兜底类（``insect``、``other`` 前缀族）互匹。
        """
        a = str(pred_raw or "").strip()
        b = str(gt_raw or "").strip()
        if not a or not b:
            return False
        if a.lower() == "insect" or b.lower() == "insect":
            if fuzzy_only_wildcard:
                return is_fuzzy_eval_class(
                    a, label_alias_map=label_alias_map
                ) and is_fuzzy_eval_class(b, label_alias_map=label_alias_map)
            return True
        if a == b:
            return True
        if a not in self.all_nodes or b not in self.all_nodes:
            return False
        da = self.descendants.get(a, frozenset())
        db = self.descendants.get(b, frozenset())
        return b in da or a in db


def build_cls_hierarchy_type_index(obj: dict[str, Any]) -> ClsHierarchyIndex:
    """由已解析的分级根对象构建 ``ClsHierarchyIndex``。"""
    edges: list[tuple[str, str]] = []
    nodes: set[str] = set()
    node_to_top: dict[str, str] = {}
    for top_key, body in obj.items():
        if not isinstance(top_key, str) or not str(top_key).strip():
            raise ValueError("分级配置顶层 key 须为非空字符串")
        tk = str(top_key).strip()
        if not isinstance(body, dict) or not body:
            raise ValueError(f"分级配置顶层 {tk!r} 的值须为非空对象")
        nodes.add(tk)
        node_to_top[tk] = tk
        _walk_cls_hierarchy_edges(body, tk, edges, nodes, node_to_top, tk)
    children: dict[str, set[str]] = {}
    for pa, ch in edges:
        children.setdefault(pa, set()).add(ch)
    alln: set[str] = set(nodes) | {ch for _pa, ch in edges}
    descendants_map: dict[str, frozenset[str]] = {}
    for n in alln:
        acc: set[str] = set()
        stack = list(children.get(n, ()))
        seen_walk: set[str] = set()
        while stack:
            c = stack.pop()
            if c in seen_walk:
                continue
            seen_walk.add(c)
            acc.add(c)
            stack.extend(children.get(c, ()))
        descendants_map[n] = frozenset(acc)
    return ClsHierarchyIndex(frozenset(alln), descendants_map, node_to_top)


def confusion_matrix_stat_label(
    raw: str,
    merge: dict[str, list[str]] | None,
    hierarchy_type_index: ClsHierarchyIndex | None,
    *,
    label_alias_map: dict[str, str] | None = None,
) -> str:
    """
    混淆矩阵行/列标签：先做 ``normalize_class_name``；若启用分级索引，再归并到顶层父类。
    """
    n = normalize_class_name(
        str(raw or "").strip(), merge, label_alias_map=label_alias_map
    ) or "?"
    if hierarchy_type_index is None:
        return n
    return hierarchy_type_index.rollup_to_top(n)


def is_class_match(
    pred_raw: str,
    gt_raw: str,
    merge: dict[str, list[str]] | None,
    hierarchy_type_index: ClsHierarchyIndex | None = None,
    *,
    label_alias_map: dict[str, str] | None = None,
    fuzzy_only_wildcard: bool = False,
    tier_equivalence: ClassTierEquivalence | None = None,
) -> bool:
    """
    类别比对放宽：标注/预测都可能是粗分类或细分类。
    若二者在配置映射关系中可归为同一组（含 super group：dilaohu/yee），则算正确；
    不在配置中的类别仍按名称精确匹配。

    若 ``hierarchy_type_index`` 非空，则在上述规则未命中时追加：二者在 ``cls_merge`` 分级树上为
    祖先-后代则视为类型一致；字面量 ``insect`` 与任意非空类名一致（``fuzzy_only_wildcard`` 为真时
    仅与 ``insect``/``other`` 前缀族互匹）。

    ``fuzzy_only_wildcard``：为真且 merge 含 ``insect:['*']`` 时，通配仅适用于粗/兜底类彼此，
    具体物种不与 ``insect``/``other`` 互匹为 TP。

    ``label_alias_map`` 可将 xml 中的中文 ``<name>`` 解析为 canonical 后再比对（见
    ``load_eval_label_alias_map``）。

    ``tier_equivalence``：``postprocess.class_tier_aliases`` 的无向等价类（可传递），不参与
    ``label_alias_map`` 的 canonical 覆盖。
    """
    pred_raw = resolve_eval_label(pred_raw, label_alias_map)
    gt_raw = resolve_eval_label(gt_raw, label_alias_map)
    if not pred_raw or not gt_raw:
        return False

    # other 开头（含 other_part / other_small 等）为非目标/兜底语义：
    # 只能与同为 other 开头的类互相匹配（彼此不要求精确同名）；
    # 一侧为 other、另一侧为真实类时判不匹配（属类型错误）。
    pred_is_other = is_other_prefix_classname(pred_raw)
    gt_is_other = is_other_prefix_classname(gt_raw)
    if pred_is_other or gt_is_other:
        if pred_is_other and gt_is_other:
            return True
        # 一侧 other、一侧 insect 等粗类：在严格通配模式下仍视为互匹
        if fuzzy_only_wildcard and is_fuzzy_eval_class(
            pred_raw, label_alias_map=label_alias_map
        ) and is_fuzzy_eval_class(gt_raw, label_alias_map=label_alias_map):
            return True
        return False

    if tier_equivalence is not None and tier_equivalence.are_equivalent(
        pred_raw, gt_raw
    ):
        return True

    def _l1_ok() -> bool:
        return hierarchy_type_index is not None and hierarchy_type_index.is_hierarchy_consistent(
            pred_raw,
            gt_raw,
            fuzzy_only_wildcard=fuzzy_only_wildcard,
            label_alias_map=label_alias_map,
        )

    wildcard_key = _get_wildcard_group_key(merge)
    if wildcard_key:
        pred_norm = normalize_class_name(
            pred_raw, merge, label_alias_map=label_alias_map
        )
        gt_norm = normalize_class_name(
            gt_raw, merge, label_alias_map=label_alias_map
        )
        pred_wc = pred_norm == wildcard_key
        gt_wc = gt_norm == wildcard_key
        if pred_wc or gt_wc:
            if fuzzy_only_wildcard:
                return is_fuzzy_eval_class(
                    pred_raw, label_alias_map=label_alias_map
                ) and is_fuzzy_eval_class(
                    gt_raw, label_alias_map=label_alias_map
                )
            # 任一侧归一后为 wildcard 组：视为昆虫粗分类匹配任意具体昆虫
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
            # 组 key 本身即「粗类」（_build_class_groups 把 key 也并入成员集合）：
            # 粗类可与组内任意具体类互相匹配；两侧均为不同的具体类时判类型错误
            # （仍保留分级树祖先-后代关系作为兜底）。
            if pred_raw == gt_raw:
                return True
            if pred_raw == pg or gt_raw == gg:
                return True
            return _l1_ok()
        if pred_raw in gm and gt_raw in pm:
            return True
        if _l1_ok():
            return True
        return False

    # 只有一边在组内：要求另一边正好是该组 key 或该组成员
    if pm is not None and gm is None:
        return gt_raw in pm or _l1_ok()
    if gm is not None and pm is None:
        return pred_raw in gm or _l1_ok()

    # 都不在配置中：精确匹配
    return pred_raw == gt_raw or _l1_ok()


_VOC_XML_ENCODINGS = ("utf-8", "gbk", "gb2312", "latin-1")


def _parse_voc_xml_tree(xml_path: str) -> ET.ElementTree:
    """
    解析 Pascal VOC xml。

    部分历史标注未声明 encoding，中文 ``<name>`` 为 GBK/GB2312 字节；
    ``ET.parse`` 默认按 UTF-8 会报 ``not well-formed (invalid token)``。
    另：LabelMe / 部分工具导出的 UTF-8 文件带 BOM，需在解析前剔除。
    """
    raw = Path(xml_path).read_bytes()
    if not raw:
        raise ValueError(f"empty xml: {xml_path}")
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    parse_err: Exception | None = None
    for enc in _VOC_XML_ENCODINGS:
        try:
            text = raw.decode(enc)
        except UnicodeDecodeError:
            continue
        text = text.lstrip("\ufeff")
        try:
            return ET.ElementTree(ET.fromstring(text))
        except ET.ParseError as e:
            parse_err = e
            continue
    if parse_err is not None:
        raise parse_err
    raise ValueError(f"cannot decode voc xml: {xml_path}")


def parse_pascal_voc_objects(xml_path: str) -> list[dict[str, Any]]:
    """读取 VOC xml，返回 object 列表：name, x1,y1,x2,y2。"""
    tree = _parse_voc_xml_tree(xml_path)
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


def _voc_object_float_el(obj: Any, tag: str) -> float | None:
    el = obj.find(tag) if hasattr(obj, "find") else None
    if el is None or el.text is None:
        return None
    try:
        return float(el.text.strip())
    except (TypeError, ValueError):
        return None


def voc_xml_has_dual_confidence(xml_path: str | Path) -> bool:
    """预测 xml 是否含 ``det_conf`` / ``cls_conf``（``OUTPUT_XML_DUAL_CONF`` 落盘）。"""
    try:
        tree = _parse_voc_xml_tree(str(xml_path))
    except (OSError, ValueError, ET.ParseError):
        return False
    for obj in tree.getroot().findall("object"):
        if obj.find("det_conf") is not None or obj.find("cls_conf") is not None:
            return True
    return False


def parse_pascal_voc_pred_objects(xml_path: str | Path) -> list[dict[str, Any]]:
    """
    读取预测 VOC xml：name、bbox、det_conf、cls_conf。

    与 ``predict_all`` 的 ``OUTPUT_XML_DUAL_CONF`` 输出一致；无置信度字段时回退 0。
    """
    tree = _parse_voc_xml_tree(str(xml_path))
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

        det_conf = _voc_object_float_el(obj, "det_conf")
        if det_conf is None:
            det_conf = _voc_object_float_el(obj, "conf")
        cls_conf = _voc_object_float_el(obj, "cls_conf")
        if det_conf is None and cls_conf is not None:
            det_conf = cls_conf
        if cls_conf is None:
            cls_conf = det_conf if det_conf is not None else 0.0
        if det_conf is None:
            det_conf = float(cls_conf or 0.0)
        det_f = float(det_conf)
        cls_f = float(cls_conf)
        row_out: dict[str, Any] = {
            "name": name,
            "cls_name": name,
            "cn_name": name,
            "x1": _int("xmin"),
            "y1": _int("ymin"),
            "x2": _int("xmax"),
            "y2": _int("ymax"),
            "location": [
                _int("xmin"),
                _int("ymin"),
                _int("xmax"),
                _int("ymax"),
            ],
            "score": det_f,
            "conf": det_f,
            "det_conf": det_f,
            "cls_conf": cls_f,
        }
        cls_topn_el = obj.find("cls_topn")
        if cls_topn_el is not None:
            cls_topk: list[dict[str, Any]] = []
            for item in cls_topn_el.findall("item"):
                name_el = item.find("name")
                conf_el = item.find("conf")
                if name_el is None or not (name_el.text or "").strip():
                    continue
                top_name = name_el.text.strip()
                top_conf = 0.0
                if conf_el is not None and conf_el.text is not None:
                    try:
                        top_conf = float(conf_el.text.strip())
                    except (TypeError, ValueError):
                        top_conf = 0.0
                cls_topk.append(
                    {
                        "class_name": top_name,
                        "name": top_name,
                        "conf": top_conf,
                    }
                )
            if cls_topk:
                row_out["cls_topk"] = cls_topk
                row_out["cls_top3"] = cls_topk[:3]
                row_out["cls_top_n"] = len(cls_topk)
        out.append(row_out)
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


def box_ior(box1, box2) -> float:
    """两框 IoR（交集面积 / 较小框面积），与 ``model_detect.ior`` 一致。"""
    return float(
        ior(
            [box1[0], box1[1], box1[2], box1[3]],
            [box2[0], box2[1], box2[2], box2[3]],
        )
    )


def _row_box_area(r: dict) -> float:
    x1, y1, x2, y2 = float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _row_det_conf(r: dict) -> float:
    try:
        return float(r.get("conf", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def apply_ior_partial_box_filter(
    rows: list[dict],
    *,
    enabled: bool,
    ior_threshold: float,
    small_conf_threshold: float,
) -> None:
    """
    IoR 过滤虫体局部检出框（原地修改 ``rows``）。

    对任意一对未 ``filter`` 的框：若 IoR > ``ior_threshold``，且较小框检测置信度
    < ``small_conf_threshold``，则将较小框 ``cls_name`` 置为 ``other_part``（保留原类名于
    ``cls_name_before_ior_partial``）。已是 ``other`` 前缀的框不再处理。
    """
    if not enabled or not rows:
        return
    thr_ior = float(ior_threshold)
    thr_conf = float(small_conf_threshold)
    if thr_ior <= 0:
        return

    active = [i for i, r in enumerate(rows) if not r.get("filter")]
    for ai in range(len(active)):
        i = active[ai]
        ri = rows[i]
        if is_other_prefix_classname(str(ri.get("cls_name", "") or "")):
            continue
        bi = [ri["x1"], ri["y1"], ri["x2"], ri["y2"]]
        for aj in range(ai + 1, len(active)):
            j = active[aj]
            rj = rows[j]
            if is_other_prefix_classname(str(rj.get("cls_name", "") or "")):
                continue
            sc = box_ior(bi, [rj["x1"], rj["y1"], rj["x2"], rj["y2"]])
            if sc <= thr_ior:
                continue
            area_i = _row_box_area(ri)
            area_j = _row_box_area(rj)
            if area_i < area_j:
                small_idx = i
            elif area_j < area_i:
                small_idx = j
            elif _row_det_conf(ri) <= _row_det_conf(rj):
                small_idx = i
            else:
                small_idx = j
            rs = rows[small_idx]
            if is_other_prefix_classname(str(rs.get("cls_name", "") or "")):
                continue
            if _row_det_conf(rs) >= thr_conf:
                continue
            if str(rs.get("cls_name", "") or "") != "other_part":
                rs["cls_name_before_ior_partial"] = rs.get("cls_name")
            rs["cls_name"] = "other_part"
            rs["ior_partial_filter"] = True


def match_pred_gt(
    preds: list[dict],
    gts: list[dict],
    threshold: float,
    metric: str = "iou",
    *,
    pred_tier: list[int] | None = None,
    prefer_class_match: list[list[bool]] | None = None,
) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    """
    贪心匹配 pred 与 gt。

    :param metric: ``"iou"``（默认，交并比）或 ``"ior"``（交集/最小框面积，与 model_detect.ior 一致）。
    :param pred_tier: 可选，与 ``preds`` 等长的优先级档位；**数值越小越优先**匹配（如报出框 0、仅参与几何匹配的 filtered-other 1）。
    :param prefer_class_match: 可选 ``[pred_idx][gt_idx]``；为 True 的配对在同等 tier 下优先于类名不一致的配对。
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

    pairs: list[tuple[tuple[int, int, float], int, int, float]] = []
    for i, p in enumerate(preds):
        bp = _box_tuple(p)
        tier_i = int(pred_tier[i]) if pred_tier is not None else 0
        for j, g in enumerate(gts):
            bg = _box_tuple(g)
            score = _score(bp, bg)
            if score >= threshold:
                cm_penalty = (
                    0
                    if prefer_class_match is None or prefer_class_match[i][j]
                    else 1
                )
                pairs.append(((tier_i, cm_penalty, -score), i, j, score))
    pairs.sort(key=lambda x: x[0])
    used_p: set[int] = set()
    used_g: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for _sort_key, i, j, score in pairs:
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
    *,
    pred_tier: list[int] | None = None,
    prefer_class_match: list[list[bool]] | None = None,
) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    """向后兼容：等价于 ``match_pred_gt(..., metric='ior')``。"""
    return match_pred_gt(
        preds,
        gts,
        ior_threshold,
        metric="ior",
        pred_tier=pred_tier,
        prefer_class_match=prefer_class_match,
    )


def _visible_index(r: dict, results_visible: list[dict]) -> int | None:
    for idx, x in enumerate(results_visible):
        if x is r:
            return idx
    return None


# 跨平台中文字体候选（macOS / Linux-Ubuntu/Debian / Windows）。
_CJK_FONT_CANDIDATES = (
    # macOS
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    # Linux（Ubuntu/Debian 常见中文字体包安装路径）
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "/usr/share/fonts/truetype/arphic/ukai.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    # Windows
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
)

# 解析一次后缓存（含 fc-match 兜底，避免每次绘制重复探测）。
_CJK_FONT_PATH_CACHE: str | None = None
_CJK_FONT_PATH_RESOLVED = False


def resolve_cjk_font_path() -> str | None:
    """
    解析一个可用的中文字体文件路径，按优先级：
    1) 环境变量 ``INSECT_CJK_FONT`` 显式指定的字体文件；
    2) ``_CJK_FONT_CANDIDATES`` 常见系统路径；
    3) fontconfig ``fc-match``（Ubuntu 安装任意中文字体后即可命中，如 fonts-noto-cjk / wqy）。
    解析结果缓存；未找到返回 None（上层退化为 PIL 默认字体，中文会显示为方块）。
    """
    global _CJK_FONT_PATH_CACHE, _CJK_FONT_PATH_RESOLVED
    if _CJK_FONT_PATH_RESOLVED:
        return _CJK_FONT_PATH_CACHE
    _CJK_FONT_PATH_RESOLVED = True

    env_fp = os.environ.get("INSECT_CJK_FONT")
    for fp in ([env_fp] if env_fp else []) + list(_CJK_FONT_CANDIDATES):
        if fp and os.path.isfile(fp):
            _CJK_FONT_PATH_CACHE = fp
            logging.info("中文标签字体: %s", fp)
            return fp

    try:
        import subprocess

        out = subprocess.run(
            ["fc-match", "-f", "%{file}", ":lang=zh"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        fp = (out.stdout or "").strip()
        if fp and os.path.isfile(fp):
            _CJK_FONT_PATH_CACHE = fp
            logging.info("中文标签字体(fc-match): %s", fp)
            return fp
    except Exception:
        pass

    logging.warning(
        "未找到可用中文字体，标签中文将显示为方块；"
        "请安装中文字体（Ubuntu: sudo apt-get install -y fonts-noto-cjk && fc-cache -f）"
        "或设置环境变量 INSECT_CJK_FONT 指向字体文件。"
    )
    _CJK_FONT_PATH_CACHE = None
    return None


def _apply_matplotlib_cjk_font() -> None:
    """为 matplotlib 轴标签设置中文字体（复用 ``resolve_cjk_font_path``）。"""
    fp = resolve_cjk_font_path()
    if not fp:
        return
    try:
        from matplotlib import font_manager
        import matplotlib.pyplot as plt

        font_manager.fontManager.addfont(fp)
        family = font_manager.FontProperties(fname=fp).get_name()
        plt.rcParams["font.sans-serif"] = [family, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        logging.debug("matplotlib 中文字体设置失败: %s", fp, exc_info=True)


@contextmanager
def _suppress_matplotlib_missing_glyph_warnings():
    """屏蔽 DejaVu 缺中文字形时的 UserWarning（无中文字体时的兜底）。"""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Glyph .* missing from font",
            category=UserWarning,
        )
        yield


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

    # 跨平台中文字体解析（含环境变量 / fc-match 兜底）；失败退回默认字体。
    font = None
    fp = resolve_cjk_font_path()
    if fp:
        try:
            font = ImageFont.truetype(fp, font_size)
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


def iter_row_cls_topk(r: dict) -> list[tuple[str, float]]:
    """
    从检测行读取分类 top 列表（优先 ``cls_topk``，否则 ``cls_top3``），
    每项为 ``(class_name, conf)``；若无列表则退回当前 ``cls_name``/``cls_conf``。
    """
    raw = r.get("cls_topk")
    if raw is None:
        raw = r.get("cls_top3")
    out: list[tuple[str, float]] = []
    if isinstance(raw, list):
        for it in raw:
            if not isinstance(it, dict):
                continue
            nm = str(it.get("class_name", "") or "").strip()
            if not nm:
                continue
            try:
                cf = float(it.get("conf", 0) or 0.0)
            except Exception:
                cf = 0.0
            out.append((nm, cf))
    if not out:
        nm = str(r.get("cls_name", "") or "").strip()
        if nm:
            try:
                cf = float(r.get("cls_conf", r.get("conf", 0)) or 0.0)
            except Exception:
                cf = 0.0
            out.append((nm, cf))
    return out


def _append_cls_rank_extra_lines(
    line_specs: list[tuple[str, float]],
    r: dict,
    *,
    cls_output_top_n: int,
    base_font_scale: float,
) -> None:
    """
    在框标签主行之后追加第 2、第 3 类（另起行），最多画到第 3 名；受 ``cls_output_top_n`` 约束。
    """
    n = int(cls_output_top_n)
    if n <= 1:
        return
    ents = iter_row_cls_topk(r)
    max_rank_draw = min(3, n)
    fs = float(max(0.22, min(0.45, base_font_scale * 0.88)))
    for rank in range(2, max_rank_draw + 1):
        idx = rank - 1
        if idx >= len(ents):
            break
        name, conf = ents[idx]
        line_specs.append((f"#{rank} {name} {conf:.2f}", fs))


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
    hierarchy_type_index: ClsHierarchyIndex | None = None,
    cls_output_top_n: int = 1,
) -> Any:
    """
    主输出图：无源 xml 时与 PredictSize._draw_results 一致；
    有源 xml 时：正确绿框、多余上报蓝框、类型错误黄框、漏检粉框；other 与 filter 深灰框（绘制含过滤框需 return_full_final）。

    :param cls_output_top_n: 分类展示深度；为 1 仅主标签（top1）；≥2 时在主标签下最多再画第 2、第 3 类及置信度（换行）。
    """
    h_img, w_img = image.shape[:2]
    _top_n = max(1, int(cls_output_top_n))
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
        other 与过滤框均为灰色，便于与正常预测框区分。
        但额外在标签中显示 detect 模型的 class_name，便于对比 detect vs cls。
        """
        img_draw_local = img_bgr.copy()
        color_out = (0, 0, 255)
        color_other = (100, 100, 100)  # other：深灰（BGR）
        color_drop_debug = (100, 100, 100)

        for r in rows:
            x1, y1, x2, y2 = int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"])
            cls_name = r.get("cls_name", "unknown")
            det_name = r.get("class_name", "")
            cls_conf = float(r.get("cls_conf", 0.0) or 0.0)
            det_conf = float(r.get("conf", 0.0) or 0.0)

            if predict_debug:
                color = color_drop_debug if r.get("filter") else color_out
            else:
                color = color_other if is_other_prefix_classname(cls_name) else color_out

            cv2.rectangle(img_draw_local, (x1, y1), (x2, y2), color, int(params["rect_thk"]))

            if is_other_prefix_classname(cls_name):
                if mode == "minimal":
                    label = f"{cls_name}-{det_conf:.2f}-{cls_conf:.2f}"
                else:
                    label = str(cls_name)
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

            # label 为空则跳过文字绘制（正常不会出现）
            if not label:
                continue

            font_scale = float(params["font_scale"])
            line_specs: list[tuple[str, float]] = [(label, font_scale)]
            _append_cls_rank_extra_lines(
                line_specs,
                r,
                cls_output_top_n=_top_n,
                base_font_scale=font_scale,
            )
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

    color_gray = (100, 100, 100)  # 过滤/未参与匹配：深灰（BGR）
    color_ok = (0, 255, 0)  # 正确：绿色
    color_fp = (255, 128, 0)  # 多余上报：蓝色（BGR，约 #0080FF）
    color_fn = (255, 0, 255)  # 漏报：粉色（BGR）
    color_other = color_gray  # other：灰色（BGR）
    color_cls_err = (0, 255, 255)  # 类型错误：黄色（BGR）
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
            if is_class_match(pred_cls, gt_name, merge, hierarchy_type_index):
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
        elif is_other_prefix_classname(str(r.get("cls_name", "") or "")):
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
                if is_class_match(pred_cls, gt_name, merge, hierarchy_type_index):
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
        if is_other_prefix_classname(cls_name):
            if mode == "minimal":
                label = f"{cls_name}-{det_conf:.2f}-{cls_conf:.2f}"
            else:
                label = str(cls_name)
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
            if not is_class_match(cls_name, gt_name, merge, hierarchy_type_index):
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

        # label 为空则跳过文字绘制（正常不会出现）
        if not label:
            continue

        font_scale = float(params["font_scale"])
        line_specs: list[tuple[str, float]] = [(label, font_scale)]
        _append_cls_rank_extra_lines(
            line_specs,
            r,
            cls_output_top_n=_top_n,
            base_font_scale=font_scale,
        )
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

    # 漏检：依据源 xml 标注画粉框
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
        f"多余上报(蓝): {n_fp}",
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


def outputs_exist_for_incremental_skip(result_output_dir: str, rel_path: Path) -> bool:
    """
    增量跳过：输出目录下已存在与源图同名的结果图及对应 VOC xml（stem 一致）则视为已完成。
    """
    img_p = os.path.join(result_output_dir, rel_path.name)
    xml_p = os.path.join(result_output_dir, rel_path.stem + ".xml")
    if not (os.path.isfile(img_p) and os.path.isfile(xml_p)):
        return False
    try:
        if os.path.getsize(img_p) <= 0 or os.path.getsize(xml_p) <= 0:
            return False
    except OSError:
        return False
    return True


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
    merge: dict[str, list[str]] | None = None,
    hierarchy_type_index: ClsHierarchyIndex | None = None,
    cls_output_top_n: int = 1,
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
                merge=merge,
                hierarchy_type_index=hierarchy_type_index,
                cls_output_top_n=cls_output_top_n,
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
                merge=merge,
                hierarchy_type_index=hierarchy_type_index,
                cls_output_top_n=cls_output_top_n,
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



# --------------------------------------------------------------------------- #
#  标准评估导出（原 predict_size_validate_dual.py，供 predict_all / predict_seg_validate）
# --------------------------------------------------------------------------- #
_STAT_BY_CLS_KEYS = ("gt", "pred", "tp", "fp", "fn", "cls_err")


def build_eval_focus_set(
    names: list[str] | set[str] | None,
    *,
    merge: dict[str, list[str]] | None = None,
    label_alias_map: dict[str, str] | None = None,
) -> frozenset[str]:
    """将关注类名列表展开为匹配用集合（含原始拼音与 normalize 后的 canonical 名）。"""
    out: set[str] = set()
    for n in names or []:
        raw = str(n or "").strip()
        if not raw:
            continue
        out.add(raw)
        canon = normalize_class_name(raw, merge, label_alias_map=label_alias_map)
        if canon:
            out.add(canon)
    return frozenset(x for x in out if x)


def sum_stat_by_cls_focus(
    stat_by_cls: dict[str, dict[str, int]],
    focus: frozenset[str],
) -> dict[str, int]:
    """汇总 focus 内各类的 tp/fp/fn/cls_err（用于分组报出率/正确率等）。"""
    out = {"tp": 0, "fp": 0, "fn": 0, "cls_err": 0, "geom_pairs": 0}
    if not stat_by_cls:
        return out
    for cls_name, s in stat_by_cls.items():
        if focus and str(cls_name) not in focus:
            continue
        out["tp"] += int(s.get("tp", 0))
        out["fp"] += int(s.get("fp", 0))
        out["fn"] += int(s.get("fn", 0))
        out["cls_err"] += int(s.get("cls_err", 0))
    return out


def _print_overall_stat_summary(title: str, s: dict[str, int]) -> None:
    tp = int(s.get("tp", 0))
    fp = int(s.get("fp", 0))
    fn = int(s.get("fn", 0))
    ce = int(s.get("cls_err", 0))
    geom = int(s.get("geom_pairs", 0))
    denom_gt = float(tp + ce + fn)
    denom_pred = float(tp + fp)
    report_rate = (float(tp + ce) / denom_gt) if denom_gt > 0 else 0.0
    acc_rate = (float(tp) / denom_gt) if denom_gt > 0 else 0.0
    err_rate = (float(fp) / denom_pred) if denom_pred > 0 else 0.0
    miss_fn_rate = (float(fn) / denom_gt) if denom_gt > 0 else 0.0
    cls_err_rate = (float(ce) / denom_gt) if denom_gt > 0 else 0.0
    recall_gap = (float(fn + ce) / denom_gt) if denom_gt > 0 else 0.0
    total_dev = max(recall_gap, err_rate)
    print(
        f"{title}: tp={tp} fp={fp} fn={fn} cls_err={ce} geom_pairs={geom} | "
        f"报出率={report_rate * 100:.2f}% 正确率={acc_rate * 100:.2f}% 错误率={err_rate * 100:.2f}% "
        f"漏检率(仅几何无框)={miss_fn_rate * 100:.2f}% 类错率={cls_err_rate * 100:.2f}% "
        f"召回缺口={recall_gap * 100:.2f}% 总偏差率=max(召回缺口,错误率)={total_dev * 100:.2f}% "
        f"(报出率=(TP+类型错)/标注 正确率=TP/标注)"
    )


def _collect_eval_class_display_from_alg_node(node: Any, out: dict[str, str]) -> None:
    """递归收集 class key / infer_name → cn_name（评估统计展示用）。"""
    if isinstance(node, dict):
        out_table = node.get("out")
        if isinstance(out_table, dict):
            for cls_key, entry in out_table.items():
                if not isinstance(entry, dict):
                    continue
                cn = str(entry.get("cn_name") or "").strip()
                if cn:
                    route_key = str(cls_key).strip()
                    infer_key = str(entry.get("infer_name") or "").strip()
                    if route_key and route_key != "*":
                        out[route_key] = cn
                    if infer_key and infer_key != "*" and infer_key != route_key:
                        out.setdefault(infer_key, cn)
                _collect_eval_class_display_from_alg_node(entry, out)
        for key in ("models", "cls"):
            sub = node.get(key)
            if isinstance(sub, dict):
                _collect_eval_class_display_from_alg_node(sub, out)
        for k, v in node.items():
            if k in ("out", "models", "cls"):
                continue
            if isinstance(v, dict):
                _collect_eval_class_display_from_alg_node(v, out)
    elif isinstance(node, list):
        for item in node:
            _collect_eval_class_display_from_alg_node(item, out)


def build_eval_class_display_index(
    *,
    alg_config: dict[str, Any] | None = None,
) -> dict[str, str]:
    """canonical 类键 → 中文展示名（无配置时回退拼音/原键）。"""
    index: dict[str, str] = {
        "other": "其他",
        "unknown": "未知",
        "insect": "昆虫",
        "insect_small": "小虫",
    }
    try:
        from script.config.insect_info import load_json_catalog  # noqa: PLC0415
    except ImportError:
        load_json_catalog = None  # type: ignore[misc, assignment]
    if load_json_catalog is not None:
        try:
            for pinyin, rec in load_json_catalog().items():
                zh = str(getattr(rec, "name_zh", "") or "").strip()
                if zh:
                    index.setdefault(str(pinyin), zh)
        except Exception as e:
            logging.warning("构建评估展示名索引：加载 insect_info 失败: %s", e)
    if isinstance(alg_config, dict):
        _collect_eval_class_display_from_alg_node(alg_config.get("models"), index)
    return index


def resolve_eval_class_display_name(
    canonical_key: str,
    *,
    display_index: dict[str, str] | None = None,
) -> str:
    """评估统计展示名：优先中文；已是中文则原样；无映射时保留拼音/原键。"""
    key = str(canonical_key or "").strip()
    if not key:
        return ""
    if _CJK_RE.search(key):
        return key
    if display_index:
        hit = display_index.get(key)
        if hit:
            return hit
    try:
        from script.ls_classification_ingest import cls_name_to_zh  # noqa: PLC0415

        zh = cls_name_to_zh(key)
        if zh and zh != key:
            return zh
    except ImportError:
        pass
    return key


def merge_stat_by_cls(
    stat_by_cls: dict[str, dict[str, int]],
    *,
    merge: dict[str, list[str]] | None = None,
    label_alias_map: dict[str, str] | None = None,
    tier_equivalence: ClassTierEquivalence | None = None,
) -> dict[str, dict[str, int]]:
    """按 canonical 类键合并按类统计，避免中文标注与拼音键重复计数。"""
    merged: dict[str, dict[str, int]] = {}
    for raw_key, counts in (stat_by_cls or {}).items():
        canon = normalize_class_name(
            str(raw_key or ""),
            merge,
            label_alias_map=label_alias_map,
        )
        if not canon:
            continue
        if tier_equivalence is not None:
            canon = tier_equivalence.canonical_stat_name(canon)
        bucket = merged.setdefault(
            canon, {k: 0 for k in _STAT_BY_CLS_KEYS}
        )
        for k in _STAT_BY_CLS_KEYS:
            bucket[k] = int(bucket.get(k, 0)) + int(counts.get(k, 0))
    return merged


def _disp_w(s: str) -> int:
    # 终端显示宽度：中日韩宽字符按 2 计
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
    sort_by_acc: bool = True,
    class_display_index: dict[str, str] | None = None,
    focus: frozenset[str] | None = None,
) -> None:
    if not stat_by_cls:
        return
    focus_set = frozenset(str(x).strip() for x in (focus or []) if str(x).strip())
    headers = [
        "类别",
        "标注数",
        "预测数",
        "正确TP",
        "报出率",
        "类型错",
        "正确率",
        "漏检FN",
        "漏检率",
        "类错率",
        "多检FP",
        "误报率",
        "总偏差率",
    ]

    # sort keys: total_dev_rate asc -> report_rate desc -> acc_rate desc
    rows_with_sort: list[tuple[str, float, float, float, list[str]]] = []
    total = {"gt": 0, "pred": 0, "tp": 0, "cls_err": 0, "fn": 0, "fp": 0}
    for cls_name in stat_by_cls.keys():
        if focus_set and str(cls_name) not in focus_set:
            continue
        s = stat_by_cls[cls_name]
        gt_n = int(s.get("gt", 0))
        pred_n = int(s.get("pred", 0))
        tp_n = int(s.get("tp", 0))
        ce_n = int(s.get("cls_err", 0))
        fn_n = int(s.get("fn", 0))
        fp_n = int(s.get("fp", 0))

        denom_gt = float(gt_n)
        denom_pred = float(pred_n)
        # 报出率：有预测框的标注占比（TP+类型错）/标注数
        report_rate = (float(tp_n + ce_n) / denom_gt) if denom_gt > 0 else 0.0
        # 正确率：完全匹配 TP / 标注数
        acc_rate = (float(tp_n) / denom_gt) if denom_gt > 0 else 0.0
        fp_rate = (float(fp_n) / denom_pred) if denom_pred > 0 else 0.0
        # 漏检率：仅「几何上无任何匹配框」的 GT 占比（与类型错互斥，不去重相加）
        miss_fn_rate = (float(fn_n) / denom_gt) if denom_gt > 0 else 0.0
        # 类错率：有框但分错（相对标注数）
        cls_err_rate = (float(ce_n) / denom_gt) if denom_gt > 0 else 0.0
        # 召回缺口(相对标注) = 漏检 + 类型错，与误报率分母不同；总偏差率取 max 避免简单相加超过 100%
        recall_gap = (float(fn_n + ce_n) / denom_gt) if denom_gt > 0 else 0.0
        total_dev_rate = max(recall_gap, fp_rate)

        disp_name = resolve_eval_class_display_name(
            cls_name, display_index=class_display_index
        )
        row = [
            disp_name,
            str(gt_n),
            str(pred_n),
            str(tp_n),
            f"{report_rate*100:.2f}%",
            str(ce_n),
            f"{acc_rate*100:.2f}%",
            str(fn_n),
            f"{miss_fn_rate*100:.2f}%",
            f"{cls_err_rate*100:.2f}%",
            str(fp_n),
            f"{fp_rate*100:.2f}%",
            f"{total_dev_rate*100:.2f}%",
        ]
        rows_with_sort.append(
            (str(cls_name), float(total_dev_rate), float(report_rate), float(acc_rate), row)
        )
        total["gt"] += gt_n
        total["pred"] += pred_n
        total["tp"] += tp_n
        total["cls_err"] += ce_n
        total["fn"] += fn_n
        total["fp"] += fp_n

    if sort_by_acc:
        rows_with_sort.sort(
            key=lambda x: (
                x[1],  # total_dev_rate asc
                -x[2],  # report_rate desc
                -x[3],  # acc_rate desc
                -int(stat_by_cls.get(x[0], {}).get("gt", 0)),  # tie-breaker: support desc
                x[0],
            )
        )
    else:
        rows_with_sort.sort(key=lambda x: x[0])
    rows = [r for _cls, _dev, _rep, _acc, r in rows_with_sort]

    tg = int(total["gt"])
    tpr = int(total["pred"])
    ttp = int(total["tp"])
    tce = int(total["cls_err"])
    tfn = int(total["fn"])
    tfp = int(total["fp"])
    dgt = float(tg)
    dpr = float(tpr)
    sum_miss_fn = (float(tfn) / dgt) if dgt > 0 else 0.0
    sum_cls_err_r = (float(tce) / dgt) if dgt > 0 else 0.0
    sum_recall_gap = (float(tfn + tce) / dgt) if dgt > 0 else 0.0
    sum_report = (float(ttp + tce) / dgt) if dgt > 0 else 0.0
    sum_acc = (float(ttp) / dgt) if dgt > 0 else 0.0
    sum_fp_r = (float(tfp) / dpr) if dpr > 0 else 0.0
    sum_total_dev = max(sum_recall_gap, sum_fp_r)

    all_lines = [headers] + rows + [
        [
            "合计",
            str(tg),
            str(tpr),
            str(ttp),
            f"{sum_report*100:.2f}%",
            str(tce),
            f"{sum_acc*100:.2f}%",
            str(tfn),
            f"{sum_miss_fn*100:.2f}%",
            f"{sum_cls_err_r*100:.2f}%",
            str(tfp),
            f"{sum_fp_r*100:.2f}%",
            f"{sum_total_dev*100:.2f}%",
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

    sort_hint = (
        "总偏差率升序(max(召回缺口,误报率))，其次报出率降序，再次正确率降序；"
        "报出率=(TP+类型错)/标注，正确率=TP/标注，漏检率=FN/标注，类错率=类型错/标注，与多检误报分列"
        if sort_by_acc
        else "标注类别名升序"
    )
    print(f"{title}（按{sort_hint}）:")
    print(_fmt_line(headers))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(_fmt_line([str(x) for x in r]))
    print("-+-".join("-" * w for w in widths))
    print(_fmt_line(all_lines[-1]))


def _row_output_suppressed(
    r: dict,
    suppress: frozenset[str] | set[str],
    class_merge_to_groups: dict[str, list[str]] | None,
) -> bool:
    """
    输出暂隐：cls_name 与配置集合比对时同时支持「预测原始名」与 normalize_class_name 后的组名。
    """
    if not suppress:
        return False
    raw = str(r.get("cls_name", "") or "").strip()
    if not raw:
        return False
    if raw in suppress:
        return True
    norm = normalize_class_name(raw, class_merge_to_groups)
    return bool(norm) and norm in suppress


def _apply_output_suppress(
    results: list[dict],
    all_rows: list[dict],
    suppress: frozenset[str],
    class_merge_to_groups: dict[str, list[str]] | None,
) -> tuple[list[dict], list[dict]]:
    if not suppress:
        return results, all_rows
    return (
        [r for r in results if not _row_output_suppressed(r, suppress, class_merge_to_groups)],
        [r for r in all_rows if not _row_output_suppressed(r, suppress, class_merge_to_groups)],
    )


def _apply_output_suppress_combined(
    results: list[dict],
    all_rows: list[dict],
    suppress: frozenset[str],
    merge_small: dict[str, list[str]] | None,
    merge_large: dict[str, list[str]] | None,
) -> tuple[list[dict], list[dict]]:
    if not suppress:
        return results, all_rows

    def _merge_for(r: dict) -> dict[str, list[str]] | None:
        return merge_small if r.get("model_src") == "small" else merge_large

    return (
        [r for r in results if not _row_output_suppressed(r, suppress, _merge_for(r))],
        [r for r in all_rows if not _row_output_suppressed(r, suppress, _merge_for(r))],
    )


def _apply_focus_filter(
    results: list[dict],
    all_rows: list[dict],
    focus: frozenset[str],
    class_merge_to_groups: dict[str, list[str]] | None,
) -> tuple[list[dict], list[dict]]:
    """
    只保留关注列表中的虫子（用于“统计/分析/画图/写xml”全链路）。

    focus 内同时支持：
    - 原始类别名（如 xml/pred 里的 name/cls_name）
    - normalize_class_name 后的归一名（与 CLASS_MERGE_TO_GROUPS_* 的组名一致）
    """
    if not focus:
        return results, all_rows

    def _keep_name(raw_name: str) -> bool:
        raw = str(raw_name or "").strip()
        if not raw:
            return False
        if raw in focus:
            return True
        norm = normalize_class_name(raw, class_merge_to_groups)
        return bool(norm) and norm in focus

    def _keep_row(r: dict) -> bool:
        return _keep_name(str(r.get("cls_name", "") or ""))

    return ([r for r in results if _keep_row(r)], [r for r in all_rows if _keep_row(r)])


def _apply_focus_filter_combined(
    results: list[dict],
    all_rows: list[dict],
    focus: frozenset[str],
    merge_small: dict[str, list[str]] | None,
    merge_large: dict[str, list[str]] | None,
) -> tuple[list[dict], list[dict]]:
    if not focus:
        return results, all_rows

    def _merge_for(r: dict) -> dict[str, list[str]] | None:
        return merge_small if r.get("model_src") == "small" else merge_large

    def _keep_row(r: dict) -> bool:
        raw = str(r.get("cls_name", "") or "").strip()
        if not raw:
            return False
        if raw in focus:
            return True
        norm = normalize_class_name(raw, _merge_for(r))
        return bool(norm) and norm in focus

    return ([r for r in results if _keep_row(r)], [r for r in all_rows if _keep_row(r)])


# 混淆矩阵：行=真实类别(GT)，列=预测类别；与 YOLO val 常见约定一致地增加「无检测」「多余预测」虚拟类
_STD_EVAL_BG_COL = "__no_pred__"  # FN：该 GT 未与任一预测框匹配
_STD_EVAL_BG_ROW = "__extra_pred__"  # FP：该预测未与任一 GT 匹配


def _std_eval_collect_confusion(
    cm: dict[tuple[str, str], int],
    preds: list[dict],
    gts: list[dict] | None,
    matches: list[tuple[int, int, float]] | None,
    matched_p: set[int] | None,
    class_merge_to_groups: dict[str, list[str]] | None,
    hierarchy_type_index: ClsHierarchyIndex | None = None,
    *,
    label_alias_map: dict[str, str] | None = None,
    fuzzy_only_wildcard: bool = False,
    tier_equivalence: ClassTierEquivalence | None = None,
) -> None:
    """
    按几何匹配结果更新 (gt_label, pred_label) 计数；未匹配 GT 记入 (gt, __no_pred__)，未匹配 pred 记入 (__extra_pred__, pred)。
    ``other`` 类不参与混淆矩阵与 per-class 指标（与 _eval_one 一致）。

    - 标签先做 ``normalize_class_name``；若提供 ``hierarchy_type_index``，再 **归并到 cls_merge 顶层父类** 统计，
      子类/叶目录名不再单独占行列（与「父子等价、非错误」一致）。
    - ``is_class_match``（含 L1 父子、``insect`` 通配等）为真时，将 pred 列对齐为 gt 标签，落在对角格。
    """
    if gts is None:
        return
    mlist = list(matches or [])
    m_p = matched_p if matched_p is not None else set()
    if not gts:
        for p in preds:
            if is_metric_ignored_other(str(p.get("cls_name", "") or ""), class_merge_to_groups):
                continue
            pn = confusion_matrix_stat_label(
                eval_pred_row_display_class(p),
                class_merge_to_groups,
                hierarchy_type_index,
                label_alias_map=label_alias_map,
            )
            cm[(_STD_EVAL_BG_ROW, pn)] += 1
        return
    matched_g = {int(j) for _, j, _ in mlist}
    for i, j, _ in mlist:
        if is_metric_ignored_other(str(preds[int(i)].get("cls_name", "") or ""), class_merge_to_groups):
            continue
        if is_metric_ignored_other(str(gts[int(j)].get("name", "") or ""), class_merge_to_groups):
            continue
        pred_eval = str(preds[int(i)].get("cls_name", "") or "")
        pred_display = eval_pred_row_display_class(preds[int(i)])
        gt_raw = str(gts[int(j)].get("name", "") or "")
        gn = confusion_matrix_stat_label(
            gt_raw,
            class_merge_to_groups,
            hierarchy_type_index,
            label_alias_map=label_alias_map,
        )
        pn = confusion_matrix_stat_label(
            pred_display,
            class_merge_to_groups,
            hierarchy_type_index,
            label_alias_map=label_alias_map,
        )
        if is_class_match(
            pred_eval,
            gt_raw,
            class_merge_to_groups,
            hierarchy_type_index,
            label_alias_map=label_alias_map,
            fuzzy_only_wildcard=fuzzy_only_wildcard,
            tier_equivalence=tier_equivalence,
        ):
            pn = gn
        cm[(gn, pn)] += 1
    for pi in range(len(preds)):
        if pi not in m_p:
            if is_metric_ignored_other(str(preds[pi].get("cls_name", "") or ""), class_merge_to_groups):
                continue
            pn = confusion_matrix_stat_label(
                eval_pred_row_display_class(preds[pi]),
                class_merge_to_groups,
                hierarchy_type_index,
                label_alias_map=label_alias_map,
            )
            cm[(_STD_EVAL_BG_ROW, pn)] += 1
    for gj in range(len(gts)):
        if gj not in matched_g:
            if is_metric_ignored_other(str(gts[gj].get("name", "") or ""), class_merge_to_groups):
                continue
            gn = confusion_matrix_stat_label(
                str(gts[gj].get("name", "") or ""),
                class_merge_to_groups,
                hierarchy_type_index,
                label_alias_map=label_alias_map,
            )
            cm[(gn, _STD_EVAL_BG_COL)] += 1


def _std_eval_collect_confusion_filename_gt(
    cm: dict[tuple[str, str], int],
    preds: list[dict],
    gt_name: str,
    class_merge_to_groups: dict[str, list[str]] | None,
    hierarchy_type_index: ClsHierarchyIndex | None = None,
    *,
    label_alias_map: dict[str, str] | None = None,
    fuzzy_only_wildcard: bool = False,
    tier_equivalence: ClassTierEquivalence | None = None,
) -> None:
    """
    文件名 GT 混淆矩阵：每个非 other 报出框与图级 GT 类名比对（无几何匹配）。
    无报出框时记入 (gt, __no_pred__)。
    """
    gt_raw = str(gt_name or "").strip()
    if not gt_raw:
        return
    if is_metric_ignored_other(gt_raw, class_merge_to_groups):
        return
    gn = confusion_matrix_stat_label(
        gt_raw,
        class_merge_to_groups,
        hierarchy_type_index,
        label_alias_map=label_alias_map,
    )
    eval_preds = [
        p
        for p in preds
        if not is_metric_ignored_other(
            str(p.get("cls_name", "") or ""), class_merge_to_groups
        )
    ]
    if not eval_preds:
        cm[(gn, _STD_EVAL_BG_COL)] += 1
        return
    for p in eval_preds:
        pred_eval = str(p.get("cls_name", "") or "")
        pred_raw = eval_pred_row_display_class(p)
        pn = confusion_matrix_stat_label(
            pred_raw,
            class_merge_to_groups,
            hierarchy_type_index,
            label_alias_map=label_alias_map,
        )
        if is_class_match(
            pred_eval,
            gt_raw,
            class_merge_to_groups,
            hierarchy_type_index,
            label_alias_map=label_alias_map,
            fuzzy_only_wildcard=fuzzy_only_wildcard,
            tier_equivalence=tier_equivalence,
        ):
            pn = gn
        cm[(gn, pn)] += 1


def _save_type_confusion_crops_for_filename_gt(
    img_bgr,
    rel_path: Path,
    preds: list[dict],
    gt_name: str,
    *,
    class_merge_to_groups: dict[str, list[str]] | None,
    hierarchy_type_index: ClsHierarchyIndex | None,
    eval_root: str,
    branch: str,
    label_alias_map: dict[str, str] | None = None,
    fuzzy_only_wildcard: bool = False,
) -> None:
    """文件名 GT：对类错的报出框按预测框切图落盘。"""
    gt_raw = str(gt_name or "").strip()
    if not gt_raw or is_metric_ignored_other(gt_raw, class_merge_to_groups):
        return
    base_gt = os.path.join(eval_root, branch, "type_confusion_crops")
    base_pred = os.path.join(eval_root, branch, "type_confusion_crops_by_pred")
    gn = confusion_matrix_stat_label(
        gt_raw,
        class_merge_to_groups,
        hierarchy_type_index,
        label_alias_map=label_alias_map,
    )
    for pi, p in enumerate(preds):
        if is_metric_ignored_other(
            str(p.get("cls_name", "") or ""), class_merge_to_groups
        ):
            continue
        pred_eval = str(p.get("cls_name", "") or "")
        pred_display = eval_pred_row_display_class(p)
        if is_class_match(
            pred_eval,
            gt_raw,
            class_merge_to_groups,
            hierarchy_type_index,
            label_alias_map=label_alias_map,
            fuzzy_only_wildcard=fuzzy_only_wildcard,
        ):
            continue
        crop = _safe_crop_xyxy(
            img_bgr,
            int(p.get("x1", 0)),
            int(p.get("y1", 0)),
            int(p.get("x2", 0)),
            int(p.get("y2", 0)),
        )
        if crop is None:
            continue
        pn = confusion_matrix_stat_label(
            pred_display,
            class_merge_to_groups,
            hierarchy_type_index,
            label_alias_map=label_alias_map,
        )
        det_conf = float(p.get("conf", 0.0) or 0.0)
        cls_conf = float(p.get("cls_conf", det_conf) or det_conf)
        stem = f"{rel_path.stem}__pi{pi:03d}__det{det_conf:.3f}__cls{cls_conf:.3f}.jpg"
        _write_type_confusion_crop_pair(
            crop,
            base_gt=base_gt,
            base_pred=base_pred,
            gn=gn,
            pn=pn,
            stem=stem,
        )


_FS_UNSAFE_SEGMENT_CHARS = r'\/:*?"<>|' + "\x00"


def _fs_safe_segment(name: str, *, max_len: int = 160) -> str:
    """目录名片段（与 predict_cls_validate_from_xml 一致）。"""
    s = str(name or "").strip()
    for ch in _FS_UNSAFE_SEGMENT_CHARS:
        s = s.replace(ch, "_")
    s = s.strip(" .")
    if not s:
        s = "_unknown"
    return s[:max_len]


def _safe_crop_xyxy(img_bgr, x1: int, y1: int, x2: int, y2: int):
    """从整图裁 GT 框，越界裁剪（与 predict_cls_validate_from_xml 一致）。"""
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


def _type_confusion_crop_stem(rel_path: Path, pi: int, gj: int, pred: dict) -> str:
    det_conf = float(pred.get("conf", 0.0) or 0.0)
    cls_conf = float(pred.get("cls_conf", det_conf) or 0.0)
    return f"{rel_path.stem}__pi{pi:03d}__gj{gj:03d}__det{det_conf:.3f}__cls{cls_conf:.3f}.jpg"


_TYPE_CONFUSION_MISS_PRED = "__miss__"


def _type_miss_crop_stem(rel_path: Path, gj: int) -> str:
    return f"{rel_path.stem}__gj{gj:03d}__miss.jpg"


def _write_type_confusion_crop_pair(
    crop,
    *,
    base_gt: str,
    base_pred: str,
    gn: str,
    pn: str,
    stem: str,
) -> None:
    out_sub_gt_first = os.path.join(
        base_gt,
        f"gt={_fs_safe_segment(gn)}",
        f"pred={_fs_safe_segment(pn)}",
    )
    os.makedirs(out_sub_gt_first, exist_ok=True)
    out_mis = os.path.join(out_sub_gt_first, stem)
    if not cv2.imwrite(out_mis, crop):
        logging.warning("未能写入类型混淆切图: %s", out_mis)
    out_sub_pred_first = os.path.join(
        base_pred,
        f"pred={_fs_safe_segment(pn)}",
        f"gt={_fs_safe_segment(gn)}",
    )
    os.makedirs(out_sub_pred_first, exist_ok=True)
    out_mis2 = os.path.join(out_sub_pred_first, stem)
    if not cv2.imwrite(out_mis2, crop):
        logging.warning("未能写入类型混淆切图: %s", out_mis2)


def _save_type_confusion_crops_for_branch(
    img_bgr,
    rel_path: Path,
    preds: list[dict],
    gts: list[dict],
    matches: list[tuple[int, int, float]] | None,
    *,
    class_merge_to_groups: dict[str, list[str]] | None,
    hierarchy_type_index: ClsHierarchyIndex | None,
    eval_root: str,
    branch: str,
    label_alias_map: dict[str, str] | None = None,
    fuzzy_only_wildcard: bool = False,
) -> None:
    """
    按 GT 框切图落盘（索引方式同 predict_cls_validate_from_xml）：
    - 几何已匹配但类型不一致：type_confusion_crops/gt=.../pred=...
    - 几何漏检（GT 无匹配预测框）：.../pred=__miss__
    目录名使用混淆矩阵同款标签（含 cls_merge 顶层归并）。
    """
    if not gts:
        return
    base_gt = os.path.join(eval_root, branch, "type_confusion_crops")
    base_pred = os.path.join(eval_root, branch, "type_confusion_crops_by_pred")
    mlist = list(matches or [])
    matched_g = {int(j) for _, j, _ in mlist}
    for pi, gj, _sc in mlist:
        pi, gj = int(pi), int(gj)
        if pi < 0 or pi >= len(preds) or gj < 0 or gj >= len(gts):
            continue
        if is_metric_ignored_other(str(preds[pi].get("cls_name", "") or ""), class_merge_to_groups):
            continue
        if is_metric_ignored_other(str(gts[gj].get("name", "") or ""), class_merge_to_groups):
            continue
        pred_eval = str(preds[pi].get("cls_name", "") or "")
        pred_display = eval_pred_row_display_class(preds[pi])
        gt_raw = str(gts[gj].get("name", "") or "")
        if is_class_match(
            pred_eval,
            gt_raw,
            class_merge_to_groups,
            hierarchy_type_index,
            label_alias_map=label_alias_map,
            fuzzy_only_wildcard=fuzzy_only_wildcard,
        ):
            continue
        g = gts[gj]
        crop = _safe_crop_xyxy(img_bgr, int(g["x1"]), int(g["y1"]), int(g["x2"]), int(g["y2"]))
        if crop is None:
            continue
        gn = confusion_matrix_stat_label(
            gt_raw,
            class_merge_to_groups,
            hierarchy_type_index,
            label_alias_map=label_alias_map,
        )
        pn = confusion_matrix_stat_label(
            pred_display,
            class_merge_to_groups,
            hierarchy_type_index,
            label_alias_map=label_alias_map,
        )
        stem = _type_confusion_crop_stem(rel_path, pi, gj, preds[pi])
        _write_type_confusion_crop_pair(
            crop,
            base_gt=base_gt,
            base_pred=base_pred,
            gn=gn,
            pn=pn,
            stem=stem,
        )

    for gj in range(len(gts)):
        if gj in matched_g:
            continue
        if is_metric_ignored_other(str(gts[gj].get("name", "") or ""), class_merge_to_groups):
            continue
        g = gts[gj]
        crop = _safe_crop_xyxy(img_bgr, int(g["x1"]), int(g["y1"]), int(g["x2"]), int(g["y2"]))
        if crop is None:
            continue
        gt_raw = str(gts[gj].get("name", "") or "")
        gn = confusion_matrix_stat_label(
            gt_raw,
            class_merge_to_groups,
            hierarchy_type_index,
            label_alias_map=label_alias_map,
        )
        stem = _type_miss_crop_stem(rel_path, gj)
        _write_type_confusion_crop_pair(
            crop,
            base_gt=base_gt,
            base_pred=base_pred,
            gn=gn,
            pn=_TYPE_CONFUSION_MISS_PRED,
            stem=stem,
        )


def _std_eval_labels_from_cm(cm: dict[tuple[str, str], int]) -> list[str]:
    rows = {a for (a, _) in cm.keys()}
    cols = {b for (_, b) in cm.keys()}
    ordered = [x for x in sorted(rows | cols) if x not in (_STD_EVAL_BG_ROW, _STD_EVAL_BG_COL)]
    # 虚拟行列放末尾，便于阅读
    if _STD_EVAL_BG_ROW in rows or _STD_EVAL_BG_ROW in cols:
        ordered.append(_STD_EVAL_BG_ROW)
    if _STD_EVAL_BG_COL in rows or _STD_EVAL_BG_COL in cols:
        ordered.append(_STD_EVAL_BG_COL)
    return ordered


def _std_eval_matrix_and_metrics(
    cm: dict[tuple[str, str], int],
    *,
    labels: list[str],
) -> tuple[list[list[int]], list[dict], dict[str, float]]:
    """返回 count 矩阵、每类指标行、整体 micro 指标。"""
    n = len(labels)
    idx = {lab: i for i, lab in enumerate(labels)}
    mat = [[0] * n for _ in range(n)]
    for (a, b), c in cm.items():
        if a not in idx or b not in idx:
            continue
        mat[idx[a]][idx[b]] += int(c)

    per_rows: list[dict] = []
    tp_sum = fp_micro = fn_micro = 0
    for lab in labels:
        if lab in (_STD_EVAL_BG_ROW, _STD_EVAL_BG_COL):
            continue
        i = idx[lab]
        tp = mat[i][i]
        fn = sum(mat[i][k] for k in range(n) if k != i)
        fp = sum(mat[k][i] for k in range(n) if k != i)
        support = tp + fn
        pred_pos = tp + fp
        rec = (tp / support) if support > 0 else 0.0
        prec = (tp / pred_pos) if pred_pos > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_rows.append(
            {
                "class": lab,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "support_gt": support,
                "support_pred": pred_pos,
                "precision": round(prec, 6),
                "recall": round(rec, 6),
                "f1": round(f1, 6),
            }
        )
        tp_sum += tp
        fp_micro += fp
        fn_micro += fn

    denom_pr = tp_sum + fp_micro
    denom_re = tp_sum + fn_micro
    overall = {
        "micro_precision": (tp_sum / denom_pr) if denom_pr > 0 else 0.0,
        "micro_recall": (tp_sum / denom_re) if denom_re > 0 else 0.0,
        "macro_precision": (
            sum(r["precision"] for r in per_rows) / len(per_rows) if per_rows else 0.0
        ),
        "macro_recall": (sum(r["recall"] for r in per_rows) / len(per_rows) if per_rows else 0.0),
        "macro_f1": (sum(r["f1"] for r in per_rows) / len(per_rows) if per_rows else 0.0),
    }
    if overall["micro_precision"] + overall["micro_recall"] > 0:
        p, r = overall["micro_precision"], overall["micro_recall"]
        overall["micro_f1"] = 2 * p * r / (p + r)
    else:
        overall["micro_f1"] = 0.0
    for k in ("micro_precision", "micro_recall", "micro_f1", "macro_precision", "macro_recall", "macro_f1"):
        overall[k] = round(float(overall[k]), 6)

    return mat, per_rows, overall


def _cm_blues_color(t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, float(t)))
    return (
        int(255 * (1.0 - t * 0.85)),
        int(255 * (1.0 - t * 0.55)),
        255,
    )


def _cm_row_normalize(mat: list[list[int]]) -> list[list[float]]:
    out: list[list[float]] = []
    for row in mat:
        total = float(sum(row))
        if total <= 0:
            out.append([0.0] * len(row))
        else:
            out.append([v / total for v in row])
    return out


def _load_cm_label_font(size: int):
    if ImageFont is None:
        return None
    fp = resolve_cjk_font_path()
    if fp:
        try:
            return ImageFont.truetype(fp, size)
        except Exception:
            pass
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def _truncate_cm_label(text: str, max_len: int = 14) -> str:
    s = str(text or "")
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def _draw_cm_panel_pil(
    data: list[list[float | int]],
    *,
    labels: list[str],
    title: str,
    color_from_unit_interval: bool,
    cell_size: int,
    label_font,
    cell_font,
) -> Image.Image:
    if Image is None or ImageDraw is None:
        raise RuntimeError("PIL 不可用")
    n = len(labels)
    label_w = max(72, min(220, cell_size * 4))
    label_h = max(72, min(220, cell_size * 4))
    top_h = 34
    panel_w = label_w + n * cell_size
    panel_h = top_h + label_h + n * cell_size
    img = Image.new("RGB", (panel_w, panel_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((4, 6), title, fill=(0, 0, 0), font=label_font)

    max_val = max((max(row) for row in data if row), default=1.0)
    if max_val <= 0:
        max_val = 1.0

    for i in range(n):
        for j in range(n):
            val = float(data[i][j])
            t = val if color_from_unit_interval else val / float(max_val)
            color = _cm_blues_color(t)
            x0 = label_w + j * cell_size
            y0 = top_h + label_h + i * cell_size
            draw.rectangle(
                [x0, y0, x0 + cell_size - 1, y0 + cell_size - 1],
                fill=color,
            )
            if n <= 30 and cell_size >= 14 and cell_font is not None:
                txt = f"{val:.2f}" if color_from_unit_interval else str(int(val))
                draw.text((x0 + 2, y0 + 1), txt, fill=(0, 0, 0), font=cell_font)

    for i, lab in enumerate(labels):
        y0 = top_h + label_h + i * cell_size + max(0, cell_size // 2 - 6)
        draw.text((4, y0), _truncate_cm_label(lab), fill=(0, 0, 0), font=label_font)

    for j, lab in enumerate(labels):
        x0 = label_w + j * cell_size + 2
        y0 = top_h + 6
        short = _truncate_cm_label(lab, 8)
        if len(short) <= 6:
            draw.text((x0, y0), "\n".join(short), fill=(0, 0, 0), font=label_font)
        else:
            draw.text((x0, y0), short, fill=(0, 0, 0), font=label_font)

    return img


def _save_confusion_matrix_png_matplotlib(
    png_path: str,
    *,
    labels: list[str],
    mat: list[list[int]],
    branch: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    _apply_matplotlib_cjk_font()

    arr = np.array(mat, dtype=float)
    row_sums = arr.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    norm = arr / row_sums

    fig, axes = plt.subplots(
        1, 2, figsize=(max(10, len(labels) * 0.45), max(5, len(labels) * 0.38))
    )
    for ax, data, title in (
        (axes[0], arr, "counts"),
        (axes[1], norm, "row-normalized (recall diag)"),
    ):
        im = ax.imshow(data, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(title)
        tick_marks = range(len(labels))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_ylabel("GT")
        ax.set_xlabel("Pred")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Confusion matrix — {branch}")
    with _suppress_matplotlib_missing_glyph_warnings():
        fig.tight_layout()
        fig.savefig(png_path, dpi=160)
    plt.close(fig)


def _save_confusion_matrix_png_pil(
    png_path: str,
    *,
    labels: list[str],
    mat: list[list[int]],
    branch: str,
) -> None:
    n = len(labels)
    cell_size = max(10, min(24, 720 // max(n, 1)))
    label_font = _load_cm_label_font(11)
    cell_font = _load_cm_label_font(max(8, cell_size // 2))
    norm = _cm_row_normalize(mat)
    left = _draw_cm_panel_pil(
        mat,
        labels=labels,
        title="counts",
        color_from_unit_interval=False,
        cell_size=cell_size,
        label_font=label_font,
        cell_font=cell_font,
    )
    right = _draw_cm_panel_pil(
        norm,
        labels=labels,
        title="row-normalized (recall diag)",
        color_from_unit_interval=True,
        cell_size=cell_size,
        label_font=label_font,
        cell_font=cell_font,
    )
    gap = 20
    title_h = 28
    combined = Image.new(
        "RGB",
        (left.width + gap + right.width, max(left.height, right.height) + title_h),
        (255, 255, 255),
    )
    draw = ImageDraw.Draw(combined)
    draw.text((4, 4), f"Confusion matrix — {branch}", fill=(0, 0, 0), font=label_font)
    combined.paste(left, (0, title_h))
    combined.paste(right, (left.width + gap, title_h))
    combined.save(png_path)


def _save_confusion_matrix_png(
    branch_dir: str,
    *,
    labels: list[str],
    mat: list[list[int]],
    branch: str,
) -> str:
    png_path = os.path.join(branch_dir, "confusion_matrix.png")
    mpl_err: Exception | None = None
    try:
        _save_confusion_matrix_png_matplotlib(
            png_path, labels=labels, mat=mat, branch=branch
        )
        print(f"混淆矩阵图: {png_path}")
        return png_path
    except Exception as e:
        mpl_err = e
        logging.debug("matplotlib 混淆矩阵失败，尝试 PIL: %s", e, exc_info=True)
    try:
        _save_confusion_matrix_png_pil(
            png_path, labels=labels, mat=mat, branch=branch
        )
        note = f"（matplotlib 不可用: {mpl_err}）" if mpl_err is not None else ""
        print(f"混淆矩阵图: {png_path} {note}".rstrip())
        return png_path
    except Exception as pil_err:
        msg = (
            f"混淆矩阵图写入失败: {png_path} "
            f"matplotlib={mpl_err} PIL={pil_err}"
        )
        logging.warning(msg)
        print(f"WARNING: {msg}")
        return ""


def _std_eval_save_branch(
    out_root: str,
    branch: str,
    cm: dict[tuple[str, str], int],
    stat_block: dict[str, int],
    meta: dict,
) -> None:
    branch_dir = os.path.join(out_root, branch)
    os.makedirs(branch_dir, exist_ok=True)
    labels = _std_eval_labels_from_cm(cm)
    mat, per_rows, overall = _std_eval_matrix_and_metrics(cm, labels=labels)

    csv_path = os.path.join(branch_dir, "confusion_matrix.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gt\\pred"] + labels)
        for i, row_lab in enumerate(labels):
            w.writerow([row_lab] + [str(mat[i][j]) for j in range(len(labels))])

    per_path = os.path.join(branch_dir, "per_class_precision_recall.csv")
    with open(per_path, "w", newline="", encoding="utf-8") as f:
        if per_rows:
            w = csv.DictWriter(f, fieldnames=list(per_rows[0].keys()))
            w.writeheader()
            w.writerows(per_rows)

    summary = {
        "branch": branch,
        "note": (
            "VOC-style matching; confusion rows/cols use normalize_class_name"
            + (
                " + hierarchy top-key rollup"
                if meta.get("l1_confusion_rollup") or meta.get("cls_merge_hierarchy")
                else ""
            )
            + "; __no_pred__=FN __extra_pred__=FP"
        ),
        "meta": meta,
        "aggregate_counts": {k: int(stat_block.get(k, 0)) for k in ("tp", "fp", "fn", "cls_err", "geom_pairs", "img_with_xml")},
        "overall_prf": overall,
        "confusion_labels": labels,
    }
    with open(os.path.join(branch_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if not labels:
        logging.warning("标准评估 %s: 无混淆计数，已跳过 heatmap", branch)
        return

    _save_confusion_matrix_png(branch_dir, labels=labels, mat=mat, branch=branch)


def _export_stat_by_cls_csv(
    out_root: str,
    branch: str,
    stat_by_cls: dict[str, dict[str, int]],
    *,
    sort_by_acc: bool = True,
    class_display_index: dict[str, str] | None = None,
    focus: frozenset[str] | None = None,
) -> None:
    if not stat_by_cls:
        return
    branch_dir = os.path.join(out_root, branch)
    os.makedirs(branch_dir, exist_ok=True)
    focus_set = frozenset(str(x).strip() for x in (focus or []) if str(x).strip())

    headers = [
        "class_norm",
        "class_display",
        "gt",
        "pred",
        "tp",
        "report_rate",
        "cls_err",
        "acc_rate",
        "fn",
        "miss_fn_rate",
        "cls_err_rate",
        "recall_gap",
        "fp",
        "fp_rate",
        "total_dev_rate",
    ]

    rows_with_sort: list[tuple[str, float, float, float, dict]] = []
    for cls_name in stat_by_cls.keys():
        if focus_set and str(cls_name) not in focus_set:
            continue
        s = stat_by_cls[cls_name]
        gt_n = int(s.get("gt", 0))
        pred_n = int(s.get("pred", 0))
        tp_n = int(s.get("tp", 0))
        ce_n = int(s.get("cls_err", 0))
        fn_n = int(s.get("fn", 0))
        fp_n = int(s.get("fp", 0))

        denom_gt = float(gt_n)
        denom_pred = float(pred_n)
        report_rate = (float(tp_n + ce_n) / denom_gt) if denom_gt > 0 else 0.0
        acc_rate = (float(tp_n) / denom_gt) if denom_gt > 0 else 0.0
        fp_rate = (float(fp_n) / denom_pred) if denom_pred > 0 else 0.0
        miss_fn_rate = (float(fn_n) / denom_gt) if denom_gt > 0 else 0.0
        cls_err_rate = (float(ce_n) / denom_gt) if denom_gt > 0 else 0.0
        recall_gap = (float(fn_n + ce_n) / denom_gt) if denom_gt > 0 else 0.0
        total_dev_rate = max(recall_gap, fp_rate)

        row = {
            "class_norm": str(cls_name),
            "class_display": resolve_eval_class_display_name(
                cls_name, display_index=class_display_index
            ),
            "gt": gt_n,
            "pred": pred_n,
            "tp": tp_n,
            "report_rate": round(report_rate, 6),
            "cls_err": ce_n,
            "acc_rate": round(acc_rate, 6),
            "fn": fn_n,
            "miss_fn_rate": round(miss_fn_rate, 6),
            "cls_err_rate": round(cls_err_rate, 6),
            "recall_gap": round(recall_gap, 6),
            "fp": fp_n,
            "fp_rate": round(fp_rate, 6),
            "total_dev_rate": round(total_dev_rate, 6),
        }
        rows_with_sort.append(
            (str(cls_name), float(total_dev_rate), float(report_rate), float(acc_rate), row)
        )

    if sort_by_acc:
        rows_with_sort.sort(
            key=lambda x: (
                x[1],  # total_dev_rate asc
                -x[2],  # report_rate desc
                -x[3],  # acc_rate desc
                -int(stat_by_cls.get(x[0], {}).get("gt", 0)),  # support desc
                x[0],
            )
        )
    else:
        rows_with_sort.sort(key=lambda x: x[0])

    out_path = os.path.join(branch_dir, "stat_by_class.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for _cls, _dev, _rep, _acc, row in rows_with_sort:
            w.writerow(row)


def _export_group_summary_csv(
    out_root: str,
    branch: str,
    stat_by_cls: dict[str, dict[str, int]],
    *,
    c1_set: set[str],
    c2_set: set[str],
) -> None:
    if not stat_by_cls:
        return
    branch_dir = os.path.join(out_root, branch)
    os.makedirs(branch_dir, exist_ok=True)

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

    headers = [
        "group",
        "gt",
        "pred",
        "tp",
        "fp",
        "fn",
        "cls_err",
        "miss_fn_rate",
        "cls_err_rate",
        "recall_gap",
        "fp_rate",
        "total_dev_rate",
    ]
    rows: list[dict] = []
    for group in ("一类害虫", "二类害虫", "其他虫子"):
        s = group_total[group]
        gt_n = int(s.get("gt", 0))
        pred_n = int(s.get("pred", 0))
        tp_n = int(s.get("tp", 0))
        ce_n = int(s.get("cls_err", 0))
        fn_n = int(s.get("fn", 0))
        fp_n = int(s.get("fp", 0))
        denom_gt = float(tp_n + ce_n + fn_n)
        denom_pred = float(tp_n + fp_n)
        miss_fn_rate = (float(fn_n) / denom_gt) if denom_gt > 0 else 0.0
        cls_err_rate = (float(ce_n) / denom_gt) if denom_gt > 0 else 0.0
        recall_gap = (float(fn_n + ce_n) / denom_gt) if denom_gt > 0 else 0.0
        fp_rate = (float(fp_n) / denom_pred) if denom_pred > 0 else 0.0
        total_dev = max(recall_gap, fp_rate)
        rows.append(
            {
                "group": group,
                "gt": gt_n,
                "pred": pred_n,
                "tp": tp_n,
                "fp": fp_n,
                "fn": fn_n,
                "cls_err": ce_n,
                "miss_fn_rate": round(miss_fn_rate, 6),
                "cls_err_rate": round(cls_err_rate, 6),
                "recall_gap": round(recall_gap, 6),
                "fp_rate": round(fp_rate, 6),
                "total_dev_rate": round(total_dev, 6),
            }
        )

    out_path = os.path.join(branch_dir, "group_summary_c1_c2_other.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(rows)


def _export_hn_report_csv(
    out_root: str,
    branch: str,
    stat_by_cls: dict[str, dict[str, int]],
    *,
    c1_set: set[str],
    hn_c2_set: set[str],
) -> None:
    if not stat_by_cls:
        return
    branch_dir = os.path.join(out_root, branch)
    os.makedirs(branch_dir, exist_ok=True)

    def _hn_acc(pred_n: int, gt_n: int) -> float:
        pred_n = int(pred_n)
        gt_n = int(gt_n)
        if gt_n <= 0:
            return 0.0
        acc = (1.0 - abs(float(pred_n - gt_n)) / float(gt_n)) * 100.0
        if acc < 0:
            return 0.0
        return float(acc)

    def _bucket(cls_norm: str) -> str:
        if cls_norm in c1_set:
            return "一类害虫"
        if cls_norm in hn_c2_set:
            return "湖南二类害虫"
        return "其他"

    hn_rows: list[dict] = []
    for cls_norm, s in stat_by_cls.items():
        cls_norm = str(cls_norm)
        gt_n = int(s.get("gt", 0))
        pred_n = int(s.get("pred", 0))
        if gt_n <= 0 and pred_n <= 0:
            continue
        hn_rows.append(
            {
                "group": _bucket(cls_norm),
                "class_norm": cls_norm,
                "gt": gt_n,
                "pred": pred_n,
                "acc_percent": round(_hn_acc(pred_n, gt_n), 6),
            }
        )
    bucket_rank = {"一类害虫": 0, "湖南二类害虫": 1, "其他": 9}
    hn_rows.sort(key=lambda x: (bucket_rank.get(x["group"], 9), x["group"], x["class_norm"]))

    out_path = os.path.join(branch_dir, "hn_counting_accuracy.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["group", "class_norm", "gt", "pred", "acc_percent"])
        w.writeheader()
        w.writerows(hn_rows)

    def _avg_acc(bucket: str) -> float:
        vals = [float(r["acc_percent"]) for r in hn_rows if r["group"] == bucket]
        if not vals:
            return 0.0
        return float(sum(vals) / float(len(vals)))

    avg_c1 = _avg_acc("一类害虫")
    avg_hn_c2 = _avg_acc("湖南二类害虫")
    day_acc = 0.6 * avg_c1 + 0.4 * avg_hn_c2
    day_score_30 = day_acc * 0.30
    with open(os.path.join(branch_dir, "hn_counting_accuracy_summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "avg_c1_percent",
                "avg_hn_c2_percent",
                "weighted_day_acc_percent",
                "score_30",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "avg_c1_percent": round(avg_c1, 6),
                "avg_hn_c2_percent": round(avg_hn_c2, 6),
                "weighted_day_acc_percent": round(day_acc, 6),
                "score_30": round(day_score_30, 6),
            }
        )


def _export_overall_summary_csv(out_root: str, branch: str, s: dict[str, int]) -> None:
    branch_dir = os.path.join(out_root, branch)
    os.makedirs(branch_dir, exist_ok=True)
    tp = int(s.get("tp", 0))
    fp = int(s.get("fp", 0))
    fn = int(s.get("fn", 0))
    ce = int(s.get("cls_err", 0))
    geom = int(s.get("geom_pairs", 0))
    denom_gt = float(tp + ce + fn)
    denom_pred = float(tp + fp)
    report_rate = (float(tp + ce) / denom_gt) if denom_gt > 0 else 0.0
    acc_rate = (float(tp) / denom_gt) if denom_gt > 0 else 0.0
    err_rate = (float(fp) / denom_pred) if denom_pred > 0 else 0.0
    miss_fn_rate = (float(fn) / denom_gt) if denom_gt > 0 else 0.0
    cls_err_rate = (float(ce) / denom_gt) if denom_gt > 0 else 0.0
    recall_gap = (float(fn + ce) / denom_gt) if denom_gt > 0 else 0.0
    total_dev = max(recall_gap, err_rate)
    out_path = os.path.join(branch_dir, "overall_summary.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "tp",
                "fp",
                "fn",
                "cls_err",
                "geom_pairs",
                "report_rate",
                "acc_rate",
                "err_rate",
                "miss_fn_rate",
                "cls_err_rate",
                "recall_gap",
                "total_dev_rate",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "cls_err": ce,
                "geom_pairs": geom,
                "report_rate": round(report_rate, 6),
                "acc_rate": round(acc_rate, 6),
                "err_rate": round(err_rate, 6),
                "miss_fn_rate": round(miss_fn_rate, 6),
                "cls_err_rate": round(cls_err_rate, 6),
                "recall_gap": round(recall_gap, 6),
                "total_dev_rate": round(total_dev, 6),
            }
        )
