#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""害虫配置：从 `insect_info.json` 加载关键字段，并提供区域索引与兼容别名（c1/c2）。"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Set, Tuple, Union, Any

Zone = Literal["全国", "华北", "华东", "华中", "华南", "西南", "西北", "东北"]
FeatureLevel = Literal["极低", "低", "中", "高", "-"]
PlanVersion = Literal["-", "已支持", "v1", "v2", "v3", "v1/v2", "v2？"]

ALL_REGIONAL_ZONES: Tuple[Zone, ...] = ("华北", "华东", "华中", "华南", "西南", "西北", "东北")

DEFAULT_JSON_PATH = Path(__file__).resolve().parent / "insect_info.json"


@dataclass(frozen=True)
class BodyLengthMm:
    max_mm: Optional[float]
    min_mm: Optional[float]
    note: Optional[str]
    raw: Optional[str]

    @staticmethod
    def from_obj(obj: Any) -> BodyLengthMm:
        if not isinstance(obj, dict):
            return BodyLengthMm(None, None, None, None)
        return BodyLengthMm(
            _f(obj.get("max_mm")),
            _f(obj.get("min_mm")),
            obj.get("note"),
            obj.get("raw"),
        )


@dataclass(frozen=True)
class PatchAveragePx:
    height_px: Optional[int]
    width_px: Optional[int]
    raw: Optional[str]

    @staticmethod
    def from_obj(obj: Any) -> PatchAveragePx:
        if not isinstance(obj, dict):
            return PatchAveragePx(None, None, None)
        return PatchAveragePx(
            _i(obj.get("height_px")),
            _i(obj.get("width_px")),
            obj.get("raw"),
        )


@dataclass(frozen=True)
class PestLevelJson:
    group_zh: str
    is_other_dataset_class: bool
    label_zh: str
    national_list_class: Optional[int]

    @staticmethod
    def from_obj(obj: Any) -> PestLevelJson:
        if not isinstance(obj, dict):
            return PestLevelJson("", True, "", None)
        return PestLevelJson(
            str(obj.get("group_zh") or ""),
            bool(obj.get("is_other_dataset_class", True)),
            str(obj.get("label_zh") or ""),
            obj.get("national_list_class") if obj.get("national_list_class") is not None else None,
        )


@dataclass(frozen=True)
class InsectJsonRecord:
    """`insect_info.json` 单条记录（全量昆虫/样本元数据）。"""

    pinyin: str
    name_zh: str
    category: str
    group: str
    data_dirs_raw: Optional[str]
    deliver_biong: bool
    max_mm: Optional[float]
    min_mm: Optional[float]
    sample_count: int
    sheet_row: int
    size_label: str
    regions: Tuple[str, ...]
    regions_raw: Optional[str]
    remark: Optional[str]
    training_plan_reserve_row: bool
    body_length_mm: BodyLengthMm
    patch_average_px: PatchAveragePx
    pest_level: PestLevelJson

    @staticmethod
    def from_dict(key: str, d: Dict[str, Any]) -> InsectJsonRecord:
        regs = d.get("regions") or []
        if not isinstance(regs, list):
            regs = []
        return InsectJsonRecord(
            pinyin=str(d.get("pinyin") or key),
            name_zh=str(d.get("name_zh") or ""),
            category=str(d.get("category") or ""),
            group=str(d.get("group") or ""),
            data_dirs_raw=d.get("data_dirs_raw"),
            deliver_biong=bool(d.get("deliver_biong", False)),
            max_mm=_f(d.get("max_mm")),
            min_mm=_f(d.get("min_mm")),
            sample_count=int(d.get("sample_count") or 0),
            sheet_row=int(d.get("sheet_row") or 0),
            size_label=str(d.get("size_label") or ""),
            regions=tuple(str(x) for x in regs),
            regions_raw=d.get("regions_raw"),
            remark=d.get("remark"),
            training_plan_reserve_row=bool(d.get("training_plan_reserve_row", False)),
            body_length_mm=BodyLengthMm.from_obj(d.get("body_length_mm")),
            patch_average_px=PatchAveragePx.from_obj(d.get("patch_average_px")),
            pest_level=PestLevelJson.from_obj(d.get("pest_level")),
        )


def _f(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _i(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class InsectCfg:
    """
    面向训练/规划的害虫配置（由 JSON 中国家一类/二类害虫记录规范化得到）。

    - key: 拼音类名
    - level: 对应 `national_list_class`（1 / 2）
    - zones: 覆盖区域；「全国」会在区域索引中按需展开到各大区
    - feature / plan: JSON 未提供时默认为 "-"（保留字段供后续人工或表外配置覆盖）
    - enabled: 对应 JSON 的 `deliver_biong`
    """

    key: str
    name_cn: str
    level: int
    zones: Tuple[Zone, ...]
    feature: FeatureLevel = "-"
    plan: PlanVersion = "-"
    enabled: bool = True
    notes: Optional[str] = None


def _normalize_region_token(raw: str) -> Optional[Zone]:
    base = raw.split("（")[0].split("(")[0].strip()
    if not base:
        return None
    if "全国" in base:
        return "全国"
    if base in ALL_REGIONAL_ZONES:
        return base  # type: ignore[return-value]
    return None


def regions_to_zones(regions: Iterable[str]) -> Tuple[Zone, ...]:
    """将 JSON 中的 `regions` 规范为 `Zone` 元组（去重且保序）。"""
    out: List[Zone] = []
    seen: Set[Zone] = set()
    for raw in regions:
        z = _normalize_region_token(raw)
        if z is None:
            continue
        if z == "全国":
            return ("全国",)
        if z not in seen:
            seen.add(z)
            out.append(z)
    return tuple(out)


def load_json_catalog(path: Optional[Union[str, Path]] = None) -> Dict[str, InsectJsonRecord]:
    """加载完整 JSON 目录（含「其他昆虫」）。"""
    p = Path(path) if path else DEFAULT_JSON_PATH
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("insect_info.json root must be an object")
    return {str(k): InsectJsonRecord.from_dict(str(k), v) for k, v in data.items() if isinstance(v, dict)}


def _build_insect_cfg_map(catalog: Dict[str, InsectJsonRecord]) -> Dict[str, InsectCfg]:
    out: Dict[str, InsectCfg] = {}
    for key, rec in catalog.items():
        nl = rec.pest_level.national_list_class
        if nl not in (1, 2):
            continue
        zones = regions_to_zones(rec.regions)
        out[key] = InsectCfg(
            key=key,
            name_cn=rec.name_zh,
            level=nl,
            zones=zones,
            feature="-",
            plan="-",
            enabled=rec.deliver_biong,
            notes=rec.remark,
        )
    return out


_JSON_CATALOG: Dict[str, InsectJsonRecord] = load_json_catalog()
INSECTS: Dict[str, InsectCfg] = _build_insect_cfg_map(_JSON_CATALOG)


def json_record(key: str) -> Optional[InsectJsonRecord]:
    """任意拼音 key 的原始 JSON 记录（含其他昆虫）；不存在则返回 None。"""
    return _JSON_CATALOG.get(key)


def _body_length_mm_bounds(rec: InsectJsonRecord) -> Tuple[float, float]:
    """体长毫米区间 ``(min_mm, max_mm)``：优先顶层 ``min_mm`` / ``max_mm``，否则用 ``body_length_mm``。"""
    lo = rec.min_mm if rec.min_mm is not None else rec.body_length_mm.min_mm
    hi = rec.max_mm if rec.max_mm is not None else rec.body_length_mm.max_mm
    if lo is None or hi is None:
        raise ValueError(f"昆虫 {rec.pinyin!r} 缺少有效的体长 min/max（毫米）")
    if lo > hi:
        lo, hi = hi, lo
    return (lo, hi)


def random_body_length_px(
    pinyin: str,
    px_per_mm: float,
    *,
    rng: Optional[random.Random] = None,
) -> float:
    """
    根据标定比例 ``px_per_mm``（1 mm 对应多少像素），在 JSON 中该昆虫体长范围内随机取一个长度（毫米），
    再换算为像素长度返回。

    - ``pinyin``：昆虫拼音 key，须在 ``insect_info.json`` 中存在。
    - ``px_per_mm``：须为正数。
    - ``rng``：可选 ``random.Random``，便于单测或可复现序列。
    """
    if px_per_mm <= 0:
        raise ValueError("px_per_mm 必须为正数")
    rec = _JSON_CATALOG.get(pinyin)
    if rec is None:
        raise ValueError(f"未知昆虫拼音: {pinyin!r}")
    lo, hi = _body_length_mm_bounds(rec)
    rnd = rng if rng is not None else random
    mm = rnd.uniform(lo, hi)
    return mm * px_per_mm


def reload_catalog(path: Optional[Union[str, Path]] = None) -> None:
    """测试或热加载用：重新读 JSON 并刷新 `INSECTS` / 兼容变量 / 区域索引。"""
    global _JSON_CATALOG, INSECTS, c1, c2, ZONE_TO_PEST_PINYIN
    _JSON_CATALOG = load_json_catalog(path)
    INSECTS = _build_insect_cfg_map(_JSON_CATALOG)
    c1 = [k for k, v in INSECTS.items() if v.level == 1 and v.enabled]
    c2 = {
        k: {"name": v.name_cn, "zone": list(v.zones)}
        for k, v in INSECTS.items()
        if v.level == 2 and v.enabled
    }
    ZONE_TO_PEST_PINYIN = build_zone_to_pinyin_index()


# ----------------------------
# 兼容层：保留原始 c1/c2 变量形态（仅供旧脚本快速使用）
# ----------------------------
c1: List[str] = [k for k, v in INSECTS.items() if v.level == 1 and v.enabled]

c2: Dict[str, Dict[str, Union[str, List[str]]]] = {
    k: {"name": v.name_cn, "zone": list(v.zones)}
    for k, v in INSECTS.items()
    if v.level == 2 and v.enabled
}


def build_zone_to_pinyin_index(
    *,
    include_disabled: bool = False,
    expand_nationwide_to_regions: bool = True,
    include_key_zone_quanguo: bool = True,
) -> Dict[str, List[str]]:
    """
    区域 -> 害虫拼音 key 的反向索引（用于「按区域建议可能害虫」）。

    - include_disabled: 是否包含 enabled=False（无样本/暂不做）的项
    - expand_nationwide_to_regions: 全国性害虫是否同时挂到各大区（华北/华东/…）
    - include_key_zone_quanguo: 是否保留 "全国" 这个索引键
    """

    zone_to_keys: Dict[str, Set[str]] = {}

    def _add(zone: str, key: str) -> None:
        zone_to_keys.setdefault(zone, set()).add(key)

    for key, cfg in INSECTS.items():
        if (not include_disabled) and (not cfg.enabled):
            continue

        if "全国" in cfg.zones:
            if include_key_zone_quanguo:
                _add("全国", key)
            if expand_nationwide_to_regions:
                for z in ALL_REGIONAL_ZONES:
                    _add(z, key)
        else:
            for z in cfg.zones:
                _add(z, key)

    return {z: sorted(keys) for z, keys in sorted(zone_to_keys.items(), key=lambda x: x[0])}


ZONE_TO_PEST_PINYIN: Dict[str, List[str]] = build_zone_to_pinyin_index()


def suggest_pests_by_zone(zone: str, *, level: Optional[int] = None) -> List[str]:
    """
    给定区域（如 "华北"），返回建议害虫拼音 key 列表。
    """

    keys = ZONE_TO_PEST_PINYIN.get(zone, [])
    if level is None:
        return list(keys)
    return [k for k in keys if INSECTS.get(k) and INSECTS[k].level == level]


def iter_deliver_pests(level: Optional[int] = None) -> List[InsectJsonRecord]:
    """遍历 `deliver_biong=True` 且国家一类/二类的记录（可选按 level 过滤）。"""
    rows: List[InsectJsonRecord] = []
    for rec in _JSON_CATALOG.values():
        if not rec.deliver_biong:
            continue
        nl = rec.pest_level.national_list_class
        if nl not in (1, 2):
            continue
        if level is not None and nl != level:
            continue
        rows.append(rec)
    return sorted(rows, key=lambda r: (r.pest_level.national_list_class or 9, r.pinyin))


if __name__ == "__main__":
    # 直接运行本文件时在下方改参数试跑（无需命令行参数）
    DEMO_ZONE = "华北"
    DEMO_KEY = "bazidilaohu"

    print("--- 统计 ---")
    print(f"JSON 总条数: {len(_JSON_CATALOG)}")
    print(f"国家一类/二类（INSECTS）: {len(INSECTS)}")
    print(f"c1 enabled: {len(c1)} -> {c1}")
    print(f"c2 enabled: {len(c2)} 条")

    print("\n--- 单条 JSON 记录（含尺寸/样本）---")
    r = json_record(DEMO_KEY)
    if r:
        print(f"  key={DEMO_KEY} name={r.name_zh} deliver={r.deliver_biong}")
        print(f"  category={r.category} national_class={r.pest_level.national_list_class}")
        print(f"  regions={list(r.regions)} sample_count={r.sample_count}")
        bl = r.body_length_mm
        print(f"  body_mm: raw={bl.raw} min={bl.min_mm} max={bl.max_mm}")

    print(f"\n--- 区域「{DEMO_ZONE}」建议害虫（启用项）---")
    print(suggest_pests_by_zone(DEMO_ZONE))

    print("\n--- deliver_biong 且二类（节选拼音）---")
    l2 = iter_deliver_pests(level=2)
    print([x.pinyin for x in l2[:12]], "...")

    print("\n--- 同类记录按样本数 Top 5（仅国家一二类）---")
    ranked = sorted(
        (x for x in _JSON_CATALOG.values() if x.pest_level.national_list_class in (1, 2)),
        key=lambda x: -x.sample_count,
    )[:5]
    for x in ranked:
        print(f"  {x.pinyin}: {x.name_zh} samples={x.sample_count}")

    print("\n--- 随机体长（像素），px/mm=8.0 ---")
    for _ in range(3):
        px = random_body_length_px(DEMO_KEY, 8.0)
        print(f"  {DEMO_KEY}: {px:.2f} px")
