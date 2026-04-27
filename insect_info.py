#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/04/25
# @Author  : ysy
# @Email   : xxx@qq.com 
# @Detail  : 
# @Software: PyCharm
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Set, Tuple, Union

Zone = Literal["全国", "华北", "华东", "华中", "华南", "西南", "西北", "东北"]
FeatureLevel = Literal["极低", "低", "中", "高", "-"]
PlanVersion = Literal["-", "已支持", "v1", "v2", "v3", "v1/v2", "v2？"]

ALL_REGIONAL_ZONES: Tuple[Zone, ...] = ("华北", "华东", "华中", "华南", "西南", "西北", "东北")


@dataclass(frozen=True)
class InsectCfg:
    """
    统一的害虫配置结构（c1/c2 的“规范化版本”）。

    - key: 训练/推理侧使用的拼音类名（建议全小写、无空格）
    - level: 需求表中的级别（1/2）
    - zones: 覆盖区域；使用 "全国" 表示全国性（可在反向索引中展开到各大区）
    - feature: 特征程度（用于优先级规划，不影响代码逻辑）
    - plan: 规划版本（用于路线图，不影响代码逻辑）
    - enabled: 是否纳入当前“建议/输出”（没有样本/暂不处理的置 False）
    """

    key: str
    name_cn: str
    level: int
    zones: Tuple[Zone, ...]
    feature: FeatureLevel = "-"
    plan: PlanVersion = "-"
    enabled: bool = True
    notes: Optional[str] = None


def _z(*zones: Zone) -> Tuple[Zone, ...]:
    return tuple(zones)


# ----------------------------
# 规范化配置（建议后续统一从这里读）
# ----------------------------
INSECTS: Dict[str, InsectCfg] = {
    # 级别 1：全国
    "daozongjuanyeming": InsectCfg("daozongjuanyeming", "稻纵卷叶螟", 1, _z("全国"), "高", "v1/v2", True, "v1 检出高但误报多"),
    "daofeishi": InsectCfg("daofeishi", "稻飞虱", 1, _z("全国"), "低", "已支持", True, "含褐飞虱/白背飞虱（细粒度）"),
    "hefeishi": InsectCfg("hefeishi", "褐飞虱", 1, _z("全国"), "低", "已支持", True, "稻飞虱子类"),
    "baibeifeishi": InsectCfg("baibeifeishi", "白背飞虱", 1, _z("全国"), "低", "已支持", True, "稻飞虱子类"),
    "caoditanyee": InsectCfg("caoditanyee", "草地贪夜蛾", 1, _z("全国"), "低", "v1", True, "特征不太明显，效果一般"),
    "feihuang": InsectCfg("feihuang", "飞蝗", 1, _z("全国"), "-", "-", False, "没有样本"),
    # 粘虫：含东方粘虫/劳氏粘虫；当前 key 沿用历史写法（nianchong），避免影响已有脚本
    "dongfangnianchong": InsectCfg("dongfangnianchong", "东方粘虫", 1, _z("全国"), "低", "v3", True, "与劳氏粘虫合并；v2 不做细分类"),
    "laoshinianchong": InsectCfg("laoshinianchong", "劳氏粘虫", 1, _z("全国"), "低", "v3", True, "与东方粘虫合并；v2 不做细分类"),
    "erhuaming": InsectCfg("erhuaming", "二化螟", 1, _z("全国"), "中", "v2", True, None),
    "xiaomaiaichong": InsectCfg("xiaomaiaichong", "小麦蚜虫", 1, _z("全国"), "极低", "-", False, "含麦长管蚜/禾谷缢管蚜/麦二叉蚜（暂不做）"),
    "yumiming": InsectCfg("yumiming", "玉米螟", 1, _z("全国"), "中", "v2", True, None),
    "shucaijima": InsectCfg("shucaijima", "蔬菜蓟马", 1, _z("全国"), "极低", "-", False, "比较难，暂时不处理"),
    "caodiming": InsectCfg("caodiming", "草地螟", 1, _z("全国"), "-", "-", False, "没有样本"),

    # 级别 2：分区
    "mianlingchong": InsectCfg("mianlingchong", "棉铃虫", 2, _z("华北", "华中", "西南", "西北"), "中", "v2", True, None),
    "xiaocaie": InsectCfg("xiaocaie", "小菜蛾", 2, _z("华北", "华东", "西南"), "高", "v2？", True, "小虫，视情况"),
    "ganjuxiaoshiying": InsectCfg("ganjuxiaoshiying", "柑橘小实蝇", 2, _z("华中", "华南", "西南"), "-", "-", False, "没有样本"),
    # 地老虎：v2 不做细分类；华北只关注八字地老虎
    "bazidilaohu": InsectCfg("bazidilaohu", "八字地老虎", 2, _z("华北", "西南", "西北"), "中", "v3", True, "华北地区重点关注"),
    "dadilaohu": InsectCfg("dadilaohu", "大地老虎", 2, _z("华北", "西南", "西北"), "中", "v3", True, "v2 不做细分类"),
    "xiaodilaohu": InsectCfg("xiaodilaohu", "小地老虎", 2, _z("华北", "西南", "西北"), "中", "v3", True, "又名黄地老虎；v2 不做细分类"),
    "huangdilaohu": InsectCfg("huangdilaohu", "黄地老虎", 2, _z("华北", "西南", "西北"), "中", "v3", True, "小地老虎别名"),
    "tiancaiyee": InsectCfg("tiancaiyee", "甜菜夜蛾", 2, _z("华北", "西南", "西北"), "中", "-", True, "生产环境样本，质量一般"),
    # 盲蝽：西南只关注绿盲蝽（但暂无样本）
    "lvmangchun": InsectCfg("lvmangchun", "绿盲蝽", 2, _z("华北", "西南", "西北"), "-", "-", False, "西南只关注绿盲蝽；无样本"),
    "chixumangchun": InsectCfg("chixumangchun", "赤须盲蝽", 2, _z("华北", "西南", "西北"), "-", "-", False, "无样本"),
    "muxuanmangchun": InsectCfg("muxuanmangchun", "苜蓿盲蝽", 2, _z("华北", "西南", "西北"), "-", "-", False, "无样本"),
    "malingshujiachong": InsectCfg("malingshujiachong", "马铃薯甲虫", 2, _z("华北", "东北"), "-", "-", False, "没有样本"),
    "daming": InsectCfg("daming", "大螟", 2, _z("华东", "西南"), "中", "v2", True, None),
    "caiqingchong": InsectCfg("caiqingchong", "菜青虫", 2, _z("华东", "西南"), "-", "-", False, "没有样本"),
    "chaoxiaolvyechan": InsectCfg("chaoxiaolvyechan", "茶小绿叶蝉", 2, _z("华东", "西南"), "-", "-", False, "没有样本"),
    "yachong": InsectCfg("yachong", "蚜虫", 2, _z("华中", "西南"), "-", "-", False, "没有样本"),
    "ganjudashiying": InsectCfg("ganjudashiying", "柑橘大实蝇", 2, _z("华中", "西南"), "-", "-", False, "没有样本"),
    "taoxiaoshixinchong": InsectCfg("taoxiaoshixinchong", "桃小食心虫", 2, _z("华北", "西南"), "-", "-", False, "没有样本"),
    "lixiaoshixinchong": InsectCfg("lixiaoshixinchong", "梨小食心虫", 2, _z("华北", "西南"), "-", "-", False, "没有样本"),
    "erdianweiyee": InsectCfg("erdianweiyee", "二点委夜蛾", 2, _z("华北", "西南"), "低", "-", True, "有样本"),
    "dadoushixinchong": InsectCfg("dadoushixinchong", "大豆食心虫", 2, _z("东北", "西南"), "-", "-", True, None),
    "pingguodue": InsectCfg("pingguodue", "苹果蠹蛾", 2, _z("华北", "西北"), "-", "-", True, None),
    "limushi": InsectCfg("limushi", "梨木虱", 2, _z("华东", "西北"), "-", "-", True, None),
    "mianya": InsectCfg("mianya", "棉蚜", 2, _z("华中", "西北"), "-", "-", True, None),
    "mianyeman": InsectCfg("mianyeman", "棉叶螨", 2, _z("华中", "西北"), "-", "-", True, None),
    "daoshuixiangjia": InsectCfg("daoshuixiangjia", "稻水象甲", 2, _z("西南", "东北"), "-", "-", True, None),
    "sanhuaming": InsectCfg("sanhuaming", "三化螟", 2, _z("西南"), "-", "-", True, None),
    "taozhuming": InsectCfg("taozhuming", "桃蛀螟", 2, _z("西南"), "高", "v2", True, None),
    "yanqingchong": InsectCfg("yanqingchong", "烟青虫", 2, _z("西南"), "-", "-", True, None),
    "ganjuqianyee": InsectCfg("ganjuqianyee", "柑橘潜叶蛾", 2, _z("西南"), "-", "-", True, None),
    "doujiayeming": InsectCfg("doujiayeming", "豆荚野螟", 2, _z("西南"), "-", "-", True, None),
    "doujiajiaming": InsectCfg("doujiajiaming", "豆荚荚螟", 2, _z("西南"), "-", "-", True, None),
    "chachihuo": InsectCfg("chachihuo", "茶尺蠖", 2, _z("西南"), "-", "-", True, None),
    "chamaochong": InsectCfg("chamaochong", "茶毛虫", 2, _z("西南"), "-", "-", True, None),
    "xiewenyee": InsectCfg("xiewenyee", "斜纹夜蛾", 2, _z("西南"), "-", "-", True, None),
    "erdianming": InsectCfg("erdianming", "二点螟", 2, _z("西南"), "-", "-", True, None),
    "tiaoming": InsectCfg("tiaoming", "条螟", 2, _z("西南"), "-", "-", True, None),
    "sangming": InsectCfg("sangming", "桑螟", 2, _z("西南"), "-", "-", True, None),
    "huangming": InsectCfg("huangming", "黄螟", 2, _z("西南"), "-", "-", True, None),
    "jinguizi": InsectCfg("jinguizi", "金龟子", 2, _z("西南"), "-", "-", True, None),
    "jinzhenchong": InsectCfg("jinzhenchong", "金针虫", 2, _z("西南"), "-", "-", True, None),
    "dongfanglougu": InsectCfg("dongfanglougu", "蝼蛄", 2, _z("西南"), "-", "-", True, None),
    "tiancaibaidaiyeming": InsectCfg("tiancaibaidaiyeming", "甜菜白带野螟", 2, _z("华北"), "高", "v1/v2", True, "v1 利用北京样本训练"),
}


# ----------------------------
# 兼容层：保留原始 c1/c2 变量形态（仅供旧脚本快速使用）
# ----------------------------
# 一类害虫：历史上用 list；这里按 level=1 且 enabled=True 导出
c1: List[str] = [k for k, v in INSECTS.items() if v.level == 1 and v.enabled]

# 二类害虫：历史上是 dict（pinyin -> 中文名 or dict）；这里统一导出为 dict(pinyin -> {"name":..., "zone":[...]})
c2: Dict[str, Dict[str, Union[str, List[str]]]] = {
    k: {"name": v.name_cn, "zone": list(v.zones)} for k, v in INSECTS.items() if v.level == 2 and v.enabled
}


def build_zone_to_pinyin_index(
    *,
    include_disabled: bool = False,
    expand_nationwide_to_regions: bool = True,
    include_key_zone_quanguo: bool = True,
) -> Dict[str, List[str]]:
    """
    区域 -> 害虫拼音 key 的反向索引（用于“按区域建议可能害虫”）。

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