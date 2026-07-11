#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""北京比赛清单（``比赛虫体.md``）canonical 中文名与历史/异名 → 拼音 key。"""
from __future__ import annotations

from pathlib import Path

_COMPETITION_MD = Path(__file__).resolve().parent / "比赛虫体.md"

# 拼音 key → 比赛清单 canonical 中文（覆盖 insect_info / build_info / alg 中的旧称）
COMPETITION_CANONICAL_CN: dict[str, str] = {
    "jingwendiyee": "警纹地老虎",
    "huangchizhuiyeyeming": "黄翅缀野螟",
    "simianmujinxingchie": "丝棉木金星尺蛾",
    "taozhuming": "桃蛀螟",
    "xuanqiyee": "旋歧夜蛾",
    "yinyawenyee": "隐丫纹夜蛾",
    "huangheyilijingui": "黄褐丽金龟",
    "tonglvyilijingui": "铜绿异丽金龟",
    "zhonghualanbujia": "中华星步甲",
}

# 历史标注 / 样本目录 / 异名 → 拼音（canonical 见上表或 insect_alg_all ``cn_name``）
COMPETITION_CN_ALIASES: dict[str, str] = {
    "铜绿丽金龟": "tonglvyilijingui",
    "黄褐丽金龟": "huangheyilijingui",
    "黄褐异丽金龟": "huangheyilijingui",
    "警纹地夜蛾": "jingwendiyee",
    "黄翅缀叶野螟": "huangchizhuiyeyeming",
    "丝棉木金星尺蛾": "simianmujinxingchie",
    "丝绵木金星尺蛾": "simianmujinxingchie",
    "桃柱螟": "taozhuming",
    "旋岐夜蛾": "xuanqiyee",
    "旋幽夜蛾": "xuanqiyee",
    "银丫纹夜蛾": "yinyawenyee",
    "隐丫纹夜蛾": "yinyawenyee",
    "姬蜂": "yeesoujifeng",
    "客来夜蛾": "kelaiyee",
    "白点暗野螟": "baidiananyeming",
    "旱柳原野螟": "hanliuyuanyeming",
    "华北大黑鳃金龟": "daheisaijingui",
    "榄绿歧角螟": "lanlvqijiaoming",
    "国槐尺蠖": "huaichie",
    "guohuaichihuo": "huaichie",
    "中华星步甲": "zhonghualanbujia",
    "中华婪步甲": "zhonghualanbujia",  # 历史标注/比昂目录名，同种
    "梨小食心虫": "lixiaoshixinchong",
    "小蜡卷须野螟": "xiaolajuanxuyeming",
    "白脉粘夜蛾": "baimaizhanyee",
    "竹织叶野螟":     "zhuzhiyeyeming",
    "白背飞虱": "baibeifeishi",
    "褐飞虱": "hefeishi",
    "灰飞虱": "huifeishi",
    "泛尺蛾": "fanzhie",
    "木蠹蛾": "mudue",
    "叩甲": "koujia",
    "盲蝽": "mangchun",
    "棉双斜卷蛾": "mianshuangxiejuane",
}

# alg out / top 清单拼音 key → 内部 canonical 拼音（模型输出与训练目录）
COMPETITION_PINYIN_ALIASES: dict[str, str] = {
    "zhonghuaxingbujia": "zhonghualanbujia",
}


def load_competition_tier_cn(
    md_path: Path | None = None,
) -> dict[str, list[str]]:
    """解析 ``比赛虫体.md``，返回 ``top1`` / ``top2`` / ``top3`` 中文名列表。"""
    path = md_path or _COMPETITION_MD
    sections: dict[str, list[str]] = {"top1": [], "top2": [], "top3": []}
    cur: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("# 重点一级"):
            cur = "top1"
            continue
        if line.startswith("# 重点其他"):
            cur = "top2"
            continue
        if line.startswith("# 重要top3"):
            cur = "top3"
            continue
        if line.startswith("#"):
            cur = None
            continue
        if cur:
            sections[cur].append(line)
    return sections
