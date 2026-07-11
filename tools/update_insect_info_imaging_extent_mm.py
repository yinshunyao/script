#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将 ``insect_info.json`` 每条记录 **顶层** ``min_mm`` / ``max_mm`` 更新为「成像典型跨度（毫米）」：
在分类 patch / 照片中虫体沿长轴方向的典型最大尺寸，常接近或大于 **体长**（如鳞翅目含翅展）。

- ``body_length_mm.{min_mm,max_mm}`` **不改**：仍表示身体尺度（与文献/表册体长一致）。
- 顶层 ``min_mm``：默认与 ``body_length_mm.min_mm`` 相同（成像下限仍取身体下限，避免过小）。
- 顶层 ``max_mm``：``round(body_max * k, 1)``，``k`` 由 ``name_zh`` 关键词粗分类（与农林业常见拍摄姿态一致）。

规则为 **保守估计**，便于后续按物种逐条人工替换为实测或文献翅展；运行前请备份 JSON。

用法：在 ``insect/`` 下 ``python script/tools/update_insect_info_imaging_extent_mm.py``（默认写回 ``script/config/insect_info.json``）。
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

# (pattern 或若干子串任一命中, max_mul, min_mul)；先匹配者优先
_RULES: List[Tuple[str, float, float]] = [
    # 鳞翅目：翅展常显著大于体长
    (r"(螟|野螟|夜蛾|灯蛾|毒蛾|天蛾|刺蛾|舟蛾|尺蠖|尺蛾|透翅蛾|斑蛾|麦蛾|果蛾|谷蛾|卷蛾|枯叶蛾|毒刺蛾)", 2.05, 1.0),
    (r"蝶", 2.15, 1.0),
    (r"蛾", 1.95, 1.0),
    # 蜻蜓豆娘
    (r"(蜓|蜻|豆娘)", 1.85, 1.0),
    # 直翅目：后足伸展
    (r"(蝗|蚂蚱|蟋蟀|螽斯|蝼蛄)", 1.55, 1.0),
    # 半翅同翅
    (r"(叶蝉|飞虱|蚜|粉虱|木虱)", 1.22, 1.0),
    (r"(蝽|盲蝽)", 1.28, 1.0),
    (r"蝉", 1.42, 1.0),
    # 鞘翅：翅鞘贴合，成像跨度≈体长略大
    (r"(金龟|象甲|天牛|瓢虫|叶甲|负泥虫|豆象|步甲|虎甲|铁甲|叩甲|萤|隐翅)", 1.1, 1.0),
    # 双翅
    (r"(蝇|虻|蚊)", 1.32, 1.0),
    # 膜翅
    (r"(蜂|蚁)", 1.22, 1.0),
    # 脉翅草蛉等
    (r"(草蛉|蚁蛉)", 1.28, 1.0),
    # 缨尾蜚蠊
    (r"(衣鱼|蜚蠊|蟑螂)", 1.12, 1.0),
]


def _max_mul_for_name(name_zh: str) -> Tuple[float, float]:
    s = str(name_zh or "")
    for pat, max_mul, min_mul in _RULES:
        if re.search(pat, s):
            return max_mul, min_mul
    return 1.28, 1.0


def _round1(x: float) -> float:
    return float(round(x + 1e-9, 1))


def update_catalog(data: Dict[str, Any]) -> Dict[str, int]:
    """就地修改 ``data``；返回统计 ``{"n": 条数}``。"""
    n = 0
    for key, ent in data.items():
        if not isinstance(ent, dict):
            continue
        bl = ent.get("body_length_mm")
        if not isinstance(bl, dict):
            continue
        bmin = bl.get("min_mm")
        bmax = bl.get("max_mm")
        if bmin is None or bmax is None:
            continue
        try:
            bf_min = float(bmin)
            bf_max = float(bmax)
        except (TypeError, ValueError):
            continue
        if bf_min <= 0 or bf_max <= 0:
            continue
        max_mul, min_mul = _max_mul_for_name(str(ent.get("name_zh") or ""))
        img_min = _round1(bf_min * min_mul)
        img_max = _round1(bf_max * max_mul)
        if img_max < img_min:
            img_max = img_min
        ent["min_mm"] = img_min
        ent["max_mm"] = img_max
        n += 1
    return {"n": n}


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / "script" / "config" / "insect_info.json"
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise SystemExit("root must be object")
    stats = update_catalog(data)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path} updated_entries={stats['n']}")


if __name__ == "__main__":
    main()
