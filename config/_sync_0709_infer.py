#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""一次性：0709 新增 other 类同步到 insect_alg_*.json（postprocess.other_classes + cls out）。"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent

# 0709 新增 cls_merge other 组（辛集清单 + 北京生产）；拼音序
NEW_OTHER_0709: list[str] = sorted([
    "baigouxiaojuane",
    "dihongchun",
    "fanzhie",
    "hongliudiantiane",
    "huaiyuzhoue",
    "huanglianmuchihuo",
    "huibandoutiane",
    "huixiongtusaijingui",
    "jianzhuienyeming",
    "juxianchie",
    "koujia",
    "liuyinchibanminge",
    "longshi",
    "mangchun",
    "meiguobaie",
    "mianshuangxiejuane",
    "mudue",
    "pingyanchie",
    "qingchie",
    "sandianmuxumangchun",
    "tuchun",
    "tujia",
    "xiaohongjichie",
    "yajia",
    "yanfumuheitiaoming",
    "zhonghuazhendibie",
])

CLS_OUT_ENTRY = {"enable": True, "cls_conf": 0.3}


def _sorted_unique(items: list[str]) -> list[str]:
    return sorted(set(items))


def _merge_other_classes(cfg: dict) -> list[str]:
    pp = cfg.setdefault("postprocess", {})
    cur = list(pp.get("other_classes", []))
    merged = _sorted_unique(cur + NEW_OTHER_0709)
    pp["other_classes"] = merged
    return merged


def _merge_cls_out(cfg: dict, *, rich: bool = False) -> int:
    try:
        out = cfg["models"]["detect_big"]["out"]["*"]["models"]["cls"]["out"]
    except KeyError as exc:
        raise KeyError(f"detect_big cls out not found: {exc}") from exc
    info: dict = {}
    if rich:
        with (CONFIG_DIR / "insect_info.json").open(encoding="utf-8") as f:
            info = json.load(f)
    added = 0
    for key in NEW_OTHER_0709:
        if key not in out:
            out[key] = deepcopy(CLS_OUT_ENTRY)
            added += 1
        entry = out[key]
        if rich and key in info:
            entry["cn_name"] = info[key].get("name_zh", key)
            entry.setdefault("infer_name", key)
            entry.setdefault("enable", True)
            entry.setdefault("cls_conf", 0.3)
            if key == "fanzhie":
                entry["bak"] = "泛尺蛾-比昂-生产；与 chie（尺蛾背景过滤）区分"
    cfg["models"]["detect_big"]["out"]["*"]["models"]["cls"]["out"] = dict(
        sorted(out.items(), key=lambda kv: kv[0])
    )
    return added


def _patch_file(path: Path) -> None:
    with path.open(encoding="utf-8") as f:
        cfg = json.load(f)
    rich = path.name == "insect_alg_all.json"
    if rich:
        _merge_other_classes(cfg)
    added = _merge_cls_out(cfg, rich=rich)
    with path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"{path.name}: cls out +{added}")


def main() -> None:
    for name in ("insect_alg_all.json", "insect_alg_shengchan.json", "insect_alg_other.json"):
        _patch_file(CONFIG_DIR / name)
    with (CONFIG_DIR / "insect_alg_all.json").open(encoding="utf-8") as f:
        other_n = len(json.load(f)["postprocess"]["other_classes"])
    print(f"other_classes total: {other_n}")


if __name__ == "__main__":
    main()
