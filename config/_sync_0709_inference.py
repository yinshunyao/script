#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""一次性同步 0709 新增物种到推理配置（postprocess.other_classes / cls.out / insect_info）。"""
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

_CONFIG_DIR = Path(__file__).resolve().parent
_TRAIN_ROOT = _CONFIG_DIR.parents[1] / "train" / "train_cls"
if str(_TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAIN_ROOT))

from cls_map import leaf_to_zh_dir  # noqa: E402

# merge 组拼音 key → 中文名（0709 新增 other 组）
NEW_OTHER_KEYS: dict[str, str] = {
    "baigouxiaojuane": "白勾小卷蛾",
    "dihongchun": "地红蝽",
    "fanzhie": "泛尺蛾",
    "hongliudiantiane": "红六点天蛾",
    "huaiyuzhoue": "槐羽舟蛾",
    "huanglianmuchihuo": "黄连木尺蠖",
    "huibandoutiane": "灰斑豆天蛾",
    "huixiongtusaijingui": "灰胸突鳃金龟",
    "jianzhuienyeming": "尖锥额野螟",
    "juxianchie": "锯线尺蛾",
    "koujia": "叩甲",
    "liuyinchibanminge": "柳阴翅斑螟蛾",
    "longshi": "龙虱",
    "mangchun": "盲蝽",
    "meiguobaie": "美国白蛾",
    "mianshuangxiejuane": "棉双斜卷蛾",
    "mudue": "木蠹蛾",
    "pingyanchie": "苹烟尺蛾",
    "qingchie": "青尺蛾",
    "sandianmuxumangchun": "三点苜蓿盲蝽",
    "tuchun": "土蝽",
    "tujia": "土甲",
    "xiaohongjichie": "小红姬尺蛾",
    "yajia": "牙甲",
    "yanfumuheitiaoming": "盐肤木黑条螟",
    "zhonghuazhendibie": "中华真地鳖",
}

CLS_OUT_DEFAULT = {"enable": True, "cls_conf": 0.3}

ALG_FILES = (
    "insect_alg_all.json",
    "insect_alg_shengchan.json",
    "insect_alg_other.json",
)


def _leaf_dir_for_key(key: str) -> str:
    for leaf in leaf_to_zh_dir:
        if leaf == f"{key}-ba-sc" or leaf == f"{key}-ba" or leaf == key:
            return leaf
    return f"{key}-ba-sc"


def _minimal_insect_info(key: str, name_zh: str) -> dict:
    return {
        "body_length_mm": {"max_mm": None, "min_mm": None, "note": None, "raw": None},
        "category": "其他昆虫",
        "data_dirs_raw": _leaf_dir_for_key(key),
        "deliver_biong": False,
        "group": "其他昆虫",
        "max_mm": None,
        "min_mm": None,
        "name_zh": name_zh,
        "patch_average_px": {"height_px": None, "raw": None, "width_px": None},
        "pest_level": {
            "group_zh": "其他昆虫",
            "is_other_dataset_class": True,
            "label_zh": "其他昆虫",
            "national_list_class": None,
        },
        "pinyin": key,
        "regions": [],
        "regions_raw": None,
        "remark": "0709 北京生产+辛集样本更新批量补齐",
        "sample_count": None,
        "sheet_row": None,
        "size_label": "中虫",
        "training_plan_reserve_row": False,
    }


def _cls_out_path(cfg: dict) -> dict:
    return cfg["models"]["detect_big"]["out"]["*"]["models"]["cls"]["out"]


def sync_alg_json(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as f:
        cfg = json.load(f)
    added: list[str] = []

    if path.name == "insect_alg_all.json":
        other = cfg.setdefault("postprocess", {}).setdefault("other_classes", [])
        for key in sorted(NEW_OTHER_KEYS):
            if key not in other:
                # 按拼音序插入，不整体重排已有项
                inserted = False
                for i, existing in enumerate(other):
                    if key < existing:
                        other.insert(i, key)
                        inserted = True
                        break
                if not inserted:
                    other.append(key)
                added.append(f"other_classes:{key}")

    cls_out = _cls_out_path(cfg)
    for key in sorted(NEW_OTHER_KEYS):
        if key not in cls_out:
            # cls.out 保持拼音序：在首个大于 key 的项前插入
            new_out: dict = {}
            inserted = False
            for existing_key, existing_val in cls_out.items():
                if not inserted and key < existing_key:
                    new_out[key] = deepcopy(CLS_OUT_DEFAULT)
                    inserted = True
                new_out[existing_key] = existing_val
            if not inserted:
                new_out[key] = deepcopy(CLS_OUT_DEFAULT)
            else:
                cls_out.clear()
                cls_out.update(new_out)
                added.append(f"cls.out:{key}")
                continue
            cls_out.clear()
            cls_out.update(new_out)
            added.append(f"cls.out:{key}")

    with path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return added


def sync_insect_info(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as f:
        info = json.load(f)
    added: list[str] = []
    for key, name_zh in sorted(NEW_OTHER_KEYS.items()):
        if key not in info:
            info[key] = _minimal_insect_info(key, name_zh)
            added.append(key)
    with path.open("w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return added


def main() -> None:
    for name in ALG_FILES:
        changes = sync_alg_json(_CONFIG_DIR / name)
        print(f"{name}: {len(changes)} changes")
        for c in changes:
            print(f"  + {c}")
    info_added = sync_insect_info(_CONFIG_DIR / "insect_info.json")
    print(f"insect_info.json: {len(info_added)} new entries")


if __name__ == "__main__":
    main()
