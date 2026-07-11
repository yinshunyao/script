#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
为 ``insect/script/config/insect_info.json`` 补齐 ``batch_aug_classes.CLASS_BATCH_ENTRIES`` 中尚缺的物种 key。

- 每条记录格式与现有条目一致；``body_length_mm`` 取该物种常见**体长**（mm）。
- 顶层 ``min_mm`` / ``max_mm`` 由 ``update_insect_info_imaging_extent_mm.py`` 的规则统一推导（成像跨度，含翅展等），
  本脚本只填体长与元数据，运行后再跑一次更新脚本会覆盖顶层成像跨度。
- ``size_label`` 由体长粗分：``<8`` 小虫；``8–25`` 中虫；``>25`` 大虫。
- 其余字段统一为 ``其他昆虫`` 模板；按需后续在 OR/DR 文档里再细化。

参考资料：
- ``insect/doc/参考资料/样本分析/分类清单.md``（中文名映射）
- 体长按常见种类与文献一般描述给出区间，``remark`` 标注「批量补齐」便于后续逐条校对。

用法：在 ``insect/`` 下 ``python script/tools/add_missing_insect_info_entries.py``，再运行
``python script/tools/update_insect_info_imaging_extent_mm.py`` 同步顶层成像跨度。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

# (pinyin, name_zh, body_min_mm, body_max_mm, note_for_remark)
ENTRIES: List[Tuple[str, str, float, float, str]] = [
    ("bowenyee", "波纹夜蛾", 15.0, 19.0, "Diarsia canescens 体长参考 15–19mm，翅展约 32–40mm"),
    ("caoling", "草蛉", 8.0, 15.0, "Chrysopidae 体长 8–15mm，翅展可达 30mm"),
    ("chenwudenge", "尘污灯蛾", 15.0, 20.0, "Spilarctia obliqua 体长约 15–20mm；翅展 40–55mm"),
    ("chimeidongyee", "齿美冬夜蛾", 13.0, 17.0, "Erannis defoliaria 类，体长 13–17mm"),
    ("danmainianchong", "淡脉粘虫", 16.0, 20.0, "Mythimna loreyi 体长 16–20mm，翅展 32–40mm"),
    ("daofeishi", "稻飞虱", 3.0, 5.0, "稻飞虱（褐飞虱/白背飞虱合称） 3–5mm"),
    ("dianguangyechan", "电光叶蝉", 3.0, 5.0, "Recilia dorsalis 体长 3–5mm"),
    ("dingyee", "鼎夜蛾", 13.0, 17.0, "Aedia leucomelas / 类，体长 13–17mm"),
    ("fuyou", "蜉蝣", 8.0, 20.0, "Ephemeroptera 体长 8–20mm（含尾丝可更长）"),
    ("ganshutiane", "甘薯天蛾", 40.0, 55.0, "Agrius convolvuli 体长 40–55mm，翅展 90–120mm"),
    ("ganwenyee", "干纹夜蛾", 14.0, 18.0, "Mythimna 类体长 14–18mm"),
    ("heiweiyechan", "黑尾叶蝉", 4.0, 6.0, "Nephotettix cincticeps 体长 4–6mm"),
    ("hongzonghuiyee", "红棕灰夜蛾", 14.0, 19.0, "Polia bombycina / 同类，体长 14–19mm"),
    ("huangchong", "蝗虫", 25.0, 50.0, "蝗虫（统称） 25–50mm；含若干常见科种"),
    ("huangyangjuanyeming", "黄杨绢野螟", 14.0, 18.0, "Cydalima perspectalis 体长 14–18mm，翅展 40–45mm"),
    ("jiachong", "甲虫", 5.0, 25.0, "甲虫（统称） 5–25mm；含步甲/叶甲/瓢虫等多科"),
    ("jinbanyee", "金斑夜蛾", 14.0, 18.0, "Diachrysia chrysitis 体长 14–18mm"),
    ("juanyee", "卷叶蛾", 8.0, 14.0, "Tortricidae 统称；体长 8–14mm，翅展 15–24mm"),
    ("liushangyee", "柳裳夜蛾", 28.0, 38.0, "Catocala 系 体长 28–38mm，翅展 70–90mm"),
    ("miandajuanyeming", "棉大卷叶螟", 11.0, 14.0, "Sylepta derogata 体长 11–14mm，翅展 30–35mm"),
    ("muxuyee", "苜蓿夜蛾", 14.0, 18.0, "Heliothis viriplaca 体长 14–18mm"),
    ("sanchadilaohu", "三叉地老虎", 18.0, 22.0, "Agrotis trifurca 类 体长 18–22mm"),
    ("wen", "蚊", 4.0, 8.0, "蚊（统称） 4–8mm；与 dawen「大蚊」区分"),
    ("xiaozaoqiaoe", "小造桥蛾", 12.0, 16.0, "Anomis flava 体长 12–16mm，翅展 28–32mm"),
    ("xishuai", "蟋蟀", 15.0, 25.0, "Gryllidae 统称 15–25mm"),
    ("xuanyouyee", "旋幽夜蛾", 12.0, 16.0, "Hadula trifolii 体长 12–16mm"),
    ("yanyee", "焰夜蛾", 12.0, 15.0, "Pyrrhia umbra 体长 12–15mm；与 yanqingchong「烟青虫」区分"),
    ("yinchichong", "隐翅虫", 5.0, 10.0, "Staphylinidae 统称 5–10mm"),
    ("ying", "蝇", 5.0, 10.0, "蝇（双翅目 统称） 5–10mm"),
    # 北京比赛害虫清单对齐（insect_info 缺 key；体长参考 insect_build_info / 常见文献）
    ("baixiansanwenyee", "白线散纹夜蛾", 14.0, 18.0, "北京比赛清单；体长约 14–18mm"),
    ("danjiantanyee", "淡剑贪夜蛾", 15.0, 20.0, "北京比赛清单；Mythimna separata 类 15–20mm"),
    ("kelaiyee", "客来夜蛾", 13.0, 17.0, "北京比赛清单；kelaiyee-ba 样本目录"),
    ("yinyawenyee", "隐丫纹夜蛾", 14.0, 18.0, "北京比赛清单；与 yinwenyee「银纹夜蛾」不同种"),
    ("fenyuanzuanyee", "粉缘钻夜蛾", 17.0, 23.0, "北京比赛清单；体长约 17–23mm"),
    ("xieyee", "谐夜蛾", 12.0, 16.0, "北京比赛清单；xieyee-ba 样本目录"),
    ("baidiananyeming", "白点暗野螟", 10.0, 14.0, "北京比赛清单；baidiananyeming 样本目录"),
    ("baimaizhanyee", "白脉粘夜蛾", 14.0, 18.0, "北京比赛清单 top3；cls-seg 叶目录待补"),
    ("lanlvqijiaoming", "榄绿歧角螟", 12.0, 16.0, "北京比赛清单；与 qijiaoming「歧角螟」不同种"),
    ("mianhehuanyeming", "棉褐环野螟", 10.0, 14.0, "北京比赛清单；mianhehuanyeming-ba 样本目录"),
    ("huaichie", "槐尺蛾", 18.0, 24.0, "北京比赛清单；C. sinica 类体长约 18–24mm"),
    ("simianmujinxingchie", "丝棉木金星尺蛾", 25.0, 35.0, "北京比赛清单；simianmujinxingchie-ba 样本目录"),
    ("renwenwudenge", "人纹污灯蛾", 17.0, 23.0, "北京比赛清单；体长约 17–23mm"),
    ("bujia", "步甲", 8.0, 20.0, "北京比赛清单；Carabidae 统称 8–20mm"),
    ("mifeng", "蜜蜂", 12.0, 16.0, "北京比赛清单；工蜂体长约 12–16mm"),
    ("piaochong", "瓢虫", 5.0, 8.0, "北京比赛清单；Coccinellidae 常见种 5–8mm"),
    ("chachichun", "茶翅蝽", 12.0, 17.0, "北京比赛清单；Halyomorpha halys 类 12–17mm"),
    ("zhuzhiyeyeming", "竹织叶野螟", 10.0, 14.0, "北京比赛清单 top3；cls-seg 叶目录待补"),
]


def _size_label(body_max: float) -> str:
    if body_max < 8.0:
        return "小虫"
    if body_max <= 25.0:
        return "中虫"
    return "大虫"


def _build_entry(pinyin: str, name_zh: str, bmin: float, bmax: float, remark_note: str) -> Dict[str, Any]:
    return {
        "body_length_mm": {
            "max_mm": float(bmax),
            "min_mm": float(bmin),
            "note": None,
            "raw": f"{bmin:g}–{bmax:g}",
        },
        "category": "其他昆虫",
        "data_dirs_raw": pinyin,
        "deliver_biong": False,
        "group": "其他昆虫",
        # 顶层 min/max 先与体长同步；随后 update_insect_info_imaging_extent_mm.py 会按物种类型修正
        "max_mm": float(bmax),
        "min_mm": float(bmin),
        "name_zh": name_zh,
        "patch_average_px": {
            "height_px": None,
            "raw": None,
            "width_px": None,
        },
        "pest_level": {
            "group_zh": "其他昆虫",
            "is_other_dataset_class": True,
            "label_zh": "其他昆虫",
            "national_list_class": None,
        },
        "pinyin": pinyin,
        "regions": [],
        "regions_raw": None,
        "remark": f"批量补齐（{remark_note}）；顶层 min/max 由成像跨度脚本后续修正",
        "sample_count": None,
        "sheet_row": None,
        "size_label": _size_label(bmax),
        "training_plan_reserve_row": False,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / "script" / "config" / "insect_info.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("root must be object")

    added: List[str] = []
    skipped: List[str] = []
    for pinyin, name_zh, bmin, bmax, note in ENTRIES:
        if pinyin in data:
            skipped.append(pinyin)
            continue
        data[pinyin] = _build_entry(pinyin, name_zh, bmin, bmax, note)
        added.append(pinyin)

    # 重新按 key 字典序写回，保证可读性
    sorted_data = {k: data[k] for k in sorted(data.keys())}
    path.write_text(json.dumps(sorted_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    print(f"added {len(added)}: {added}")
    if skipped:
        print(f"skipped (already exists) {len(skipped)}: {skipped}")


if __name__ == "__main__":
    main()
