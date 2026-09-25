#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
按 compare 目录中的主文件名（不含后缀），从源目录拷贝同名文件到目标目录。

compare 中不存在的不处理；compare 中存在、源目录不存在的也不处理。
"""
from __future__ import annotations

import logging
import shutil
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _iter_files(root: Path):
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        name = path.name
        if not name or name.startswith(".") or name.startswith("._"):
            continue
        yield path


def _files_by_stem(root: Path) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in _iter_files(root):
        grouped[path.stem].append(path)
    return grouped


def copy_by_compare_stem(src: Path, compare: Path, dst: Path) -> dict[str, int]:
    src = src.resolve()
    compare = compare.resolve()
    dst = dst.resolve()

    if not src.is_dir():
        raise NotADirectoryError(f"源目录不存在或不是目录: {src}")
    if not compare.is_dir():
        raise NotADirectoryError(f"compare 目录不存在或不是目录: {compare}")

    dst.mkdir(parents=True, exist_ok=True)

    compare_stems = {path.stem for path in _iter_files(compare)}
    src_by_stem = _files_by_stem(src)

    stats = {
        "compare_stems": len(compare_stems),
        "copied": 0,
        "missing_in_src": 0,
        "skipped_src_only": 0,
        "failed": 0,
    }

    for stem in sorted(src_by_stem):
        if stem not in compare_stems:
            stats["skipped_src_only"] += len(src_by_stem[stem])

    for stem in sorted(compare_stems):
        src_files = src_by_stem.get(stem)
        if not src_files:
            stats["missing_in_src"] += 1
            logger.info("compare 有、源目录无，跳过: %s", stem)
            continue
        for src_file in src_files:
            dst_file = dst / src_file.name
            try:
                shutil.copy2(src_file, dst_file)
                logger.info("已拷贝: %s", src_file.name)
                stats["copied"] += 1
            except OSError as exc:
                stats["failed"] += 1
                logger.error("拷贝失败 %s: %s", src_file.name, exc)

    logger.info(
        "完成: compare主文件名=%d, 已拷贝=%d, compare有源无=%d, 源有compare无(未拷贝文件)=%d, 失败=%d",
        stats["compare_stems"],
        stats["copied"],
        stats["missing_in_src"],
        stats["skipped_src_only"],
        stats["failed"],
    )
    return stats


if __name__ == "__main__":
    # 源目录
    src = "/Volumes/shunyao-h1/比赛数据/2026辛集正式比赛/辛集市田间比赛"

    # compare
    compare = "/Volumes/shunyao-h1/比赛数据/2026辛集正式比赛/河北辛集识别结果呈现（比昂）/比昂田间识别/annotated"

    # 目标  从源目录中拷贝， 与compare目录中文件名相同的（不管后缀）文件，都复制到目标目录
    # compare中不存在的不处理； compare中存在， 源目录不存在的也不会处理
    dst = "/Volumes/shunyao-h1/比赛数据/2026辛集正式比赛/田间比赛对比"

    copy_by_compare_stem(Path(src), Path(compare), Path(dst))
