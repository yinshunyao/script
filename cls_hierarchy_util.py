#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Detail  : 分级类别树加载：默认 ``insect/script/cls_merge.py``，亦支持独立 JSON 文件。

from __future__ import annotations

import importlib.util
import json
import logging
from pathlib import Path
from typing import Any

CLS_MERGE_PY_FILENAME = "cls_merge.py"
_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_CLS_MERGE_PATH = (_SCRIPT_DIR / CLS_MERGE_PY_FILENAME).resolve()


def default_cls_merge_py_path() -> Path:
    return _DEFAULT_CLS_MERGE_PATH


def load_cls_merge_dict(*, path: Path | None = None) -> dict[str, Any]:
    """从 ``cls_merge.py`` 读取 ``cls_merge`` 字典。"""
    p = (path or _DEFAULT_CLS_MERGE_PATH).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"cls_merge.py 不存在: {p}")
    spec = importlib.util.spec_from_file_location("_cls_merge_loaded", p)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块: {p}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    raw = getattr(mod, "cls_merge", None)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{p} 中 cls_merge 须为非空 dict")
    return raw


def resolve_cls_hierarchy_path(
    raw: str,
    *,
    script_dir: Path | None = None,
    insect_script_dir: Path | None = None,
) -> Path:
    """
    解析分级配置路径。

    - **绝对路径**：必须存在。
    - 文件名为 ``cls_merge.py``（或裸名 ``cls_merge``）：优先 ``insect/script/cls_merge.py``。
    - **其它相对路径**：相对 ``script_dir``（默认 ``insect/script``）。
    """
    s = (raw or "").strip()
    if not s:
        raise ValueError("分级配置路径不能为空")
    base = (script_dir or _SCRIPT_DIR).resolve()
    insect_dir = (insect_script_dir or _SCRIPT_DIR).resolve()
    cand = Path(s).expanduser()
    if cand.is_absolute():
        p = cand.resolve()
        if not p.is_file():
            raise FileNotFoundError(f"分级配置不存在: {p}")
        return p

    candidates: list[Path] = []
    if cand.name in (CLS_MERGE_PY_FILENAME, "cls_merge"):
        candidates.append((insect_dir / CLS_MERGE_PY_FILENAME).resolve())
    candidates.append((base / cand).resolve())

    tried = [str(x) for x in candidates]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"分级配置不存在。已尝试: {tried}（cls_merge.py 优先 insect/script/）"
    )


def load_hierarchy_dict_at_path(path: Path) -> dict[str, Any]:
    """按扩展名加载分级树：``.py`` 读 ``cls_merge``，否则按 JSON。"""
    p = path.resolve()
    if p.suffix == ".py":
        return load_cls_merge_dict(path=p)
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as e:
        raise ValueError(f"读取分级 JSON 失败 {p}: {e}") from e
    if not isinstance(obj, dict) or not obj:
        raise ValueError(f"分级 JSON 根须为非空 dict: {p}")
    return obj


def normalize_cls_merge_hierarchy_raw(raw: dict[str, Any]) -> dict[str, Any]:
    """
    将 ``cls_merge`` 转为树形分级 dict（顶层 value 均为非空 dict）。

    顶层标量（单输出类）归一为 ``{key: {key: 采样数}}``；list 转为 ``{子key: 1}`` dict。
    """
    out: dict[str, Any] = {}
    for group_key, v in raw.items():
        if not isinstance(group_key, str) or not str(group_key).strip():
            continue
        gk = str(group_key).strip()
        if v is None:
            continue
        if isinstance(v, bool):
            raise ValueError(f"cls_merge[{gk!r}]: 禁止 bool")
        if isinstance(v, (int, float)):
            out[gk] = {gk: float(v)}
            continue
        if isinstance(v, dict):
            if not v:
                raise ValueError(f"cls_merge[{gk!r}]: 空 dict")
            out[gk] = v
            continue
        if isinstance(v, (list, tuple)):
            members = {
                str(x).strip(): 1.0
                for x in v
                if isinstance(x, str) and str(x).strip()
            }
            out[gk] = members if members else {gk: 1.0}
            continue
        raise ValueError(
            f"cls_merge[{gk!r}]: value 须为数字 / dict / list，当前: {type(v).__name__}"
        )
    return out


def load_hierarchy_dict(raw_path: str, *, script_dir: Path | None = None) -> tuple[Path, dict[str, Any]]:
    """解析路径并加载分级 dict；``cls_merge.py`` 会先归一化顶层标量。"""
    p = resolve_cls_hierarchy_path(raw_path, script_dir=script_dir)
    obj = load_hierarchy_dict_at_path(p)
    if p.suffix == ".py" and p.name == CLS_MERGE_PY_FILENAME:
        obj = normalize_cls_merge_hierarchy_raw(obj)
    logging.info("已加载分级配置: %s", p)
    return p, obj
