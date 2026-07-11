#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""TensorRT engine 路径解析：NVIDIA CUDA 环境下优先加载 ``trt`` 配置的 .engine。"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_MODEL_DIR_KEYS = frozenset(
    {"model", "model1", "model2", "model3", "trt", "trt1", "trt2", "trt3"}
)

_INFERENCE_GLOBAL_CFG: dict[str, Any] = {}


def set_inference_global_cfg(cfg: dict[str, Any] | None) -> None:
    """由 ``load_insect_alg_all`` / ``InsectPredictAll`` 注入顶层配置（含 ``predict_cfg``）。"""
    global _INFERENCE_GLOBAL_CFG
    _INFERENCE_GLOBAL_CFG = dict(cfg or {})


def get_inference_global_cfg() -> dict[str, Any]:
    """``load_insect_alg_all`` 注入的全局算法配置副本。"""
    return dict(_INFERENCE_GLOBAL_CFG)


def resolve_trt_switch(cfg: dict[str, Any] | None = None) -> bool:
    """
    是否启用 TensorRT engine 推理。

    优先级：当前节点顶层 ``trt_switch`` > ``predict_cfg.trt_switch`` > 全局同路径；缺省 ``True``。
    """
    from script.config_paths import resolve_predict_cfg_value

    return bool(resolve_predict_cfg_value("trt_switch", cfg, default=True))


def is_nvidia_cuda_available() -> bool:
    """当前进程是否可用 NVIDIA CUDA（用于判断是否尝试 TensorRT engine）。"""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def is_trt_engine_path(model_path: str) -> bool:
    """路径是否为 TensorRT engine（``.engine``）。"""
    return str(model_path or "").strip().lower().endswith(".engine")


@lru_cache(maxsize=64)
def read_trt_engine_max_batch(engine_path: str) -> int:
    """
    从 Ultralytics 导出的 ``.engine`` 文件头 metadata 读取最大 batch。

    固定 batch engine（``dynamic=False, batch=1``）返回 1；动态 batch 导出时返回配置的 max batch。
    """
    path = Path(engine_path).expanduser()
    if not path.is_file() or not is_trt_engine_path(str(path)):
        return 1
    try:
        with open(path, "rb") as f:
            meta_len = int.from_bytes(f.read(4), byteorder="little")
            if meta_len <= 0 or meta_len > 10_000_000:
                return 1
            metadata = json.loads(f.read(meta_len).decode("utf-8"))
        if isinstance(metadata, dict):
            batch = metadata.get("batch")
            if batch is not None:
                return max(1, int(batch))
    except Exception as exc:
        log.debug("读取 engine metadata batch 失败 %s: %s", path, exc)
    return 1


def effective_trt_batch_size(engine_path: str, requested: int) -> int:
    """将配置的 batch 限制在 engine 支持的最大 batch 内。"""
    req = max(1, int(requested or 1))
    max_batch = read_trt_engine_max_batch(engine_path)
    return max(1, min(req, max_batch))


def _resolve_trt_file(trt_raw: str, pt: str) -> Path | None:
    """
    将 ``trt`` 配置解析为 engine 文件路径。

    支持仅写文件名（如 ``daofeishi-detect-0415.engine``）：
    - 已是可访问的绝对/相对路径则直接使用；
    - 否则优先在 ``model`` 同目录查找；
    - ``load_insect_alg_all`` 在配置 ``model_dir`` 时会预先把 trt 改写为 ``<model_dir>/<文件名>``。
    """
    raw = str(trt_raw or "").strip()
    if not raw:
        return None

    direct = Path(raw).expanduser()
    if direct.is_file():
        return direct.resolve()

    name = direct.name
    if pt:
        beside = Path(pt).expanduser().resolve().parent / name
        if beside.is_file():
            return beside.resolve()

    # 可能已被 model_dir 改写为绝对路径但文件暂不存在
    if direct.is_absolute():
        return direct
    return Path(pt).expanduser().resolve().parent / name if pt else direct


def resolve_inference_model_path(
    cfg: dict[str, Any] | None,
    *,
    model_path: str | None = None,
    pt_key: str = "model",
    trt_key: str = "trt",
    log_label: str = "",
    quiet: bool = False,
) -> str:
    """
    解析实际推理权重路径：CUDA 可用且 ``trt`` 文件存在时用 engine，否则用 ``model``（pt）。

    ``cfg`` 可为嵌套分类节点或根模型配置；``model_path`` 显式传入时作为 pt 回退路径。
    """
    cfg = cfg or {}
    pt = str(model_path or cfg.get(pt_key) or "").strip()
    if not pt:
        return ""

    if not resolve_trt_switch(cfg):
        return pt

    if not is_nvidia_cuda_available():
        return pt

    trt_raw = str(cfg.get(trt_key) or "").strip()
    if not trt_raw:
        return pt

    trt_path = _resolve_trt_file(trt_raw, pt)
    if trt_path is not None and trt_path.is_file():
        resolved = str(trt_path.resolve())
        if not quiet:
            prefix = f"{log_label} " if log_label else ""
            log.info(
                "%sNVIDIA CUDA: 使用 TensorRT engine %s (pt=%s)",
                prefix,
                resolved,
                pt,
            )
        return resolved

    if not quiet:
        prefix = f"{log_label} " if log_label else ""
        log.warning("%strt 不存在 (%s)，回退 pt: %s", prefix, trt_raw, pt)
    return pt


def merge_trt_overlay(target: dict[str, Any], base: dict[str, Any]) -> None:
    """
    就地合并：``target`` 缺少 ``trt`` 时从 ``base`` 同层拷贝；并递归子 dict。

    用于 shengchan/other 场景配置继承 ``insect_alg_all.json`` 中的 trt 路径。
    """
    base_trt = base.get("trt")
    if isinstance(base_trt, str) and base_trt.strip():
        cur = target.get("trt")
        if not (isinstance(cur, str) and cur.strip()):
            target["trt"] = base_trt

    for key, tv in target.items():
        if not isinstance(tv, dict):
            continue
        bv = base.get(key)
        if isinstance(bv, dict):
            merge_trt_overlay(tv, bv)


def rewrite_model_dir_paths(node: Any, model_dir: str) -> None:
    """
    改写 model/trt 等为 ``<model_dir>/<文件名>``。

    ``trt`` 在 JSON 中只需写 engine 文件名（如 ``seg-3.12.1.engine``），与 ``model`` 一样按
    ``model_dir`` 解析；若已写绝对路径则仅取文件名再拼接。
    """
    if isinstance(node, dict):
        for k, v in node.items():
            if k in _MODEL_DIR_KEYS and isinstance(v, str) and v.strip():
                node[k] = str(Path(model_dir) / Path(v).name)
            else:
                rewrite_model_dir_paths(v, model_dir)
    elif isinstance(node, list):
        for item in node:
            rewrite_model_dir_paths(item, model_dir)
