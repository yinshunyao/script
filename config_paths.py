#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""``insect/script/config/`` 下统一配置文件路径（相对 ``script/`` 解析基准）。"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_DIR = _SCRIPT_DIR / "config"

INSECT_ALG_LAUNCHER_JSON = CONFIG_DIR / "insect_alg_all.json"
INSECT_ALG_ALL_JSON = INSECT_ALG_LAUNCHER_JSON
DEFAULT_INSECT_ALG_ALL_JSON = INSECT_ALG_ALL_JSON
INSECT_ALG_ALL_JSON_REL = "config/insect_alg_all.json"
INSECT_ALG_SHENGCHAN_JSON = CONFIG_DIR / "insect_alg_shengchan.json"
INSECT_ALG_OTHER_JSON = CONFIG_DIR / "insect_alg_other.json"
INSECT_INFO_JSON = CONFIG_DIR / "insect_info.json"

# ``insect_alg_all.json`` 顶层 ``run_model`` → 实际加载的配置文件（相对 ``config/``）
RUN_MODEL_PROFILE_FILES: dict[str, str] = {
    "baipai": "insect_alg_all.json",
    "shengchan": "insect_alg_shengchan.json",
    "other": "insect_alg_other.json",
}
DEFAULT_RUN_MODEL = "baipai"

RUN_MODEL_UI_LABELS: dict[str, str] = {
    "baipai": "摆拍",
    "shengchan": "生产",
    "other": "其他",
}
RUN_MODEL_CHOICES: tuple[str, ...] = tuple(RUN_MODEL_PROFILE_FILES.keys())

# 默认在 ``insect_alg_all.json`` 维护完整配置；shengchan/other 仅写场景差异项，
# 加载时以 all 为基线、profile 覆盖（见 ``compose_insect_alg_from_profile``）。
LAUNCHER_SHARED_TOP_KEYS: tuple[str, ...] = ("model_dir",)
_PROFILE_JSON_NAMES = frozenset(
    {
        "insect_alg_shengchan.json",
        "insect_alg_other.json",
    }
)


def resolve_insect_alg_all_path(path: str | Path | None = None) -> Path:
    """相对路径一律相对于 ``script/``（本模块所在目录）。"""
    if path is None:
        return INSECT_ALG_LAUNCHER_JSON
    p = Path(path)
    if not p.is_absolute():
        p = _SCRIPT_DIR / p
    return p


def is_insect_alg_launcher_path(path: str | Path) -> bool:
    """是否为带 ``run_model`` 开关的启动配置 ``insect_alg_all.json``。"""
    return resolve_insect_alg_all_path(path).resolve() == INSECT_ALG_LAUNCHER_JSON.resolve()


def read_run_model_profile(launcher_path: str | Path | None = None) -> str:
    """
    读取启动配置中的 ``run_model``（仅 ``insect_alg_all.json`` 有效）。

    非启动文件或字段缺失时返回 ``DEFAULT_RUN_MODEL``。
    """
    launcher = resolve_insect_alg_all_path(launcher_path)
    if not is_insect_alg_launcher_path(launcher):
        return DEFAULT_RUN_MODEL
    if not launcher.is_file():
        return DEFAULT_RUN_MODEL
    try:
        with open(launcher, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return DEFAULT_RUN_MODEL
    if not isinstance(data, dict):
        return DEFAULT_RUN_MODEL
    profile = str(data.get("run_model") or DEFAULT_RUN_MODEL).strip().lower()
    return profile or DEFAULT_RUN_MODEL


def normalize_run_model_key(run_model: str) -> str:
    """校验并规范化 ``run_model`` 取值（``baipai`` / ``shengchan`` / ``other``）。"""
    key = str(run_model or DEFAULT_RUN_MODEL).strip().lower()
    if key not in RUN_MODEL_PROFILE_FILES:
        known = ", ".join(RUN_MODEL_CHOICES)
        raise ValueError(f"run_model={run_model!r} 无效，可选: {known}")
    return key


def write_run_model_profile(
    run_model: str,
    launcher_path: str | Path | None = None,
) -> str:
    """
    将 ``run_model`` 写回 ``insect_alg_all.json`` 顶层字段（保留其余 JSON 内容）。

    返回规范化后的 profile 键。
    """
    key = normalize_run_model_key(run_model)
    launcher = resolve_insect_alg_all_path(launcher_path)
    if not is_insect_alg_launcher_path(launcher):
        raise ValueError(f"仅支持写入启动配置 {INSECT_ALG_LAUNCHER_JSON.name}")
    if not launcher.is_file():
        raise FileNotFoundError(f"未找到启动配置: {launcher}")
    with open(launcher, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"启动配置须为 JSON 对象: {launcher}")
    if str(data.get("run_model") or "").strip().lower() == key:
        return key
    data["run_model"] = key
    text = json.dumps(data, ensure_ascii=False, indent=2)
    if not text.endswith("\n"):
        text += "\n"
    with open(launcher, "w", encoding="utf-8") as f:
        f.write(text)
    return key


def resolve_run_model_profile_path(run_model: str) -> Path:
    """将 ``run_model`` 解析为 ``config/`` 下实际配置文件路径。"""
    key = str(run_model or DEFAULT_RUN_MODEL).strip().lower()
    filename = RUN_MODEL_PROFILE_FILES.get(key)
    if not filename:
        known = ", ".join(sorted(RUN_MODEL_PROFILE_FILES))
        raise ValueError(f"run_model={run_model!r} 无效，可选: {known}")
    return CONFIG_DIR / filename


def resolve_effective_insect_alg_path(path: str | Path | None = None) -> Path:
    """
    解析实际加载的算法配置路径。

    - 入口为 ``insect_alg_all.json``（或默认）时，按顶层 ``run_model`` 跳转；
    - 显式指定 ``insect_alg_shengchan.json`` / ``insect_alg_other.json`` 时原样使用。
    """
    launcher = resolve_insect_alg_all_path(path)
    if not is_insect_alg_launcher_path(launcher):
        return launcher
    profile = read_run_model_profile(launcher)
    return resolve_run_model_profile_path(profile)


def is_insect_alg_profile_path(path: str | Path) -> bool:
    """是否为 shengchan / other 场景配置文件（加载时以 all 为基线、profile 覆盖）。"""
    return Path(path).name in _PROFILE_JSON_NAMES


def load_insect_alg_launcher_dict() -> dict[str, Any]:
    """读取 ``insect_alg_all.json`` 启动配置（含 ``run_model`` 与公共 tier/model_dir）。"""
    if not INSECT_ALG_LAUNCHER_JSON.is_file():
        return {}
    with open(INSECT_ALG_LAUNCHER_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def get_predict_cfg(cfg: dict[str, Any] | None) -> dict[str, Any]:
    """返回 ``predict_cfg`` 块；缺失或非 dict 时返回空 dict。"""
    if not isinstance(cfg, dict):
        return {}
    block = cfg.get("predict_cfg")
    return block if isinstance(block, dict) else {}


def _predict_cfg_lookup_keys(key: str) -> tuple[str, ...]:
    if key == "cls_batch_size":
        return ("cls_batch_size", "cls_trt_batch_size", "trt_batch_size")
    return (key,)


def resolve_predict_cfg_value(
    key: str,
    cfg: dict[str, Any] | None = None,
    *,
    default: Any = None,
) -> Any:
    """
    读取推理配置项。

    查找顺序（每个 cfg 源内）：当前 dict 顶层同名键 → ``predict_cfg`` 内键。
    源顺序：``cfg`` 参数 → ``load_insect_alg_all`` 注入的全局配置（若已设置）。
    """
    from script.predict.model_trt import get_inference_global_cfg

    sources: list[dict[str, Any]] = []
    if isinstance(cfg, dict):
        sources.append(cfg)
    global_cfg = get_inference_global_cfg()
    if isinstance(global_cfg, dict) and global_cfg is not cfg:
        sources.append(global_cfg)

    lookup_keys = _predict_cfg_lookup_keys(key)
    for src in sources:
        for lk in lookup_keys:
            if lk in src:
                return src[lk]
        block = get_predict_cfg(src)
        for lk in lookup_keys:
            if lk in block:
                return block[lk]
    return default


DEFAULT_RUN_COUNT = 1


def resolve_run_count(
    cfg: dict[str, Any] | None = None,
    *,
    default: int = DEFAULT_RUN_COUNT,
) -> int:
    """
    推理工作进程数（``predict_cfg.run_count``）。

  未配置或非法时返回 ``default``（默认 1）；结果始终 ``>= 1``。
    """
    v = resolve_predict_cfg_value("run_count", cfg, default=default)
    try:
        n = int(v)
    except (TypeError, ValueError):
        n = int(default)
    return max(1, n)


DEFAULT_API_CONCURRENCY = 2


def resolve_api_concurrency(
    cfg: dict[str, Any] | None = None,
    *,
    default: int | None = None,
) -> int:
    """
    进程内 API/Gradio 可同时处理的图片数（``predict_cfg.api_concurrency``）。

    ``run_count=1`` 时由主进程线程池承担；``run_count>1`` 时线程池接收并发 HTTP，
    实际 GPU 推理仍由 ``run_count`` 个 worker 进程执行。

    未配置时默认 ``max(2, run_count)``；结果始终 ``>= 1``。
    """
    if default is None:
        default = max(DEFAULT_API_CONCURRENCY, resolve_run_count(cfg))
    v = resolve_predict_cfg_value("api_concurrency", cfg, default=default)
    try:
        n = int(v)
    except (TypeError, ValueError):
        n = int(default)
    return max(1, n)


def resolve_use_gpu_crop(
    cfg: dict[str, Any] | None = None,
    *,
    default: bool = False,
) -> bool:
    """
    是否在 CUDA 上用 GPU 做 crop/pad（``predict_cfg.use_gpu_crop``）。

    关闭或非 CUDA 环境时保持原有 CPU numpy/cv2 路径。
    """
    v = resolve_predict_cfg_value("use_gpu_crop", cfg, default=default)
    return bool(v)


DEFAULT_DETECT_SEG_BATCH_SIZE = 32


def resolve_clip_profiles_enable(
    cfg: dict[str, Any] | None = None,
    *,
    global_cfg: dict[str, Any] | None = None,
) -> bool:
    """是否启用 ``clip_profiles`` 多尺度滑窗（``predict_cfg.clip_profiles_enable``）。"""
    for src in (cfg, global_cfg):
        if not isinstance(src, dict):
            continue
        for layer in (src, get_predict_cfg(src)):
            if not layer:
                continue
            if "clip_profiles_enable" in layer:
                return bool(layer["clip_profiles_enable"])
    return True


def resolve_detect_seg_batch_size(
    cfg: dict[str, Any] | None = None,
    *,
    global_cfg: dict[str, Any] | None = None,
) -> int:
    """
    detect/seg 滑窗切片 YOLO batch 上限。

    配置在 ``predict_cfg.detect_seg_batch_size``；未配置默认 32。
    与 ``clip_profiles_enable``（是否多尺度滑窗）独立：单尺度 ``clip_size`` 下同样可 batch 推理。
    """
    for src in (cfg, global_cfg):
        if not isinstance(src, dict):
            continue
        for layer in (src, get_predict_cfg(src)):
            if not layer:
                continue
            v = layer.get("detect_seg_batch_size")
            if v is None:
                continue
            try:
                n = int(v)
                if n > 0:
                    return n
            except (TypeError, ValueError):
                continue
    return DEFAULT_DETECT_SEG_BATCH_SIZE


def _merge_dict_fill_missing(target: dict[str, Any], base: dict[str, Any]) -> None:
    """就地合并：``target`` 已有键保留；缺失键及子 dict 从 ``base`` 补齐。"""
    for key, base_val in base.items():
        if key not in target:
            target[key] = deepcopy(base_val)
            continue
        target_val = target[key]
        if isinstance(target_val, dict) and isinstance(base_val, dict):
            _merge_dict_fill_missing(target_val, base_val)


def _merge_dict_overlay(base: dict[str, Any], overlay: dict[str, Any]) -> None:
    """
    就地合并：``overlay`` 中已配置的键覆盖 ``base``；双方均为 dict 时递归。

    用于 shengchan/other 场景：以 ``insect_alg_all.json`` 为完整基线，profile 只覆盖差异项。
    """
    for key, overlay_val in overlay.items():
        if str(key).endswith("_说明"):
            continue
        if key not in base:
            base[key] = deepcopy(overlay_val)
            continue
        base_val = base[key]
        if isinstance(base_val, dict) and isinstance(overlay_val, dict):
            _merge_dict_overlay(base_val, overlay_val)
        else:
            base[key] = deepcopy(overlay_val)


def compose_insect_alg_from_profile(
    launcher: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    """
    以 ``insect_alg_all.json`` 为完整基线，用场景 profile（shengchan/other）覆盖有效配置。

    profile 中显式出现的键（含嵌套 ``prepare`` / ``predict_cfg`` / ``postprocess`` /
    ``models`` 等）优先；未配置项保留 launcher 原值。
    """
    if not isinstance(launcher, dict):
        return deepcopy(profile) if isinstance(profile, dict) else {}
    out = deepcopy(launcher)
    if isinstance(profile, dict):
        _merge_dict_overlay(out, profile)
    return out


def apply_launcher_shared_overlay(profile: dict[str, Any]) -> dict[str, Any]:
    """向后兼容：等价于 ``compose_insect_alg_from_profile(launcher, profile)``。"""
    return compose_insect_alg_from_profile(load_insect_alg_launcher_dict(), profile)


def apply_launcher_models_overlay(profile: dict[str, Any]) -> dict[str, Any]:
    """向后兼容：等价于 ``compose_insect_alg_from_profile(launcher, profile)``。"""
    return compose_insect_alg_from_profile(load_insect_alg_launcher_dict(), profile)


def apply_launcher_models_out_overlay(profile: dict[str, Any]) -> dict[str, Any]:
    """向后兼容别名；等价于 ``compose_insect_alg_from_profile(launcher, profile)``。"""
    return compose_insect_alg_from_profile(load_insect_alg_launcher_dict(), profile)


def apply_launcher_trt_overlay(profile: dict[str, Any]) -> dict[str, Any]:
    """向后兼容：等价于 ``compose_insect_alg_from_profile(launcher, profile)``。"""
    return compose_insect_alg_from_profile(load_insect_alg_launcher_dict(), profile)
