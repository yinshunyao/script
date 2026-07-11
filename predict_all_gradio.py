#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Detail  : 统一虫情推理（predict_all.py）的 Gradio 本地测试服务。
#           上传图片 + 算法开关/参数配置（基于 insect_alg_all.json），复用推理与绘图能力。
#           设计见 insect/doc/02-dr/【推理服务】Gradio图片测试与算法参数配置.md

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import tempfile
import threading
import time
import uuid
from collections import Counter, OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Any

import cv2
import gradio as gr
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

_FILE = Path(__file__).resolve()
_ROOT = _FILE.parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from script.config_paths import (
    DEFAULT_RUN_MODEL,
    RUN_MODEL_CHOICES,
    RUN_MODEL_UI_LABELS,
    normalize_run_model_key,
    read_run_model_profile,
    resolve_effective_insect_alg_path,
    resolve_insect_alg_all_path,
    resolve_api_concurrency,
    resolve_run_count,
    resolve_use_gpu_crop,
    write_run_model_profile,
)
from script.predict_all import (
    InsectPredictAll,
    draw_results,
    load_image_bgr_from_ref,
    load_insect_alg_all,
)
from script.predict_worker_pool import InferenceProcessPool, configure_stdio_logging

logger = logging.getLogger(__name__)

# 进程重启参数（``__main__`` 启动 uvicorn 时设置；用于 run_model 切换后 execv）
_RESTART_ARGV: list[str] | None = None
_RESTART_DELAY_SEC = 1.0
_SHUTDOWN_CALLBACKS: list[Callable[[], None]] = []
_HEALTH_READY_PATH = "/health/ready"
_GATE_POLL_INTERVAL_MS = 1000


def _service_gate_html() -> str:
    """服务就绪遮罩 DOM（不含 ``<script>``；逻辑见 ``_SERVICE_GATE_JS_ON_LOAD``）。"""
    return """
<div id="insect-service-gate" style="
  display:none;position:fixed;inset:0;z-index:99999;align-items:center;
  justify-content:center;background:rgba(0,0,0,0.45);">
  <div style="background:#fff;border-radius:12px;padding:28px 36px;max-width:420px;
    box-shadow:0 8px 32px rgba(0,0,0,0.18);text-align:center;">
    <div id="insect-gate-title" style="font-size:20px;font-weight:600;margin-bottom:8px;">正在加载模型</div>
    <div id="insect-gate-hint" style="color:#555;margin-bottom:12px;">推理模型正在启动，请稍候…</div>
    <div id="insect-gate-status" style="color:#888;font-size:14px;">连接服务…</div>
  </div>
</div>
"""


_SERVICE_GATE_JS_ON_LOAD = f"""
(function() {{
  const HEALTH_URL = (window.location.origin || "") + "{_HEALTH_READY_PATH}";
  const POLL_MS = {_GATE_POLL_INTERVAL_MS};

  function gateEl() {{ return document.getElementById("insect-service-gate"); }}
  function statusEl() {{ return document.getElementById("insect-gate-status"); }}
  function titleEl() {{ return document.getElementById("insect-gate-title"); }}
  function hintEl() {{ return document.getElementById("insect-gate-hint"); }}

  window.insectSetPageInteractive = function(enabled) {{
    const panel = document.getElementById("insect-main-panel");
    if (panel) {{
      panel.style.pointerEvents = enabled ? "" : "none";
      panel.style.opacity = enabled ? "" : "0.55";
    }}
    document.querySelectorAll(
      "#insect-run-btn button, #insect-run-btn, #insect-restart-btn button, #insect-restart-btn"
    ).forEach((el) => {{
      if ("disabled" in el) el.disabled = !enabled;
    }});
  }};

  window.insectShowGate = function(opts) {{
    opts = opts || {{}};
    const gate = gateEl();
    if (!gate) return;
    if (titleEl()) titleEl().textContent = opts.restarting ? "服务重启中" : "正在加载模型";
    if (hintEl()) {{
      if (!opts.restarting) {{
        hintEl().textContent = "推理模型正在启动，请稍候…";
      }} else if (opts.modeUnchanged) {{
        hintEl().textContent = "后台正在重启并重新加载模型（运行模式不变）…";
      }} else {{
        hintEl().textContent = "已切换运行模式，后台正在按新配置重启并加载模型…";
      }}
    }}
    if (statusEl()) statusEl().textContent = "连接服务…";
    gate.style.display = "flex";
    window.insectSetPageInteractive(false);
  }};

  window.insectHideGate = function() {{
    const gate = gateEl();
    if (gate) gate.style.display = "none";
    window.insectSetPageInteractive(true);
  }};

  window.insectFetchReady = async function() {{
    const resp = await fetch(HEALTH_URL, {{ cache: "no-store" }});
    if (!resp.ok) return null;
    const data = await resp.json();
    return data && data.ready ? data : null;
  }};

  window.insectStartGatePoll = function(opts) {{
    opts = opts || {{}};
    if (window.__insectGateTimer) {{
      clearInterval(window.__insectGateTimer);
      window.__insectGateTimer = null;
    }}
    window.insectShowGate(opts);

    const tick = async () => {{
      try {{
        const data = await window.insectFetchReady();
        if (data) {{
          if (window.__insectGateTimer) {{
            clearInterval(window.__insectGateTimer);
            window.__insectGateTimer = null;
          }}
          if (statusEl()) statusEl().textContent = "模型已就绪，正在恢复页面…";
          window.__insectGatePolling = false;
          if (opts.reloadOnReady) {{
            window.location.reload();
            return;
          }}
          window.insectHideGate();
          return;
        }}
        if (statusEl()) statusEl().textContent = "等待服务重启并加载模型…";
      }} catch (e) {{
        if (statusEl()) statusEl().textContent = "等待服务重启并加载模型…";
      }}
    }};

    window.__insectGatePolling = true;
    tick();
    window.__insectGateTimer = setInterval(tick, POLL_MS);
  }};

  window.insectCheckOnLoad = async function() {{
    try {{
      const data = await window.insectFetchReady();
      if (data) {{
        window.insectHideGate();
        return;
      }}
    }} catch (e) {{}}
    window.insectStartGatePoll({{ restarting: false, reloadOnReady: false }});
  }};

  window.insectSetPageInteractive(false);
}})();
"""


_PAGE_LOAD_GATE_JS = """
() => {
  if (window.insectCheckOnLoad) window.insectCheckOnLoad();
}
"""

_MODE_CHANGE_POLL_JS = """
(trigger) => {
  if (trigger === "restart" && window.insectStartGatePoll) {
    window.insectStartGatePoll({ restarting: true, reloadOnReady: true });
  } else if (trigger === "restart_same" && window.insectStartGatePoll) {
    window.insectStartGatePoll({
      restarting: true,
      modeUnchanged: true,
      reloadOnReady: true,
    });
  }
  return [];
}
"""


def _normalize_config_value(obj: Any) -> Any:
    """归一化配置值，避免 Gradio Number 与 JSON 整数/浮点差异导致误判。"""
    if isinstance(obj, dict):
        return {str(k): _normalize_config_value(v) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        return [_normalize_config_value(v) for v in obj]
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        rounded = round(obj, 6)
        if rounded == int(rounded):
            return int(rounded)
        return rounded
    return obj


def _configs_equivalent(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return _normalize_config_value(a) == _normalize_config_value(b)


def _noop_gate_poll_trigger(trigger: str) -> str:
    """占位：实际轮询由 ``change().then(js=...)`` 在前端触发。"""
    return trigger


def register_server_shutdown(callback: Callable[[], None]) -> None:
    """登记 run_model 切换重启前需执行的资源释放回调（如推理子进程、模型缓存）。"""
    _SHUTDOWN_CALLBACKS.append(callback)


def _run_server_shutdown() -> None:
    """重启前释放 GPU/子进程；``os.execv`` 不会触发 ``atexit``，必须显式调用。"""
    for cb in reversed(_SHUTDOWN_CALLBACKS):
        try:
            cb()
        except Exception:
            logger.exception("服务关闭回调执行失败")


def configure_process_logging(level: int = logging.INFO) -> None:
    """
    配置进程日志。

    若 ``stdout``/``stderr`` 被重定向到同一文件（``> web.log 2>&1``），只挂载一个
    handler，避免每条日志打印两遍。
    """
    configure_stdio_logging(level)
    # uvicorn 访问/启动日志与业务日志同级可见
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).setLevel(level)


def configure_server_restart_argv(argv: list[str] | None = None) -> None:
    """登记服务重启时使用的 argv（默认 ``[python, predict_all_gradio.py, ...]``）。"""
    global _RESTART_ARGV
    _RESTART_ARGV = argv


def _restart_server_process() -> None:
    """释放推理资源后替换当前进程，重新加载配置与模型（Gradio + FastAPI 全量重启）。"""
    argv = _RESTART_ARGV or [sys.executable, *sys.argv]
    logger.info(
        "正在关闭推理子进程/模型缓存并重启 API 服务: %s",
        " ".join(argv),
    )
    _run_server_shutdown()
    os.execv(argv[0], argv)


def _is_model_load_error(exc: BaseException) -> bool:
    """判断异常是否由模型权重缺失/无法加载引起（用于失败后丢弃管线缓存）。"""
    if isinstance(exc, FileNotFoundError):
        return True
    if isinstance(exc, OSError) and getattr(exc, "errno", None) in (2,):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        if "no such file" in msg or "cannot load" in msg or "找不到" in msg:
            return True
    cause = exc.__cause__
    if cause is not None and cause is not exc and _is_model_load_error(cause):
        return True
    context = exc.__context__
    if context is not None and context is not exc and _is_model_load_error(context):
        return True
    return False


def _run_model_choice_label(key: str) -> str:
    k = str(key or DEFAULT_RUN_MODEL).strip().lower()
    zh = RUN_MODEL_UI_LABELS.get(k, k)
    return f"{zh} ({k})"


def _run_model_status_md(
    run_model: str,
    *,
    config_path: Path,
    effective_path: Path,
) -> str:
    key = str(run_model or "").strip().lower()
    zh = RUN_MODEL_UI_LABELS.get(key, key)
    return (
        f"**运行模式**：{zh}（`run_model={key}`）  \n"
        f"启动配置：`{config_path}`  \n"
        f"生效配置：`{effective_path}`  \n"
        f"切换模式下拉框会写回 `{config_path.name}` 并**自动重启后台服务**；"
        f"也可点击「重启服务」在不改运行模式的情况下重新加载模型。"
        f"重启期间页面会显示等待遮罩，就绪后自动恢复。"
    )

# 设备下拉项 → InsectPredictAll(device=...) 实参
_DEVICE_CHOICES = ["自动", "cpu", "cuda:0", "mps"]


def _warn_gradio_drag_drop_version() -> None:
    """Gradio 5.0–5.28 拖拽替换已有图片会新开浏览器标签，5.29+ 已修复。"""
    try:
        ver = gr.__version__.split("+")[0].strip()
        parts: list[int] = []
        for token in ver.split(".")[:3]:
            try:
                parts.append(int(token))
            except ValueError:
                parts.append(0)
        while len(parts) < 3:
            parts.append(0)
        if tuple(parts) < (5, 29, 0):
            logger.warning(
                "当前 gradio=%s：拖拽替换图片可能新开浏览器标签页。"
                "请执行 pip install 'gradio>=5.31.0,<6' 后重启；"
                "或改用页面上「选择/替换图片」按钮。",
                gr.__version__,
            )
    except Exception:  # pragma: no cover
        pass


def _device_arg(label: str | None) -> str | None:
    if not label or label == "自动":
        return None
    return str(label)


def _log_infer_device(configured: str | None) -> None:
    """启动时打印推理设备配置与 torch CUDA 探测结果。"""
    try:
        import torch
    except ImportError:
        logger.warning("[推理设备] 无法 import torch，跳过设备探测")
        return

    cuda_ok = torch.cuda.is_available()
    if configured:
        logger.info(
            "[推理设备] 配置=%s | torch.cuda.is_available()=%s",
            configured,
            cuda_ok,
        )
        if str(configured).startswith("cuda") and not cuda_ok:
            logger.warning(
                "[推理设备] 已指定 GPU 但 torch 未检测到 CUDA，子模型可能回退 CPU"
            )
        elif cuda_ok:
            logger.info("[推理设备] GPU: %s", torch.cuda.get_device_name(0))
        return

    if cuda_ok:
        logger.info(
            "[推理设备] 配置=自动 | 检测到 CUDA: %s（API/Gradio 未指定设备时各子模型自动选用）",
            torch.cuda.get_device_name(0),
        )
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("[推理设备] 配置=自动 | 未检测到 CUDA，将尝试 MPS")
    else:
        logger.warning("[推理设备] 配置=自动 | 未检测到 CUDA/MPS，将使用 CPU（推理较慢）")


def _log_predict_perf_config(cfg: dict[str, Any]) -> None:
    """启动时打印影响 GPU 利用率的 predict_cfg 关键项。"""
    pc = cfg.get("predict_cfg") if isinstance(cfg.get("predict_cfg"), dict) else {}
    logger.info(
        "[推理性能] run_count=%s api_concurrency=%s use_gpu_crop=%s "
        "detect_seg_batch_size=%s cls_batch_size=%s parallel_detect_seg=%s "
        "clip_profiles_enable=%s trt_switch=%s",
        resolve_run_count(cfg),
        resolve_api_concurrency(cfg),
        resolve_use_gpu_crop(cfg),
        pc.get("detect_seg_batch_size", 32),
        pc.get("cls_batch_size", 32),
        pc.get("parallel_detect_seg", True),
        pc.get("clip_profiles_enable", True),
        pc.get("trt_switch", False),
    )


def _device_label(device: str | None) -> str:
    return device if device else "自动"


def _load_image_rgb_from_input(image: np.ndarray | str | Path | None) -> np.ndarray | None:
    """
    从 Gradio ``Image`` 入参加载 RGB 图。

    页面输入使用 ``type=\"filepath\"``，上传/替换时只传临时路径，避免大图在
    每次替换时都走 numpy 序列化（易失败并表现为「替换后变空」）。
    """
    if image is None:
        return None
    if isinstance(image, np.ndarray):
        if image.size == 0:
            return None
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image
    path = Path(str(image).strip())
    if not path.is_file():
        return None
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None or bgr.size == 0:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _insect_api_error(msg: str) -> JSONResponse:
    """与历史 Flask 虫情接口一致：HTTP 200 + body 内 code=500。"""
    return JSONResponse({"code": 500, "msg": msg})


# 标注结果图临时目录（避免把整张高分辨率图以 numpy 内联回传导致前端卡住）
_OUTPUT_DIR = Path(tempfile.gettempdir()) / "insect_gradio_outputs"


# --------------------------------------------------------------------------- #
#  生效配置构建辅助
# --------------------------------------------------------------------------- #


def find_primary_cls_path(root_cfg: dict[str, Any]) -> list[str] | None:
    """
    在根模型配置中递归查找第一个分类节点 ``out.<key>.models.cls``，
    返回从根模型配置起到该 cls 字典的键路径；找不到返回 None。
    """
    out = root_cfg.get("out")
    if not isinstance(out, dict):
        return None
    for key, entry in out.items():
        if not isinstance(entry, dict):
            continue
        models = entry.get("models")
        if isinstance(models, dict) and isinstance(models.get("cls"), dict):
            return ["out", key, "models", "cls"]
    return None


def _dig(node: dict[str, Any], path: list[str]) -> dict[str, Any] | None:
    cur: Any = node
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur if isinstance(cur, dict) else None


def _leaf_class_items(cls_cfg: dict[str, Any]) -> list[tuple[str, str, bool]]:
    """
    分类节点 ``out`` 下可启停的叶子类列表。

    返回 [(类名, 标签文本, 默认是否启用)]；跳过 ``out`` 中非 dict 的杂项键
    （如配置里误写到 out 同级的 ``cls_conf`` 数值）。
    """
    out = cls_cfg.get("out")
    items: list[tuple[str, str, bool]] = []
    if not isinstance(out, dict):
        return items
    for name, entry in out.items():
        if not isinstance(entry, dict):
            continue
        cn = str(entry.get("cn_name") or "").strip()
        # 展示用半角分隔符；CheckboxGroup 的 value 使用类名（见 build_ui），避免全角「｜」与前端回传不一致
        label = f"{name} | {cn}" if cn else str(name)
        raw_enable = entry.get("enable", True)
        enabled = True if raw_enable is None else bool(raw_enable)
        items.append((str(name), label, enabled))
    return items


# --------------------------------------------------------------------------- #
#  控件规格：构建期登记，运行期按序回写到生效配置
# --------------------------------------------------------------------------- #


class _Spec:
    """单个输入控件 → 配置路径的映射规格。"""

    def __init__(self, kind: str, root_id: str, **extra: Any):
        self.kind = kind
        self.root_id = root_id
        self.extra = extra


class GradioPredictAllApp:
    def __init__(
        self,
        config_path: str | Path | None = None,
        *,
        cache_size: int = 3,
        infer_device: str | None = None,
    ):
        self.config_path = resolve_insect_alg_all_path(config_path)
        self.effective_config_path = resolve_effective_insect_alg_path(self.config_path)
        self.run_model = read_run_model_profile(self.config_path)
        self.base_config = load_insect_alg_all(self.config_path)
        # 与 load_insect_alg_all 注入值对齐（profile 合并后仍以启动文件 run_model 为准）
        self.run_model = str(
            self.base_config.get("run_model") or self.run_model
        ).strip().lower() or self.run_model
        self.cache_size = max(1, int(cache_size))
        self.infer_device = infer_device
        self._run_count = resolve_run_count(self.base_config)
        self._api_concurrency = resolve_api_concurrency(self.base_config)
        self._pipeline_cache: "OrderedDict[str, InsectPredictAll]" = OrderedDict()
        self._process_pool: InferenceProcessPool | None = None
        self._models_ready = False
        # API/Gradio 推理线程池：predict_cfg.api_concurrency 控制进程内多图并发
        self._api_executor = ThreadPoolExecutor(
            max_workers=self._api_concurrency,
            thread_name_prefix="insect-api",
        )
        # run_count=1 时业务层可并发；GPU 推理在 model_infer_lock 按权重加锁
        self._pipeline_cache_lock = threading.Lock()
        if self._run_count > 1:
            logger.info(
                "[启动] run_count=%d：将启动 %d 个独立推理进程（各进程独立加载模型）",
                self._run_count,
                self._run_count,
            )
            self._process_pool = InferenceProcessPool(
                self.config_path,
                device=self.infer_device,
                run_count=self._run_count,
            )
        else:
            logger.info(
                "[启动] run_count=1：单进程推理；api_concurrency=%d（进程内多图并发），"
                "GPU 按模型权重加锁",
                self._api_concurrency,
            )
        # 控件与规格按相同顺序维护；运行回调按此顺序解包入参
        self.specs: list[_Spec] = []
        zh = RUN_MODEL_UI_LABELS.get(self.run_model, self.run_model)
        logger.info(
            "[启动] 运行模式=%s（run_model=%s）| 启动配置=%s | 生效配置=%s",
            zh,
            self.run_model,
            self.config_path,
            self.effective_config_path,
        )
        if self._run_count > 1:
            logger.info(
                "[推理设备] run_count=%d：主进程不加载模型，推理由 worker 执行（device=%s）",
                self._run_count,
                self.infer_device or "自动",
            )
            _log_infer_device(self.infer_device)
            logger.warning(
                "[推理性能] run_count=%d 仅在有**并发**请求时提升吞吐；"
                "单张图延迟不会减半。单 GPU（%s）上多进程会争用算力/显存，"
                "通常达不到线性加速；无并发压测时建议 run_count=1",
                self._run_count,
                self.infer_device or "cuda:0",
            )
        else:
            _log_infer_device(self.infer_device)
        _log_predict_perf_config(self.base_config)
        self._warmup_inference()
        self._models_ready = True

    def is_ready(self) -> bool:
        if not self._models_ready:
            return False
        if self._process_pool is not None and not self._process_pool.workers_alive():
            return False
        return True

    def readiness_payload(self) -> dict[str, Any]:
        workers_alive: bool | None = None
        worker_pids: list[int] = []
        if self._process_pool is not None:
            workers_alive = self._process_pool.workers_alive()
            worker_pids = self._process_pool.worker_pids()
        return {
            "ready": self.is_ready(),
            "run_model": self.run_model,
            "run_count": self._run_count,
            "api_concurrency": self._api_concurrency,
            "effective_config": self.effective_config_path.name,
            "workers_alive": workers_alive,
            "worker_pids": worker_pids,
        }

    def _warmup_inference(self) -> None:
        """服务启动时预加载默认配置推理管线（不等待首次识别请求）。"""
        if self._process_pool is not None:
            logger.info(
                "[启动] 推理 worker 池已就绪（%d 进程，各持一套默认配置模型权重）",
                self._run_count,
            )
            return
        logger.info("[启动] 预加载默认推理管线权重（run_count=1）…")
        t0 = time.perf_counter()
        pipeline = self.get_pipeline(
            self.base_config, self.infer_device, cache_tag="api_default"
        )
        summary = pipeline.warmup_models()
        logger.info(
            "[启动] 默认推理管线预加载完成，耗时 %.2fs warmup=%s",
            time.perf_counter() - t0,
            summary,
        )

    def shutdown(self) -> None:
        """释放推理子进程与模型缓存（run_model 切换重启前调用）。"""
        logger.info("[关闭] 开始释放推理资源 ...")
        if self._process_pool is not None:
            try:
                self._process_pool.close()
            except Exception:
                logger.exception("[关闭] InferenceProcessPool 关闭失败")
            self._process_pool = None
        for key, pipe in list(self._pipeline_cache.items()):
            try:
                pipe.release()
            except Exception:
                logger.exception("[关闭] 释放管线失败 key=%s", key)
        self._pipeline_cache.clear()
        if getattr(self, "_api_executor", None) is not None:
            try:
                self._api_executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                logger.exception("[关闭] API 推理线程池关闭失败")
            self._api_executor = None  # type: ignore[assignment]
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception:
            logger.debug("[关闭] torch CUDA 清理跳过", exc_info=True)
        logger.info("[关闭] 推理资源已释放")

    # ----- 生效配置 -----

    def build_effective_config(self, spec_values: list[Any]) -> dict[str, Any]:
        cfg = deepcopy(self.base_config)
        models = cfg.get("models") or {}
        for spec, value in zip(self.specs, spec_values):
            root_cfg = models.get(spec.root_id)
            if not isinstance(root_cfg, dict):
                continue
            kind = spec.kind
            if kind == "root_enable":
                root_cfg["enable"] = bool(value)
            elif kind == "root_num":
                key = spec.extra["key"]
                if value is not None and str(value) != "":
                    coerced = self._coerce_num(key, value)
                    if key == "nms_iou" and float(coerced) <= 0:
                        root_cfg.pop("nms_iou", None)
                    else:
                        root_cfg[key] = coerced
            elif kind in ("dia_lo", "dia_hi"):
                # dia 由 lo/hi 两个控件成对处理，见 _apply_dia_pairs
                pass
            elif kind == "cls_enable":
                node = _dig(root_cfg, spec.extra["path"])
                if node is not None:
                    node["enable"] = bool(value)
            elif kind == "cls_num":
                node = _dig(root_cfg, spec.extra["path"])
                if node is not None:
                    key = spec.extra["key"]
                    if value is not None and str(value) != "":
                        node[key] = float(value)
            elif kind == "cls_classes":
                node = _dig(root_cfg, spec.extra["path"])
                if node is not None:
                    self._apply_class_toggles(
                        node, value, spec.extra["toggleable_names"]
                    )
        # dia：lo/hi 成对处理
        self._apply_dia_pairs(models, spec_values)
        self._log_effective_config(models)
        return cfg

    def _log_effective_config(self, models: dict[str, Any]) -> None:
        """打印每个根模型的开关与关键参数，便于核对启停是否生效。"""
        for root_id, rc in models.items():
            if not isinstance(rc, dict):
                continue
            mtype = str(rc.get("model_type", "detect")).strip().lower()
            enabled = bool(rc.get("enable", True))
            flag = "启用" if enabled else "关闭"
            parts = [f"detect_conf={rc.get('detect_conf', rc.get('conf_thresh'))}"]
            parts.append(f"dia={rc.get('dia')}")
            if rc.get("nms_iou") is not None:
                parts.append(f"nms_iou={rc.get('nms_iou')}")
            if mtype == "segment":
                parts.append(f"seg_imgsz={rc.get('seg_imgsz')}")
            else:
                parts.append(f"clip_size={rc.get('clip_size')}")
                parts.append(f"overlap_size={rc.get('overlap_size')}")
            cls_path = find_primary_cls_path(rc)
            if cls_path is not None:
                cls_cfg = _dig(rc, cls_path) or {}
                cls_on = bool(cls_cfg.get("enable", True))
                items = _leaf_class_items(cls_cfg)
                n_on = sum(1 for _n, _l, en in items if en)
                parts.append(
                    f"cls={'启用' if cls_on else '关闭'}"
                    f"(cls_conf={cls_cfg.get('cls_conf')},类别 {n_on}/{len(items)})"
                )
            logger.info("[配置] 根模型 %s(%s) %s | %s", root_id, mtype, flag, " ".join(parts))

    @staticmethod
    def _coerce_num(key: str, value: Any) -> Any:
        int_keys = {"clip_size", "overlap_size", "seg_imgsz"}
        if key in int_keys:
            return int(round(float(value)))
        return float(value)

    def _apply_dia_pairs(self, models: dict[str, Any], spec_values: list[Any]) -> None:
        # 收集每个 root 的 dia_lo / dia_hi
        dia: dict[str, dict[str, float]] = {}
        for spec, value in zip(self.specs, spec_values):
            if spec.kind in ("dia_lo", "dia_hi"):
                d = dia.setdefault(spec.root_id, {})
                d[spec.kind] = float(value or 0)
        for root_id, d in dia.items():
            root_cfg = models.get(root_id)
            if not isinstance(root_cfg, dict):
                continue
            lo = d.get("dia_lo", 0.0)
            hi = d.get("dia_hi", 0.0)
            if lo > 0 and hi > 0 and hi >= lo:
                root_cfg["dia"] = [int(lo), int(hi)]
            else:
                root_cfg.pop("dia", None)

    @staticmethod
    def _apply_class_toggles(
        cls_cfg: dict[str, Any],
        selected: list[str] | str | None,
        toggleable_names: set[str] | frozenset[str],
    ) -> None:
        out = cls_cfg.get("out")
        if not isinstance(out, dict):
            return
        if isinstance(selected, str):
            selected = [selected]
        selected_names = set(selected or [])
        for name, entry in out.items():
            if not isinstance(entry, dict) or name not in toggleable_names:
                continue
            entry["enable"] = name in selected_names

    # ----- 管线缓存 -----

    def _evict_pipelines(self, *pipelines: InsectPredictAll | None) -> None:
        """模型加载失败后移除对应管线缓存，确保下次请求重新构建并加载权重。"""
        target_ids = {id(p) for p in pipelines if p is not None}
        if not target_ids:
            return
        stale_keys = [
            k for k, v in self._pipeline_cache.items() if id(v) in target_ids
        ]
        for key in stale_keys:
            pipe = self._pipeline_cache.pop(key, None)
            if pipe is None:
                continue
            try:
                pipe.release()
            except Exception:
                logger.warning("释放失败管线时异常", exc_info=True)
        if stale_keys:
            logger.info(
                "模型加载失败，已清除 %d 条管线缓存，下次请求将重新加载",
                len(stale_keys),
            )

    def _handle_predict_failure(
        self, exc: BaseException, pipeline: InsectPredictAll | None
    ) -> None:
        if pipeline is not None and _is_model_load_error(exc):
            self._evict_pipelines(pipeline)

    def _config_matches_base(self, cfg: dict[str, Any]) -> bool:
        return _configs_equivalent(cfg, self.base_config)

    def _can_use_process_pool(self, cfg: dict[str, Any] | None = None) -> bool:
        """``run_count>1`` 时推理仅由 worker 池承担，主进程不加载模型。"""
        return self._process_pool is not None

    def _dispatch_predict(
        self,
        image: str | np.ndarray,
        *,
        cfg: dict[str, Any],
        device: str | None = None,
        cache_tag: str | None = None,
        collect_filtered: bool = False,
        phase_profile: bool = False,
    ) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if self._process_pool is not None:
            if isinstance(image, np.ndarray):
                raise TypeError(
                    "run_count>1 时请传递图片 URL 或本地路径，勿在主进程解码后传 numpy"
                )
            image_ref = str(image).strip()
            if not image_ref:
                if collect_filtered:
                    return [], []
                return []
            if cfg is not self.base_config and not self._config_matches_base(cfg):
                logger.warning(
                    "[推理] run_count=%d：请求参数与默认配置不一致，"
                    "worker 池仍按启动时默认配置推理（忽略本次参数覆盖）",
                    self._run_count,
                )
            return self._process_pool.predict(  # type: ignore[union-attr]
                image_ref,
                collect_filtered=collect_filtered,
                phase_profile=phase_profile,
            )
        if isinstance(image, str):
            img_bgr = load_image_bgr_from_ref(image)
            if img_bgr is None or getattr(img_bgr, "size", 0) == 0:
                if collect_filtered:
                    return [], []
                return []
        else:
            img_bgr = image
        pipeline = self.get_pipeline(cfg, device, cache_tag=cache_tag)
        return pipeline.predict(
            img_bgr,
            collect_filtered=collect_filtered,
            phase_profile=phase_profile,
        )

    def get_api_pipeline(self) -> InsectPredictAll:
        """REST API 默认配置管线（``run_count>1`` 时不可用，请走 worker 池）。"""
        if self._process_pool is not None:
            raise RuntimeError("run_count>1 时主进程不持有推理管线，请使用 worker 池")
        return self.get_pipeline(
            self.base_config, self.infer_device, cache_tag="api_default"
        )

    def get_pipeline(
        self,
        cfg: dict[str, Any],
        device: str | None,
        *,
        cache_tag: str | None = None,
    ) -> InsectPredictAll:
        if self._process_pool is not None:
            raise RuntimeError(
                "run_count>1 时禁止在主进程加载模型（请使用 InferenceProcessPool）"
            )
        with self._pipeline_cache_lock:
            if cache_tag is not None and cfg is self.base_config:
                key: Any = (cache_tag, device)
            else:
                key = json.dumps(
                    {"d": device, "c": cfg}, sort_keys=True, ensure_ascii=False
                )
            if key in self._pipeline_cache:
                self._pipeline_cache.move_to_end(key)
                pipeline = self._pipeline_cache[key]
                logger.debug(
                    "[管线] 命中缓存（device=%s）",
                    _device_label(device),
                )
                return pipeline
            logger.info(
                "[管线] 未命中缓存（device=%s），开始构建并加载模型 ...",
                _device_label(device),
            )
            t0 = time.perf_counter()
            pipeline = InsectPredictAll(cfg, device=device)
            logger.info(
                "[管线] 构建完成，耗时 %.2fs；注册根模型=%s",
                time.perf_counter() - t0,
                list(pipeline._roots.keys()),  # noqa: SLF001
            )
            self._pipeline_cache[key] = pipeline
            while len(self._pipeline_cache) > self.cache_size:
                _old_key, old = self._pipeline_cache.popitem(last=False)
                try:
                    old.release()
                    logger.info("[管线] 缓存超出 %d，已释放最旧管线", self.cache_size)
                except Exception:  # pragma: no cover - 释放失败不影响主流程
                    logger.warning("旧推理管线释放失败", exc_info=True)
            return pipeline

    # ----- 运行回调 -----

    def run(
        self,
        image: np.ndarray | str | None,
        device_label: str,
        draw_polygon: bool,
        draw_filter: bool,
        show_source: bool,
        font_size: float,
        *spec_values: Any,
    ):
        use_pool = self._process_pool is not None
        if image is None:
            logger.info("[运行] 未上传图片，已跳过。")
            return None, "请先上传图片。", [], {}
        image_ref = ""
        if use_pool:
            if isinstance(image, np.ndarray):
                return (
                    None,
                    "多进程模式下请使用文件路径上传图片（勿直传像素数组）。",
                    [],
                    {},
                )
            image_ref = str(image).strip()
            if not Path(image_ref).is_file():
                logger.info("[运行] 未上传图片，已跳过。")
                return None, "请先上传图片。", [], {}
            logger.info(
                "=== [运行] 开始 === 设备=%s 图片路径=%s（worker 读图）",
                device_label,
                image_ref,
            )
        else:
            image_rgb = _load_image_rgb_from_input(image)
            if image_rgb is None:
                logger.info("[运行] 未上传图片，已跳过。")
                return None, "请先上传图片。", [], {}
            h, w = image_rgb.shape[:2]
            logger.info("=== [运行] 开始 === 设备=%s 图片尺寸=%dx%d", device_label, w, h)
        try:
            cfg = self.build_effective_config(list(spec_values))
        except Exception as e:  # pragma: no cover
            logger.exception("构建生效配置失败")
            return None, f"配置解析失败：{e}", [], {}

        enabled_roots = [
            rid
            for rid, rc in (cfg.get("models") or {}).items()
            if isinstance(rc, dict) and bool(rc.get("enable", True))
        ]
        disabled_roots = [
            rid
            for rid, rc in (cfg.get("models") or {}).items()
            if isinstance(rc, dict) and not bool(rc.get("enable", True))
        ]
        logger.info(
            "[运行] 启用根模型=%s ｜ 关闭根模型=%s", enabled_roots, disabled_roots
        )
        if not enabled_roots:
            logger.warning("[运行] 没有启用任何根模型，已中止推理。")
            return None, "没有启用任何根模型，请在下方至少启用一个。", [], cfg

        img_bgr: np.ndarray
        if use_pool:
            img_bgr = np.empty((0, 0, 3), dtype=np.uint8)
        else:
            img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        device = _device_arg(device_label)
        try:
            if use_pool:
                logger.info(
                    "[运行] 使用推理 worker 池（run_count=%d，传路径不序列化像素）",
                    self._run_count,
                )
                if not self._config_matches_base(cfg):
                    logger.warning(
                        "[运行] 页面算法参数与默认配置不完全一致，"
                        "推理仍使用 worker 默认配置；摘要仍展示本次页面参数"
                    )
            t0 = time.perf_counter()
            predict_input: str | np.ndarray = (
                image_ref if use_pool else img_bgr
            )
            out = self._dispatch_predict(
                predict_input,
                cfg=self.base_config if use_pool else cfg,
                device=device,
                collect_filtered=True,
                phase_profile=False,
            )
            results, filtered = out  # type: ignore[misc]
            logger.info("[运行] 推理完成，耗时 %.2fs", time.perf_counter() - t0)
        except Exception as e:
            logger.exception("推理失败")
            self._handle_predict_failure(e, None)
            return None, f"推理失败：{e}", [], cfg

        self._log_result_breakdown(results, filtered)

        if use_pool:
            image_rgb = _load_image_rgb_from_input(image_ref)
            if image_rgb is None:
                return None, "推理完成但无法读取原图用于绘制。", [], cfg
            img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        font_px = int(font_size) if font_size and int(font_size) > 0 else None
        t1 = time.perf_counter()
        drawn_bgr = draw_results(
            img_bgr,
            results,
            None,
            draw_bbox=True,
            draw_polygon=bool(draw_polygon),
            label_font_size=font_px,
            show_source_in_label=bool(show_source),
            filtered_results=filtered,
            draw_filter=bool(draw_filter),
        )
        out_path = self._save_annotated(drawn_bgr)
        logger.info(
            "[运行] 绘制+落盘完成，耗时 %.2fs；结果图=%s",
            time.perf_counter() - t1,
            out_path,
        )

        table = self._results_table(results)
        summary = (
            f"**识别完成**：保留目标 **{len(results)}** 个，过滤 **{len(filtered)}** 个。"
            f" 启用根模型：{', '.join(enabled_roots)}。"
        )
        logger.info("=== [运行] 结束 === 返回前端")
        return out_path, summary, table, self._config_summary(cfg)

    # ----- 纯 API 端点（只传图片，默认参数，返回过滤后结果） -----

    def predict_api(self, image: np.ndarray | str | None) -> dict[str, Any]:
        """
        纯 API 端点（``api_name="/predict"``）：仅传入一张图片，使用
        ``insect_alg_all.json`` 的默认算法参数推理，返回**过滤后保留**的结果。
        不绘制、不返回图片。

        返回：``{"count": int, "results": [ {name, cn_name, source,
        det_conf, cls_conf, location:[x1,y1,x2,y2]} , ...] }``；
        异常时返回 ``{"count": 0, "results": [], "error": "..."}``。
        """
        use_pool = self._process_pool is not None
        if image is None:
            return {"count": 0, "results": [], "error": "未提供图片"}
        if not self.is_ready():
            return {"count": 0, "results": [], "error": "推理服务尚未就绪，请稍后重试"}
        if use_pool:
            if isinstance(image, np.ndarray):
                return {
                    "count": 0,
                    "results": [],
                    "error": "多进程模式请传递图片路径，勿传 numpy",
                }
            image_ref = str(image).strip()
            if not Path(image_ref).is_file():
                return {"count": 0, "results": [], "error": "未提供图片"}
            zh = RUN_MODEL_UI_LABELS.get(self.run_model, self.run_model)
            logger.info(
                "=== [API] /predict 开始 === 图片路径=%s | "
                "运行模式=%s（run_model=%s，默认参数，worker 读图）",
                image_ref,
                zh,
                self.run_model,
            )
            predict_input: str | np.ndarray = image_ref
        else:
            image_rgb = _load_image_rgb_from_input(image)
            if image_rgb is None:
                return {"count": 0, "results": [], "error": "未提供图片"}
            h, w = image_rgb.shape[:2]
            zh = RUN_MODEL_UI_LABELS.get(self.run_model, self.run_model)
            logger.info(
                "=== [API] /predict 开始 === 图片尺寸=%dx%d | "
                "运行模式=%s（run_model=%s，默认参数）",
                w,
                h,
                zh,
                self.run_model,
            )
            predict_input = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        try:
            t0 = time.perf_counter()
            results = self._dispatch_predict(
                predict_input,
                cfg=self.base_config,
                device=self.infer_device,
                cache_tag="api_default",
                collect_filtered=False,
                phase_profile=False,
            )
            logger.info(
                "[API] /predict 推理完成，耗时 %.2fs，保留=%d",
                time.perf_counter() - t0,
                len(results),
            )
        except Exception as e:
            logger.exception("[API] /predict 推理失败")
            return {"count": 0, "results": [], "error": f"推理失败：{e}"}
        payload = self._results_payload(results)
        return {"count": len(payload), "results": payload}

    def predict_insect_api(self, url: str) -> dict[str, Any]:
        """
        虫情识别 HTTP 接口（``/insect_3_predict?input_type=url``）推理逻辑。

        传入 ``url``（远程或本地路径）；``run_count>1`` 时由 worker 读图，
        主进程不经 Queue 序列化像素。
        """
        url = str(url or "").strip()
        zh = RUN_MODEL_UI_LABELS.get(self.run_model, self.run_model)
        logger.info(
            "=== [API] /insect_3_predict 开始 === url=%s | "
            "运行模式=%s（run_model=%s）| 生效配置=%s",
            url,
            zh,
            self.run_model,
            self.effective_config_path.name,
        )
        try:
            t0 = time.perf_counter()
            results = self._dispatch_predict(
                url,
                cfg=self.base_config,
                device=self.infer_device,
                cache_tag="api_default",
                collect_filtered=False,
                phase_profile=False,
            )
            logger.info(
                "[API] /insect_3_predict pipeline.predict 耗时 %.2fs，保留=%d "
                "（run_model=%s run_count=%d）",
                time.perf_counter() - t0,
                len(results),
                self.run_model,
                self._run_count,
            )
        except Exception as e:
            logger.exception("[API] /insect_3_predict 推理失败")
            raise RuntimeError(f"推理失败：{e}") from e
        return {"results": self._format_insect_api_results(results)}

    @staticmethod
    def _format_insect_api_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """转为「图片识别接口说明文档」约定的 results 结构。"""
        out: list[dict[str, Any]] = []
        for r in results:
            loc = r.get("location") or [0, 0, 0, 0]
            x1, y1, x2, y2 = int(loc[0]), int(loc[1]), int(loc[2]), int(loc[3])
            score = float(
                r.get("cls_conf") or r.get("det_conf") or r.get("score") or 0.0
            )
            out.append(
                {
                    "name": str(r.get("name") or ""),
                    "score": score,
                    "location": {
                        "left": x1,
                        "top": y1,
                        "width": x2 - x1,
                        "height": y2 - y1,
                    },
                }
            )
        return out

    @staticmethod
    def _results_payload(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """把保留结果整理为可 JSON 序列化的精简字段列表。"""
        out: list[dict[str, Any]] = []
        for r in results:
            loc = r.get("location") or [0, 0, 0, 0]
            out.append(
                {
                    "name": str(r.get("name") or ""),
                    "cn_name": str(r.get("cn_name") or ""),
                    "source": str(r.get("source") or ""),
                    "det_conf": round(
                        float(r.get("det_conf", r.get("score", 0.0)) or 0.0), 4
                    ),
                    "cls_conf": round(
                        float(r.get("cls_conf", r.get("score", 0.0)) or 0.0), 4
                    ),
                    "location": [int(loc[0]), int(loc[1]), int(loc[2]), int(loc[3])],
                }
            )
        return out

    @staticmethod
    def _save_annotated(image_bgr: np.ndarray) -> str:
        """把标注结果图写为临时 JPEG，返回路径（避免大数组内联回传卡住前端）。"""
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = _OUTPUT_DIR / f"result_{uuid.uuid4().hex}.jpg"
        cv2.imwrite(str(out_path), image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        return str(out_path)

    @staticmethod
    def _config_summary(cfg: dict[str, Any]) -> dict[str, Any]:
        """生成精简的生效配置摘要（仅开关与关键参数），避免回传上百类导致前端卡顿。"""
        summary: dict[str, Any] = {}
        for root_id, rc in (cfg.get("models") or {}).items():
            if not isinstance(rc, dict):
                continue
            mtype = str(rc.get("model_type", "detect")).strip().lower()
            item: dict[str, Any] = {
                "enable": bool(rc.get("enable", True)),
                "model_type": mtype,
                "detect_conf": rc.get("detect_conf", rc.get("conf_thresh")),
                "dia": rc.get("dia"),
                "nms_iou": rc.get("nms_iou"),
            }
            if mtype == "segment":
                item["seg_imgsz"] = rc.get("seg_imgsz")
            else:
                item["clip_size"] = rc.get("clip_size")
                item["overlap_size"] = rc.get("overlap_size")
            cls_path = find_primary_cls_path(rc)
            if cls_path is not None:
                cls_cfg = _dig(rc, cls_path) or {}
                leaves = _leaf_class_items(cls_cfg)
                item["cls"] = {
                    "enable": bool(cls_cfg.get("enable", True)),
                    "cls_conf": cls_cfg.get("cls_conf"),
                    "classes_enabled": sum(1 for _n, _l, en in leaves if en),
                    "classes_total": len(leaves),
                }
            summary[root_id] = item
        return summary

    @staticmethod
    def _log_result_breakdown(
        results: list[dict[str, Any]], filtered: list[dict[str, Any]]
    ) -> None:
        """按来源/类别/过滤原因汇总打印，便于核对各模型是否真的出框。"""
        by_source = Counter(str(r.get("source") or "?") for r in results)
        by_name = Counter(str(r.get("name") or "?") for r in results)
        by_reason = Counter(str(r.get("filter_reason") or "?") for r in filtered)
        logger.info(
            "[结果] 保留=%d 过滤=%d ｜ 按来源=%s",
            len(results),
            len(filtered),
            dict(by_source),
        )
        if by_name:
            logger.info("[结果] 按类别 Top: %s", dict(by_name.most_common(10)))
        if by_reason:
            logger.info("[结果] 过滤原因: %s", dict(by_reason.most_common(10)))

    @staticmethod
    def _results_table(results: list[dict[str, Any]]) -> list[list[Any]]:
        rows: list[list[Any]] = []
        for r in results:
            loc = r.get("location") or [0, 0, 0, 0]
            rows.append(
                [
                    str(r.get("name") or ""),
                    str(r.get("cn_name") or ""),
                    str(r.get("source") or ""),
                    round(float(r.get("det_conf", r.get("score", 0.0)) or 0.0), 3),
                    round(float(r.get("cls_conf", r.get("score", 0.0)) or 0.0), 3),
                    f"[{int(loc[0])}, {int(loc[1])}, {int(loc[2])}, {int(loc[3])}]",
                ]
            )
        return rows

    @staticmethod
    def _file_to_preview(file_obj: Any) -> str | None:
        """把 ``gr.File`` 上传结果转为 ``gr.Image`` 可显示的 filepath。"""
        if file_obj is None:
            return None
        if isinstance(file_obj, str) and file_obj.strip():
            return file_obj
        if isinstance(file_obj, list):
            for item in file_obj:
                path = GradioPredictAllApp._file_to_preview(item)
                if path:
                    return path
            return None
        if isinstance(file_obj, dict):
            for key in ("path", "name"):
                val = file_obj.get(key)
                if isinstance(val, str) and val.strip():
                    return val
        return None

    # ----- 运行模式 run_model / 服务重启 -----

    def restart_service(self) -> tuple[str, str]:
        """在不修改 run_model 的前提下重启整站服务并重新加载模型。"""
        zh = RUN_MODEL_UI_LABELS.get(self.run_model, self.run_model)
        logger.info(
            "用户触发服务重启（run_model 不变=%s），%.1fs 后重启",
            self.run_model,
            _RESTART_DELAY_SEC,
        )
        self._models_ready = False
        threading.Timer(_RESTART_DELAY_SEC, _restart_server_process).start()
        return (
            f"✅ 正在重启后台服务（运行模式保持 **{zh}**，`run_model={self.run_model}`），"
            f"将按当前配置**重新加载模型**，请稍候…",
            "restart_same",
        )

    def change_run_model(self, new_mode: str) -> tuple[str, str]:
        """写回 ``insect_alg_all.json`` 的 ``run_model`` 并延迟重启整站服务。"""
        try:
            key = normalize_run_model_key(new_mode)
        except ValueError as exc:
            return f"❌ {exc}", ""
        if key == self.run_model:
            return (
                _run_model_status_md(
                    self.run_model,
                    config_path=self.config_path,
                    effective_path=self.effective_config_path,
                )
                + "\n\n当前已是该模式，无需重启。",
                "",
            )
        try:
            write_run_model_profile(key, self.config_path)
        except Exception as exc:
            logger.exception("写入 run_model 失败")
            return f"❌ 写入配置失败：{exc}", ""
        zh = RUN_MODEL_UI_LABELS.get(key, key)
        effective = resolve_effective_insect_alg_path(self.config_path)
        logger.info(
            "run_model 已从 %s 切换为 %s，%.1fs 后重启服务",
            self.run_model,
            key,
            _RESTART_DELAY_SEC,
        )
        self._models_ready = False
        threading.Timer(_RESTART_DELAY_SEC, _restart_server_process).start()
        return (
            f"✅ 已切换为 **{zh}**（`run_model={key}`），"
            f"生效配置 `{effective.name}`。"
            f"后台正在重启并按新配置**预加载模型**，请稍候…",
            "restart",
        )

    # ----- 界面构建 -----

    def build_ui(self) -> gr.Blocks:
        self.specs = []
        inputs: list[Any] = []

        with gr.Blocks(
            title="虫情统一推理 · 测试服务", analytics_enabled=False
        ) as demo:
            gr.HTML(
                _service_gate_html(),
                container=False,
                js_on_load=_SERVICE_GATE_JS_ON_LOAD,
            )
            mode_change_poll = gr.Textbox(value="", visible=False)
            gr.Markdown(
                "# 虫情统一推理 · 图片测试服务\n"
                "下方可切换「运行模式」或点击「重启服务」重新加载模型；"
                "服务重启期间页面会锁定，模型就绪后自动恢复。\n"
                "其余算法参数仅作用于**本次运行**。"
            )
            run_model_choices = [
                (_run_model_choice_label(k), k) for k in RUN_MODEL_CHOICES
            ]
            with gr.Row():
                run_model_in = gr.Dropdown(
                    choices=run_model_choices,
                    value=self.run_model,
                    label="运行模式 run_model",
                    info="摆拍 / 生产 / 其他；切换后写回 insect_alg_all.json 并重启后台服务，请稍等片刻",
                    scale=1,
                )
                restart_btn = gr.Button(
                    "重启服务",
                    variant="secondary",
                    elem_id="insect-restart-btn",
                    scale=0,
                )
            run_model_status = gr.Markdown(
                _run_model_status_md(
                    self.run_model,
                    config_path=self.config_path,
                    effective_path=self.effective_config_path,
                )
            )
            run_model_in.change(
                fn=self.change_run_model,
                inputs=[run_model_in],
                outputs=[run_model_status, mode_change_poll],
            ).then(
                fn=_noop_gate_poll_trigger,
                inputs=[mode_change_poll],
                outputs=[],
                js=_MODE_CHANGE_POLL_JS,
            )
            restart_btn.click(
                fn=self.restart_service,
                inputs=[],
                outputs=[run_model_status, mode_change_poll],
            ).then(
                fn=_noop_gate_poll_trigger,
                inputs=[mode_change_poll],
                outputs=[],
                js=_MODE_CHANGE_POLL_JS,
            )

            demo.load(js=_PAGE_LOAD_GATE_JS)

            with gr.Column(elem_id="insect-main-panel"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # Gradio 5.0–5.28：对已有图再拖拽会新开浏览器标签（#10281），5.29+ 已修复。
                        # 按钮上传各版本均可靠；拖拽请升级 gradio>=5.31 后使用下方图片区。
                        upload_btn = gr.UploadButton(
                            "选择/替换图片",
                            file_types=["image"],
                        )
                        image_in = gr.Image(
                            label="图片预览（gradio≥5.29 可拖拽到此处替换）",
                            type="filepath",
                            height=320,
                            interactive=True,
                            sources=["upload", "clipboard"],
                        )
                        upload_btn.upload(
                            fn=self._file_to_preview,
                            inputs=[upload_btn],
                            outputs=[image_in],
                        )
                        device_in = gr.Dropdown(
                            _DEVICE_CHOICES, value="自动", label="推理设备"
                        )
                        with gr.Group():
                            gr.Markdown("**绘制选项**")
                            draw_polygon_in = gr.Checkbox(
                                value=True, label="绘制分割多边形（mask）"
                            )
                            draw_filter_in = gr.Checkbox(
                                value=False, label="绘制被过滤实例（灰框）"
                            )
                            show_source_in = gr.Checkbox(
                                value=False, label="标签显示来源(source)"
                            )
                            font_size_in = gr.Number(
                                value=0, label="标签字号（0=自适应）", precision=0
                            )
                        run_btn = gr.Button(
                            "运行识别", variant="primary", elem_id="insect-run-btn"
                        )
                    with gr.Column(scale=2):
                        image_out = gr.Image(
                            label="识别结果",
                            type="filepath",
                            height=560,
                            interactive=False,
                        )

                summary_out = gr.Markdown("")
                table_out = gr.Dataframe(
                    headers=["类名", "中文名", "来源", "det_conf", "cls_conf", "位置"],
                    label="结果明细",
                    wrap=True,
                )

                gr.Markdown("## 算法开关与参数")
                self._build_model_controls(inputs)

                with gr.Accordion("生效配置摘要（只读）", open=False):
                    config_out = gr.JSON(label="本次运行的开关与关键参数摘要")

                run_btn.click(
                    fn=self.run,
                    inputs=[
                        image_in,
                        device_in,
                        draw_polygon_in,
                        draw_filter_in,
                        show_source_in,
                        font_size_in,
                        *inputs,
                    ],
                    outputs=[image_out, summary_out, table_out, config_out],
                )

            # 纯 API 端点：仅传图片、默认参数、返回过滤后结果（隐藏控件，仅供 API 调用）
            api_image_in = gr.Image(type="filepath", visible=False)
            api_out = gr.JSON(visible=False)
            api_btn = gr.Button("predict_api", visible=False)
            api_btn.click(
                fn=self.predict_api,
                inputs=[api_image_in],
                outputs=[api_out],
                api_name="predict",
            )
        return demo

    def _build_model_controls(self, inputs: list[Any]) -> None:
        models = self.base_config.get("models") or {}
        for root_id, root_cfg in models.items():
            if not isinstance(root_cfg, dict):
                continue
            mtype = str(root_cfg.get("model_type", "detect")).strip().lower()
            base_enable = bool(root_cfg.get("enable", True))
            title = f"{root_id} · {mtype}" + ("（默认启用）" if base_enable else "（默认关闭）")
            with gr.Accordion(title, open=base_enable):
                enable_in = gr.Checkbox(value=base_enable, label="启用该根模型")
                inputs.append(enable_in)
                self.specs.append(_Spec("root_enable", root_id))

                with gr.Row():
                    conf_default = float(
                        root_cfg.get("detect_conf", root_cfg.get("conf_thresh", 0.3))
                    )
                    conf_in = gr.Number(value=conf_default, label="检测/分割置信度 detect_conf")
                    inputs.append(conf_in)
                    self.specs.append(_Spec("root_num", root_id, key="detect_conf"))

                    dia = root_cfg.get("dia") or [0, 0]
                    dia_lo = float(dia[0]) if len(dia) >= 1 else 0.0
                    dia_hi = float(dia[1]) if len(dia) >= 2 else 0.0
                    dia_lo_in = gr.Number(value=dia_lo, label="对角线下限 dia[0]（0=不限）")
                    inputs.append(dia_lo_in)
                    self.specs.append(_Spec("dia_lo", root_id))
                    dia_hi_in = gr.Number(value=dia_hi, label="对角线上限 dia[1]（0=不限）")
                    inputs.append(dia_hi_in)
                    self.specs.append(_Spec("dia_hi", root_id))

                nms_default = float(root_cfg.get("nms_iou") or 0)
                nms_in = gr.Number(
                    value=nms_default,
                    label="YOLO NMS IoU nms_iou（0=不传，用 YOLO 默认）",
                )
                inputs.append(nms_in)
                self.specs.append(_Spec("root_num", root_id, key="nms_iou"))

                with gr.Row():
                    if mtype == "segment":
                        seg_imgsz_in = gr.Number(
                            value=int(root_cfg.get("seg_imgsz", 0) or 0),
                            label="分割推理边长 seg_imgsz（0=YOLO默认）",
                            precision=0,
                        )
                        inputs.append(seg_imgsz_in)
                        self.specs.append(_Spec("root_num", root_id, key="seg_imgsz"))
                    else:
                        clip_in = gr.Number(
                            value=int(root_cfg.get("clip_size", 640) or 0),
                            label="切片大小 clip_size（0=整图）",
                            precision=0,
                        )
                        inputs.append(clip_in)
                        self.specs.append(_Spec("root_num", root_id, key="clip_size"))
                        overlap_in = gr.Number(
                            value=int(root_cfg.get("overlap_size", 120) or 0),
                            label="切片重叠 overlap_size",
                            precision=0,
                        )
                        inputs.append(overlap_in)
                        self.specs.append(_Spec("root_num", root_id, key="overlap_size"))

                cls_path = find_primary_cls_path(root_cfg)
                if cls_path is not None:
                    cls_cfg = _dig(root_cfg, cls_path) or {}
                    with gr.Group():
                        gr.Markdown("**分类节点**")
                        with gr.Row():
                            cls_enable_in = gr.Checkbox(
                                value=bool(cls_cfg.get("enable", True)),
                                label="启用分类节点",
                            )
                            inputs.append(cls_enable_in)
                            self.specs.append(
                                _Spec("cls_enable", root_id, path=cls_path)
                            )
                            cls_conf_in = gr.Number(
                                value=float(cls_cfg.get("cls_conf", 0.3) or 0.3),
                                label="分类阈值 cls_conf",
                            )
                            inputs.append(cls_conf_in)
                            self.specs.append(
                                _Spec("cls_num", root_id, path=cls_path, key="cls_conf")
                            )

                        items = _leaf_class_items(cls_cfg)
                        if items:
                            toggleable_names = {name for name, _, _ in items}
                            # (展示标签, 类名)：value 用稳定 ASCII 类名，避免 Gradio 回传标签文本时字符不一致
                            choices = [(lbl, name) for name, lbl, _ in items]
                            default_sel = [name for name, _, en in items if en]
                            with gr.Accordion(
                                f"类别启停（共 {len(items)} 类，取消勾选即关闭）",
                                open=False,
                            ):
                                classes_in = gr.CheckboxGroup(
                                    choices=choices,
                                    value=default_sel,
                                    label="启用的输出类别",
                                )
                            inputs.append(classes_in)
                            self.specs.append(
                                _Spec(
                                    "cls_classes",
                                    root_id,
                                    path=cls_path,
                                    toggleable_names=toggleable_names,
                                )
                            )


def register_insect_http_routes(
    fastapi_app: FastAPI, svc: GradioPredictAllApp
) -> None:
    """
    注册与「图片识别接口说明文档」一致的 POST 端点。

    须挂载到独立 ``FastAPI`` 后再 ``gr.mount_gradio_app``；若用
    ``@demo.app.post`` + ``demo.launch()``，Gradio 会在 launch 时重建
    FastAPI 应用，导致 ``/insect_3_predict`` 恒为 404。

    ``POST /insect_3_predict?input_type=url``
    Body: ``{"url": "<http(s)://... 或本地图片路径>"}``
    """

    @fastapi_app.get(_HEALTH_READY_PATH, response_model=None)
    async def health_ready() -> JSONResponse:
        payload = svc.readiness_payload()
        if not payload.get("ready"):
            return JSONResponse(payload, status_code=503)
        return JSONResponse(payload)

    @fastapi_app.post("/insect_3_predict")
    async def insect_3_predict(request: Request) -> JSONResponse:
        params = request.query_params
        if params.get("input_type") != "url":
            return _insect_api_error("请求参数input_type值不正确，应该是url")
        try:
            body = await request.json()
        except Exception:
            return _insect_api_error("请求body必须是JSON")
        url = body.get("url")
        if not isinstance(body, dict) or "url" not in body:
            return _insect_api_error("请求body必须包含url参数")
        if url is None or url == "":
            return _insect_api_error("请求body必须包含url参数")
        if not svc.is_ready():
            return _insect_api_error("推理服务尚未就绪，请稍后重试")
        url = str(url)
        logger.info(
            "[API] /insect_3_predict 请求 url=%s | run_model=%s",
            url,
            svc.run_model,
        )
        try:
            t_req = time.perf_counter()
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                svc._api_executor, svc.predict_insect_api, url
            )
            total_s = time.perf_counter() - t_req
            logger.info(
                "[API] /insect_3_predict 请求完成 合计=%.2fs（worker 读图+推理）",
                total_s,
            )
            logger.info(
                "[API] /insect_3_predict 返回 results=%d",
                len(result.get("results") or []),
            )
            return JSONResponse(result)
        except ValueError as exc:
            logger.warning("[API] /insect_3_predict 读取图片失败: %s", exc)
            return _insect_api_error("读取图片异常")
        except Exception:
            logger.exception("[API] /insect_3_predict 处理失败")
            return _insect_api_error("读取图片异常")


def create_app(
    config_path: str | Path | None = None,
    *,
    cache_size: int = 3,
    infer_device: str | None = None,
) -> gr.Blocks:
    """仅构建 Gradio 页面（供测试或外部挂载）。"""
    app = GradioPredictAllApp(
        config_path, cache_size=cache_size, infer_device=infer_device
    )
    register_server_shutdown(app.shutdown)
    return app.build_ui()


def create_serving_app(
    config_path: str | Path | None = None,
    *,
    cache_size: int = 3,
    mount_path: str = "/",
    infer_device: str | None = None,
) -> FastAPI:
    """
    构建可对外提供 Gradio 页面 + ``/insect_3_predict`` REST 的 FastAPI 应用。

    使用 ``gr.mount_gradio_app`` 挂载，避免 ``demo.launch()`` 重建 FastAPI
    导致自定义路由 404。
    """
    svc = GradioPredictAllApp(
        config_path, cache_size=cache_size, infer_device=infer_device
    )
    register_server_shutdown(svc.shutdown)
    demo = svc.build_ui().queue(
        default_concurrency_limit=svc._api_concurrency,
    )
    fastapi_app = FastAPI(title="虫情统一推理")
    register_insect_http_routes(fastapi_app, svc)
    return gr.mount_gradio_app(fastapi_app, demo, path=mount_path)


if __name__ == "__main__":

    # 推荐：同时捕获 stdout+stderr（日志已双写，只重定向 stdout 也能在 web.log 看到）
    # nohup /home/shunyao/miniconda310/envs/yolo11/bin/python3  /data/script/predict_all_gradio.py > web.log 2>&1 &
    configure_process_logging(logging.INFO)

    # 入口参数（IDE 直接运行即可，按需修改）
    CONFIG_PATH: str | None = None  # None → 使用 script/config/insect_alg_all.json
    SERVER_NAME = "0.0.0.0"
    SERVER_PORT = 37860
    SHARE = False
    CACHE_SIZE = 3
    # REST API 与 Gradio 默认推理设备：None=自动；Linux 服务器建议 "cuda:0"
    INFER_DEVICE: str | None = "cuda:0"

    import uvicorn

    _warn_gradio_drag_drop_version()
    configure_server_restart_argv([sys.executable, str(_FILE)])
    if SHARE:
        logger.warning(
            "SHARE=True 与 REST mount 启动方式不兼容，已忽略 share；"
            "如需公网链接请自行做端口转发或反向代理。"
        )
    app = create_serving_app(
        CONFIG_PATH, cache_size=CACHE_SIZE, infer_device=INFER_DEVICE
    )
    _startup_cfg = load_insect_alg_all(CONFIG_PATH)
    _startup_run_count = resolve_run_count(_startup_cfg)
    _startup_api_concurrency = resolve_api_concurrency(_startup_cfg)
    logger.info(
        "启动服务 %s:%s（Gradio 页面 + POST /insect_3_predict?input_type=url）",
        SERVER_NAME,
        SERVER_PORT,
    )
    if _startup_run_count > 1:
        logger.info(
            "推理并发：run_count=%d（%d worker 进程）+ api_concurrency=%d（HTTP/Gradio）；"
            "仅多请求同时到达时才有吞吐收益，单 GPU 通常非线性加速",
            _startup_run_count,
            _startup_run_count,
            _startup_api_concurrency,
        )
    else:
        logger.info(
            "推理并发：run_count=1，进程内多图 api_concurrency=%d；"
            "GPU 推理按模型权重加锁，多图可交错 CPU/GPU",
            _startup_api_concurrency,
        )
    uvicorn.run(app, host=SERVER_NAME, port=SERVER_PORT, log_config=None)
