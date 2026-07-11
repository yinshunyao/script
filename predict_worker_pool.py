#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Detail  : ``predict_cfg.run_count`` > 1 时，多进程推理池（各进程独立加载模型）。

from __future__ import annotations

import atexit
import logging
import multiprocessing as mp
import queue
import sys
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_WORKER_READY = "__worker_ready__"


def _ensure_insect_root_on_syspath() -> None:
    """spawn 子进程不继承主进程 ``sys.path`` 补丁，须自行加入 insect 根目录。"""
    root = Path(__file__).resolve().parents[1]
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)


def _log_worker_cuda(wlog: logging.Logger, worker_id: int, device: str | None) -> None:
    try:
        import torch

        avail = torch.cuda.is_available()
        name = torch.cuda.get_device_name(0) if avail else "-"
        alloc_mib = 0.0
        if avail:
            idx = 0
            if device and ":" in str(device):
                idx = int(str(device).rsplit(":", 1)[-1])
            alloc_mib = torch.cuda.memory_allocated(idx) / (1024**2)
        wlog.info(
            "worker-%d CUDA available=%s device=%s gpu=%s allocated=%.0fMiB",
            worker_id,
            avail,
            device or "自动",
            name,
            alloc_mib,
        )
    except Exception:
        wlog.debug("worker-%d CUDA 探测跳过", worker_id, exc_info=True)


def _stdio_log_streams() -> list[Any]:
    """stdout/stderr 指向同一文件（如 ``> web.log 2>&1``）时只保留一个，避免重复日志。"""
    import os

    streams: list[Any] = []
    seen: set[tuple[int, int]] = set()
    for stream in (sys.stdout, sys.stderr):
        try:
            st = os.fstat(stream.fileno())
            key = (st.st_dev, st.st_ino)
        except (OSError, AttributeError, ValueError):
            key = (id(stream), 0)
        if key in seen:
            continue
        seen.add(key)
        streams.append(stream)
    return streams or [sys.stderr]


def configure_stdio_logging(level: int = logging.INFO) -> None:
    """配置 root logger：去重后的 stdout/stderr 各至多一个 StreamHandler。"""
    root = logging.getLogger()
    root.setLevel(level)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in list(root.handlers):
        root.removeHandler(h)
    for stream in _stdio_log_streams():
        handler = logging.StreamHandler(stream)
        handler.setLevel(level)
        handler.setFormatter(fmt)
        root.addHandler(handler)


def _configure_worker_logging(level: int = logging.INFO) -> None:
    """spawn 子进程内配置日志（spawn 不继承父进程 handler）。"""
    root = logging.getLogger()
    if root.handlers:
        return
    configure_stdio_logging(level)


def _worker_main(
    worker_id: int,
    config_path: str,
    device: str | None,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
) -> None:
    """子进程入口：加载 ``InsectPredictAll`` 并循环处理推理任务。"""
    _ensure_insect_root_on_syspath()
    _configure_worker_logging()
    wlog = logging.getLogger(f"insect.worker.{worker_id}")
    pipeline = None
    try:
        from script.config_paths import resolve_insect_alg_all_path
        from script.predict_all import InsectPredictAll

        t0 = time.perf_counter()
        cfg_path = resolve_insect_alg_all_path(config_path)
        wlog.info("worker-%d 开始加载配置 %s device=%s", worker_id, cfg_path.name, device or "自动")
        pipeline = InsectPredictAll(cfg_path, device=device)
        warmup = pipeline.warmup_models()
        load_s = time.perf_counter() - t0
        _log_worker_cuda(wlog, worker_id, device)
        wlog.info(
            "worker-%d 模型预加载完成 %.2fs；根模型=%s warmup=%s",
            worker_id,
            load_s,
            list(pipeline._roots.keys()),  # noqa: SLF001
            warmup,
        )
        result_queue.put(
            (
                _WORKER_READY,
                worker_id,
                True,
                None,
                {"load_s": load_s, "roots": list(pipeline._roots.keys()), "warmup": warmup},
            )
        )
    except Exception as exc:
        wlog.exception("模型加载失败")
        result_queue.put((_WORKER_READY, worker_id, False, repr(exc)))
        return

    while True:
        try:
            item = task_queue.get()
        except (EOFError, KeyboardInterrupt):
            break
        if item is None:
            break
        task_id, image_ref, kwargs = item
        try:
            wlog.info(
                "worker-%d 开始推理 task=%s ref=%s",
                worker_id,
                str(task_id)[:8],
                (image_ref[:96] + "…") if len(image_ref) > 96 else image_ref,
            )
            t0 = time.perf_counter()
            out = pipeline.predict_ref(image_ref, **kwargs)
            wlog.info(
                "worker-%d 推理完成 task=%s 耗时 %.2fs",
                worker_id,
                str(task_id)[:8],
                time.perf_counter() - t0,
            )
            result_queue.put((task_id, True, out, None))
        except Exception as exc:
            wlog.exception("推理失败 task=%s", task_id)
            result_queue.put((task_id, False, None, repr(exc)))

    if pipeline is not None:
        try:
            pipeline.release()
        except Exception:
            wlog.warning("释放管线异常", exc_info=True)


class InferenceProcessPool:
    """
    多进程推理池：``run_count`` 个工作进程各持一份模型，主进程按负载分发任务。

    任务队列仅传递图片 URL/本地路径字符串，worker 自行读图，避免 pickle 大图。
    仅当 ``predict_cfg.run_count`` > 1 时使用；``run_count == 1`` 时沿用单进程实现。
    """

    def __init__(
        self,
        config_path: str | Path,
        *,
        device: str | None = None,
        run_count: int = 2,
        predict_timeout: float = 600.0,
        startup_timeout: float = 600.0,
    ) -> None:
        self._config_path = str(config_path)
        self._device = device
        self._run_count = max(2, int(run_count))
        self._predict_timeout = float(predict_timeout)
        self._ctx = mp.get_context("spawn")
        self._result_queue: mp.Queue = self._ctx.Queue()
        self._task_queues: list[mp.Queue] = []
        self._workers: list[mp.Process] = []
        self._rr_index = 0
        self._rr_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._pending: dict[str, tuple[threading.Event, list[Any]]] = {}
        self._closed = False
        self._inflight = [0] * self._run_count
        self._inflight_lock = threading.Lock()

        for i in range(self._run_count):
            tq = self._ctx.Queue()
            proc = self._ctx.Process(
                target=_worker_main,
                args=(i, self._config_path, self._device, tq, self._result_queue),
                name=f"insect-infer-{i}",
                daemon=True,
            )
            self._task_queues.append(tq)
            self._workers.append(proc)
            proc.start()

        self._wait_workers_ready(timeout=startup_timeout)
        self._collector = threading.Thread(
            target=self._collect_results,
            name="insect-infer-collector",
            daemon=True,
        )
        self._collector.start()
        atexit.register(self.close)
        logger.info(
            "InferenceProcessPool 已启动 %d 个工作进程（config=%s device=%s pids=%s）",
            self._run_count,
            Path(self._config_path).name,
            device or "自动",
            [p.pid for p in self._workers],
        )

    def workers_alive(self) -> bool:
        return bool(self._workers) and all(p.is_alive() for p in self._workers)

    def worker_pids(self) -> list[int]:
        return [int(p.pid) for p in self._workers if p.pid is not None]

    def _wait_workers_ready(self, *, timeout: float) -> None:
        deadline = time.perf_counter() + timeout
        ready = 0
        errors: list[str] = []
        while ready < self._run_count:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                msg = self._result_queue.get(timeout=remaining)
            except queue.Empty:
                break
            if not msg or msg[0] != _WORKER_READY:
                continue
            _, wid, ok, err, *rest = msg
            extra = rest[0] if rest and isinstance(rest[0], dict) else {}
            if ok:
                ready += 1
                proc = self._workers[wid] if 0 <= wid < len(self._workers) else None
                logger.info(
                    "[warmup] worker-%d(pid=%s) 模型预加载完成 %.2fs 根模型=%s",
                    wid,
                    getattr(proc, "pid", "?"),
                    float(extra.get("load_s", 0.0) or 0.0),
                    extra.get("roots") or [],
                )
            else:
                errors.append(f"worker-{wid}: {err}")
        if errors:
            self.close()
            raise RuntimeError("推理工作进程启动失败: " + "; ".join(errors))
        if ready < self._run_count:
            self.close()
            raise RuntimeError(
                f"推理工作进程启动超时: {ready}/{self._run_count} 就绪"
            )

    def _collect_results(self) -> None:
        while True:
            try:
                msg = self._result_queue.get()
            except (EOFError, KeyboardInterrupt):
                break
            if not msg:
                continue
            if msg[0] == _WORKER_READY:
                continue
            task_id, ok, payload, err = msg
            with self._pending_lock:
                entry = self._pending.pop(task_id, None)
            if entry is None:
                continue
            event, holder = entry
            holder.extend([ok, payload, err])
            event.set()

    def _pick_worker_index(self) -> int:
        """优先分发给排队更短 / 未在忙的 worker（单 GPU 下减少无效堆队）。"""
        with self._inflight_lock:
            inflight = list(self._inflight)
        candidates = [
            (i, inflight[i], self._task_queues[i].qsize())
            for i in range(self._run_count)
        ]
        # inflight=0 优先，其次 queue 更短，再 round-robin 打破平局
        with self._rr_lock:
            rr = self._rr_index
            self._rr_index += 1
        candidates.sort(key=lambda x: (x[1], x[2], (x[0] - rr) % self._run_count))
        return candidates[0][0]

    def predict(self, image_ref: str, **kwargs: Any) -> Any:
        """向 worker 分发推理任务；``image_ref`` 为 URL 或本地路径（不经 Queue 序列化像素）。"""
        if self._closed:
            raise RuntimeError("InferenceProcessPool 已关闭")
        ref = str(image_ref or "").strip()
        if not ref:
            if kwargs.get("collect_filtered"):
                return [], []
            return []
        task_id = uuid.uuid4().hex
        event = threading.Event()
        holder: list[Any] = []
        with self._pending_lock:
            self._pending[task_id] = (event, holder)
        idx = self._pick_worker_index()
        with self._inflight_lock:
            self._inflight[idx] += 1
        logger.info(
            "[pool] 分发 task=%s -> worker-%d（inflight=%s queue=%s）",
            task_id[:8],
            idx,
            self._inflight,
            [q.qsize() for q in self._task_queues],
        )
        self._task_queues[idx].put((task_id, ref, kwargs))
        try:
            if not event.wait(timeout=self._predict_timeout):
                with self._pending_lock:
                    self._pending.pop(task_id, None)
                raise TimeoutError(f"推理超时（>{self._predict_timeout}s）")
            ok, payload, err = holder[0], holder[1], holder[2]
            if not ok:
                raise RuntimeError(err or "推理失败")
            return payload
        finally:
            with self._inflight_lock:
                self._inflight[idx] = max(0, self._inflight[idx] - 1)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for tq in self._task_queues:
            try:
                tq.put(None)
            except Exception:
                pass
        for proc in self._workers:
            if not proc.is_alive():
                continue
            logger.info(
                "正在关闭推理工作进程 pid=%s name=%s",
                proc.pid,
                proc.name,
            )
            proc.join(timeout=5.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2.0)
            if proc.is_alive() and hasattr(proc, "kill"):
                proc.kill()
                proc.join(timeout=2.0)
        for tq in self._task_queues:
            try:
                tq.close()
                tq.join_thread()
            except Exception:
                pass
        try:
            self._result_queue.close()
            self._result_queue.join_thread()
        except Exception:
            pass
        logger.info("InferenceProcessPool 已关闭")
