#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CUDA 分类 batch crop/pad：整图延迟上传，按 batch 在 GPU 裁剪/补方后**一次 sync** 再回 CPU。

detect/seg 滑窗 tile 仍走 CPU numpy（切片为 view，无 PCIe 往返；逐 tile GPU 下载反而更慢）。
polygon 掩码 crop 仍走 CPU（``model_cls_crop.crop_cls_instance_bgr``）。
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

import numpy as np
import torch

from script.predict.model_cls_crop import cls_crop_rect_for_row, resolve_from_bbox

log = logging.getLogger(__name__)

_gpu_crop_session_ctx: ContextVar["GpuCropSession | None"] = ContextVar(
    "gpu_crop_session", default=None
)


def get_gpu_crop_session() -> GpuCropSession | None:
    return _gpu_crop_session_ctx.get()


def cuda_gpu_crop_available(device: str | None) -> bool:
    if not torch.cuda.is_available():
        return False
    dev = str(device or "cuda:0").strip().lower()
    return dev == "cuda" or dev.startswith("cuda:")


def _resolve_cuda_device(device: str | None) -> torch.device:
    dev = str(device or "cuda:0").strip()
    if dev.lower() == "cuda":
        return torch.device("cuda:0")
    return torch.device(dev)


def cls_job_can_defer_gpu_crop(row: dict[str, Any], cfg: dict[str, Any]) -> bool:
    """bbox 矩形 crop 可 defer 到 GPU batch；polygon 掩码 crop 仍走 CPU。"""
    use_polygon = not resolve_from_bbox(cfg)
    return not (use_polygon and row.get("polygon"))


class GpuCropSession:
    """单张 BGR 图 GPU crop/pad 会话；上传推迟到首次 batch 操作。"""

    __slots__ = (
        "device",
        "_image_bgr",
        "_tensor_bgr",
        "_h",
        "_w",
        "_channels",
    )

    def __init__(self, image_bgr: np.ndarray, *, device: str | None = None):
        if image_bgr is None or getattr(image_bgr, "size", 0) == 0:
            raise ValueError("empty image for GpuCropSession")
        self.device = _resolve_cuda_device(device)
        self._image_bgr = np.ascontiguousarray(image_bgr)
        self._tensor_bgr: torch.Tensor | None = None
        self._h, self._w = int(self._image_bgr.shape[0]), int(self._image_bgr.shape[1])
        if self._image_bgr.ndim == 2:
            self._channels = 1
        elif self._image_bgr.ndim == 3:
            self._channels = int(self._image_bgr.shape[2])
        else:
            raise ValueError(f"unsupported image ndim={self._image_bgr.ndim}")

    def _ensure_bgr_tensor(self) -> torch.Tensor:
        if self._tensor_bgr is None:
            self._tensor_bgr = torch.from_numpy(self._image_bgr).to(
                self.device, non_blocking=True
            )
        return self._tensor_bgr

    def batch_crop_pad_from_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        cfg: dict[str, Any],
        pad_square: bool,
        pad_color_bgr: tuple[int, int, int] = (255, 255, 255),
    ) -> list[np.ndarray | None]:
        """按 row bbox 批量 crop（可选 pad 方），整批仅一次 CUDA sync 后回 CPU。"""
        if not rows:
            return []
        tensor = self._ensure_bgr_tensor()
        gpu_tensors: list[torch.Tensor | None] = []
        for row in rows:
            rect = cls_crop_rect_for_row(
                row, cfg, img_w=self._w, img_h=self._h
            )
            if rect is None:
                gpu_tensors.append(None)
                continue
            xi1, yi1, xi2, yi2 = rect
            patch = self._crop_rect_tensor_on(tensor, xi1, yi1, xi2, yi2)
            if patch is None:
                gpu_tensors.append(None)
                continue
            if pad_square:
                h = int(patch.shape[0])
                w = int(patch.shape[1])
                if h != w:
                    patch, _, _ = self._pad_square_tensor(
                        patch, pad_color_bgr=pad_color_bgr
                    )
            gpu_tensors.append(patch)
        return self._download_tensors_batched(gpu_tensors)

    def _crop_rect_tensor_on(
        self, tensor: torch.Tensor, x1: int, y1: int, x2: int, y2: int
    ) -> torch.Tensor | None:
        xi1 = max(0, int(x1))
        yi1 = max(0, int(y1))
        xi2 = min(self._w, int(x2))
        yi2 = min(self._h, int(y2))
        if xi2 <= xi1 or yi2 <= yi1:
            return None
        return tensor[yi1:yi2, xi1:xi2]

    def _pad_square_tensor(
        self,
        patch: torch.Tensor,
        *,
        pad_color_bgr: tuple[int, int, int] = (255, 255, 255),
    ) -> tuple[torch.Tensor, int, int]:
        h = int(patch.shape[0])
        w = int(patch.shape[1])
        if h == w:
            return patch, 0, 0
        side = max(h, w)
        off_y = (side - h) // 2
        off_x = (side - w) // 2
        if patch.dim() == 3 and int(patch.shape[2]) == 3:
            canvas = torch.empty(
                (side, side, 3), dtype=patch.dtype, device=self.device
            )
            canvas[..., 0] = int(pad_color_bgr[0])
            canvas[..., 1] = int(pad_color_bgr[1])
            canvas[..., 2] = int(pad_color_bgr[2])
            canvas[off_y : off_y + h, off_x : off_x + w] = patch
        elif patch.dim() == 3:
            fill = int(pad_color_bgr[0])
            canvas = torch.full(
                (side, side, int(patch.shape[2])),
                fill,
                dtype=patch.dtype,
                device=self.device,
            )
            canvas[off_y : off_y + h, off_x : off_x + w] = patch
        else:
            fill = int(pad_color_bgr[0])
            canvas = torch.full(
                (side, side), fill, dtype=patch.dtype, device=self.device
            )
            canvas[off_y : off_y + h, off_x : off_x + w] = patch
        return canvas, off_x, off_y

    @staticmethod
    def _download_tensors_batched(
        tensors: list[torch.Tensor | None],
    ) -> list[np.ndarray | None]:
        """非阻塞 D2H，最后统一 ``cuda.synchronize``，避免逐张 ``.numpy()`` 触发同步。"""
        if not tensors:
            return []
        out: list[np.ndarray | None] = []
        pending: list[torch.Tensor] = []
        pending_idx: list[int] = []
        for i, t in enumerate(tensors):
            if t is None:
                out.append(None)
            else:
                out.append(None)
                pending.append(t)
                pending_idx.append(i)
        if not pending:
            return out
        use_cuda = pending[0].is_cuda
        cpu_tensors = [t.detach().cpu() for t in pending]
        if use_cuda:
            torch.cuda.synchronize()
        for idx, cpu_t in zip(pending_idx, cpu_tensors):
            out[idx] = cpu_t.numpy()
        return out


@contextmanager
def gpu_crop_session_scope(
    image_bgr: np.ndarray,
    *,
    device: str | None,
    enabled: bool,
) -> Iterator[GpuCropSession | None]:
    session: GpuCropSession | None = None
    if enabled and cuda_gpu_crop_available(device):
        try:
            session = GpuCropSession(image_bgr, device=device)
        except Exception as exc:
            log.warning("GpuCropSession 初始化失败，回退 CPU crop: %s", exc)
            session = None
    token = _gpu_crop_session_ctx.set(session)
    try:
        yield session
    finally:
        _gpu_crop_session_ctx.reset(token)
