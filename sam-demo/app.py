#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Detail  : SAM2 点提示分割 Gradio Demo（多目标：按目标编号分组提示点）
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np

from sam2_engine import MODEL_PRESETS, Sam2Engine, infer_model_key_from_checkpoint

logger = logging.getLogger(__name__)

_FILE = Path(__file__).resolve()
_DEFAULT_CHECKPOINT_PATH = _FILE.parent / "checkpoints"

_MODEL_CHOICES = [(v["label"], k) for k, v in MODEL_PRESETS.items()]
_POINT_MODES = [("前景点（选中目标）", 1), ("背景点（排除区域）", 0)]


def _point_mode_value(label: str) -> int:
    return {name: value for name, value in _POINT_MODES}.get(label, 1)


class Sam2GradioApp:
    def __init__(
        self,
        *,
        model_key: str = "sam2.1_tiny",
        checkpoint_path: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        self.engine = Sam2Engine(
            model_key=model_key,
            checkpoint_path=checkpoint_path or _DEFAULT_CHECKPOINT_PATH,
            device=device,
        )
        self._checkpoint_path = Path(checkpoint_path or _DEFAULT_CHECKPOINT_PATH)
        self._checkpoint_is_file = self._checkpoint_path.is_file()
        if self._checkpoint_is_file:
            inferred = infer_model_key_from_checkpoint(self._checkpoint_path)
            if inferred and inferred != model_key:
                logger.info(
                    "权重为单文件 %s，模型切换为 %s",
                    self._checkpoint_path.name,
                    inferred,
                )
                self.engine.set_model(inferred)
                model_key = inferred
        self._model_key = model_key

    @staticmethod
    def _empty_prompt_state() -> tuple[list[list[int]], list[int], list[int], int, str]:
        return [], [], [], 1, "请上传图片，选择目标编号后在左侧点击。"

    @staticmethod
    def _format_status(
        points: list[list[int]],
        labels: list[int],
        object_ids: list[int],
        scores: dict[int, float] | None = None,
        *,
        prefix: str = "",
    ) -> str:
        if not points:
            return prefix or "尚无提示点。"

        grouped = Sam2Engine.group_prompts(points, labels, object_ids)
        parts: list[str] = []
        for obj_id in sorted(grouped):
            pts, _ = grouped[obj_id]
            seg = f"目标{obj_id}: {len(pts)}点"
            if scores and obj_id in scores:
                seg += f"，置信度 {scores[obj_id]:.3f}"
            parts.append(seg)
        summary = f"共 {len(grouped)} 个目标，{len(points)} 个提示点（{'；'.join(parts)}）"
        return f"{prefix}{summary}" if prefix else summary

    def _draw_marked(
        self,
        source_rgb: np.ndarray,
        points: list[list[int]],
        labels: list[int],
        object_ids: list[int],
    ) -> np.ndarray:
        return Sam2Engine.draw_points(source_rgb, points, labels, object_ids)

    def _infer(
        self,
        source_rgb: np.ndarray,
        points: list[list[int]],
        labels: list[int],
        object_ids: list[int],
    ) -> tuple[np.ndarray, np.ndarray | None, dict[int, float]]:
        marked = self._draw_marked(source_rgb, points, labels, object_ids)
        if not points:
            return marked, None, {}

        grouped = Sam2Engine.group_prompts(points, labels, object_ids)
        mask_results = self.engine.predict_objects(grouped)
        scores = {obj_id: score for obj_id, (_, score) in mask_results.items()}
        masks = {obj_id: mask for obj_id, (mask, _) in mask_results.items()}
        result = Sam2Engine.overlay_masks(marked, masks)
        return marked, result, scores

    def _pack(
        self,
        source_rgb: np.ndarray | None,
        points: list[list[int]],
        labels: list[int],
        object_ids: list[int],
        active_object_id: int,
        status: str,
        marked: np.ndarray | None = None,
        result: np.ndarray | None = None,
    ) -> tuple[Any, ...]:
        if marked is None:
            marked = source_rgb
        return (
            marked,
            result,
            points,
            labels,
            object_ids,
            active_object_id,
            source_rgb,
            status,
        )

    def on_upload(
        self,
        image: np.ndarray | None,
        model_key: str,
    ) -> tuple[Any, ...]:
        self.engine.set_model(model_key)
        points, labels, object_ids, active_id, status = self._empty_prompt_state()
        if image is None:
            self.engine.set_image(None)
            return self._pack(None, points, labels, object_ids, active_id, status)

        status = (
            "图片已加载。选择目标编号并点击添加提示点，"
            "全部点选完成后点击「开始推理」。"
        )
        return self._pack(image, points, labels, object_ids, active_id, status)

    def on_model_change(
        self,
        model_key: str,
        source_rgb: np.ndarray | None,
        points: list[list[int]] | None,
        labels: list[int] | None,
        object_ids: list[int] | None,
        active_object_id: int | float,
    ) -> tuple[Any, ...]:
        self.engine.set_model(model_key)
        points = list(points or [])
        labels = list(labels or [])
        object_ids = list(object_ids or [])
        active_id = int(active_object_id or 1)
        if source_rgb is None:
            return self._pack(
                None, points, labels, object_ids, active_id, "请先上传图片。"
            )

        marked = self._draw_marked(source_rgb, points, labels, object_ids)
        if not points:
            return self._pack(
                source_rgb,
                points,
                labels,
                object_ids,
                active_id,
                f"已切换模型为 {model_key}，请选择目标编号后点击。",
                marked=marked,
            )

        status = self._format_status(
            points,
            labels,
            object_ids,
            prefix=f"模型 {model_key} 已切换。",
        ) + " 请点击「开始推理」重新生成分割结果。"
        return self._pack(
            source_rgb, points, labels, object_ids, active_id, status, marked=marked
        )

    def on_predict(
        self,
        source_rgb: np.ndarray | None,
        points: list[list[int]] | None,
        labels: list[int] | None,
        object_ids: list[int] | None,
        active_object_id: int | float,
    ) -> tuple[Any, ...]:
        points = list(points or [])
        labels = list(labels or [])
        object_ids = list(object_ids or [])
        active_id = int(active_object_id or 1)
        if source_rgb is None:
            return self._pack(
                None, points, labels, object_ids, active_id, "请先上传图片。"
            )
        if not points:
            return self._pack(
                source_rgb,
                points,
                labels,
                object_ids,
                active_id,
                "尚无提示点，请先在图上点击添加。",
            )

        try:
            self.engine.set_image(source_rgb)
            marked, result, scores = self._infer(source_rgb, points, labels, object_ids)
        except Exception as exc:  # noqa: BLE001
            marked = self._draw_marked(source_rgb, points, labels, object_ids)
            return self._pack(
                source_rgb,
                points,
                labels,
                object_ids,
                active_id,
                f"推理失败: {exc}",
                marked=marked,
            )

        status = self._format_status(
            points, labels, object_ids, scores, prefix="推理完成。"
        )
        return self._pack(
            source_rgb, points, labels, object_ids, active_id, status, marked, result
        )

    def on_select(
        self,
        evt: gr.SelectData,
        source_rgb: np.ndarray | None,
        points: list[list[int]] | None,
        labels: list[int] | None,
        object_ids: list[int] | None,
        point_mode: str,
        active_object_id: int | float,
    ) -> tuple[Any, ...]:
        if source_rgb is None:
            pts, lbs, oids = list(points or []), list(labels or []), list(object_ids or [])
            return self._pack(
                None, pts, lbs, oids, int(active_object_id or 1), "请先上传图片。"
            )

        obj_id = max(1, int(active_object_id or 1))
        point_label = _point_mode_value(point_mode)
        x, y = int(evt.index[0]), int(evt.index[1])

        points = list(points or [])
        labels = list(labels or [])
        object_ids = list(object_ids or [])
        points.append([x, y])
        labels.append(point_label)
        object_ids.append(obj_id)

        marked = self._draw_marked(source_rgb, points, labels, object_ids)
        kind = "前景" if point_label == 1 else "背景"
        prefix = f"目标{obj_id} 已添加{kind}点 ({x}, {y})。"
        status = (
            self._format_status(points, labels, object_ids, prefix=prefix)
            + " 点选完成后点击「开始推理」。"
        )
        return self._pack(
            source_rgb, points, labels, object_ids, obj_id, status, marked, None
        )

    def on_new_object(self, active_object_id: int | float) -> tuple[int, str]:
        new_id = max(1, int(active_object_id or 1)) + 1
        return new_id, f"已切换到目标 {new_id}，请在图上点击该目标的区域。"

    def on_undo(
        self,
        source_rgb: np.ndarray | None,
        points: list[list[int]] | None,
        labels: list[int] | None,
        object_ids: list[int] | None,
        active_object_id: int | float,
    ) -> tuple[Any, ...]:
        points = list(points or [])
        labels = list(labels or [])
        object_ids = list(object_ids or [])
        active_id = int(active_object_id or 1)
        if not points:
            return self._pack(
                source_rgb, points, labels, object_ids, active_id, "没有可撤销的提示点。"
            )

        points.pop()
        labels.pop()
        removed_obj = object_ids.pop()
        if source_rgb is None:
            return self._pack(
                None, points, labels, object_ids, active_id, "已撤销上一个点。"
            )

        if not points:
            return self._pack(
                source_rgb,
                points,
                labels,
                object_ids,
                active_id,
                f"已撤销目标{removed_obj} 的提示点，当前无提示点。",
            )

        marked = self._draw_marked(source_rgb, points, labels, object_ids)
        status = (
            self._format_status(
                points,
                labels,
                object_ids,
                prefix=f"已撤销目标{removed_obj} 上一个点。",
            )
            + " 请点击「开始推理」更新分割结果。"
        )
        return self._pack(
            source_rgb, points, labels, object_ids, active_id, status, marked, None
        )

    def on_clear_object(
        self,
        source_rgb: np.ndarray | None,
        points: list[list[int]] | None,
        labels: list[int] | None,
        object_ids: list[int] | None,
        active_object_id: int | float,
    ) -> tuple[Any, ...]:
        points = list(points or [])
        labels = list(labels or [])
        object_ids = list(object_ids or [])
        active_id = max(1, int(active_object_id or 1))

        if not points:
            return self._pack(
                source_rgb, points, labels, object_ids, active_id, "当前没有任何提示点。"
            )

        kept = [
            (pt, lb, oid)
            for pt, lb, oid in zip(points, labels, object_ids)
            if oid != active_id
        ]
        if len(kept) == len(points):
            return self._pack(
                source_rgb,
                points,
                labels,
                object_ids,
                active_id,
                f"目标 {active_id} 尚无提示点。",
            )

        if kept:
            points, labels, object_ids = map(list, zip(*kept))
        else:
            points, labels, object_ids = [], [], []

        if source_rgb is None:
            return self._pack(
                None, points, labels, object_ids, active_id, f"已清空目标 {active_id}。"
            )

        if not points:
            return self._pack(
                source_rgb,
                points,
                labels,
                object_ids,
                active_id,
                f"已清空目标 {active_id}，当前无提示点。",
            )

        marked = self._draw_marked(source_rgb, points, labels, object_ids)
        status = (
            self._format_status(
                points, labels, object_ids, prefix=f"已清空目标 {active_id}。"
            )
            + " 请点击「开始推理」更新分割结果。"
        )
        return self._pack(
            source_rgb, points, labels, object_ids, active_id, status, marked, None
        )

    def on_clear(
        self,
        source_rgb: np.ndarray | None,
        active_object_id: int | float,
    ) -> tuple[Any, ...]:
        points, labels, object_ids, active_id, status = self._empty_prompt_state()
        active_id = max(1, int(active_object_id or 1))
        status = "已清除全部目标的提示点，请重新选择目标编号后点击。"
        if source_rgb is None:
            return self._pack(None, points, labels, object_ids, active_id, status)
        return self._pack(source_rgb, points, labels, object_ids, active_id, status)

    def build_ui(self) -> gr.Blocks:
        with gr.Blocks(title="SAM2 多目标点提示分割 Demo") as demo:
            gr.Markdown(
                "# SAM2 多目标交互式分割\n"
                "1. 上传图片\n"
                "2. 设置 **当前目标编号**，在左侧点击添加提示点（可切换多个目标）\n"
                "3. **全部点选完成后**，点击 **「开始推理」** 生成分割结果\n"
                "4. 修改提示点后需再次点击「开始推理」更新结果\n\n"
                f"**权重路径**：`{self._checkpoint_path}`"
            )

            points_state = gr.State([])
            labels_state = gr.State([])
            object_ids_state = gr.State([])
            source_state = gr.State(None)

            with gr.Row():
                model_dd = gr.Dropdown(
                    choices=_MODEL_CHOICES,
                    value=self._model_key,
                    label="模型",
                    interactive=not self._checkpoint_is_file,
                )
                active_object = gr.Number(
                    value=1,
                    precision=0,
                    minimum=1,
                    label="当前目标编号",
                )
                point_mode = gr.Radio(
                    choices=[label for label, _ in _POINT_MODES],
                    value=_POINT_MODES[0][0],
                    label="点击类型",
                )

            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(
                        label="原图（点击添加当前目标的提示点，点上会标注目标编号）",
                        type="numpy",
                        interactive=True,
                        height=480,
                    )
                with gr.Column():
                    output_image = gr.Image(
                        label="多目标分割结果",
                        type="numpy",
                        interactive=False,
                        height=480,
                    )

            with gr.Row():
                predict_btn = gr.Button("开始推理", variant="primary")
                new_object_btn = gr.Button("新建目标（编号 +1）")
                undo_btn = gr.Button("撤销上一点")
                clear_object_btn = gr.Button("清空当前目标")
                clear_btn = gr.Button("清空全部目标")

            status = gr.Textbox(label="状态", interactive=False, lines=3)

            main_outputs = [
                input_image,
                output_image,
                points_state,
                labels_state,
                object_ids_state,
                active_object,
                source_state,
                status,
            ]

            input_image.upload(
                self.on_upload,
                inputs=[input_image, model_dd],
                outputs=main_outputs,
            )
            input_image.select(
                self.on_select,
                inputs=[
                    source_state,
                    points_state,
                    labels_state,
                    object_ids_state,
                    point_mode,
                    active_object,
                ],
                outputs=main_outputs,
            )
            model_dd.change(
                self.on_model_change,
                inputs=[
                    model_dd,
                    source_state,
                    points_state,
                    labels_state,
                    object_ids_state,
                    active_object,
                ],
                outputs=main_outputs,
            )
            predict_btn.click(
                self.on_predict,
                inputs=[
                    source_state,
                    points_state,
                    labels_state,
                    object_ids_state,
                    active_object,
                ],
                outputs=main_outputs,
            )
            new_object_btn.click(
                self.on_new_object,
                inputs=[active_object],
                outputs=[active_object, status],
            )
            undo_btn.click(
                self.on_undo,
                inputs=[
                    source_state,
                    points_state,
                    labels_state,
                    object_ids_state,
                    active_object,
                ],
                outputs=main_outputs,
            )
            clear_object_btn.click(
                self.on_clear_object,
                inputs=[
                    source_state,
                    points_state,
                    labels_state,
                    object_ids_state,
                    active_object,
                ],
                outputs=main_outputs,
            )
            clear_btn.click(
                self.on_clear,
                inputs=[source_state, active_object],
                outputs=main_outputs,
            )

        return demo


def create_app(
    *,
    model_key: str | None = "sam2.1_tiny",
    checkpoint_path: str | Path | None = None,
    device: str | None = None,
) -> gr.Blocks:
    ckpt = Path(checkpoint_path) if checkpoint_path else _DEFAULT_CHECKPOINT_PATH
    resolved_model_key = model_key
    if ckpt.is_file() and resolved_model_key is None:
        resolved_model_key = infer_model_key_from_checkpoint(ckpt) or "sam2.1_tiny"
    if resolved_model_key is None:
        resolved_model_key = "sam2.1_tiny"

    app = Sam2GradioApp(
        model_key=resolved_model_key,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    return app.build_ui()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )

    # IDE 直接运行：按需修改以下变量
    MODEL_KEY: str | None = "sam2.1_tiny"  # 传 .pt 文件时可设为 None 自动推断
    # 权重目录或单个 .pt 文件路径（必填，指向你的自定义目录）
    CHECKPOINT_PATH = "/data/models/sam2.1_hiera_small.pt"
    DEVICE: str | None = None  # None / "auto" → 自动选择 cuda / mps / cpu
    SERVER_NAME = "0.0.0.0"
    SERVER_PORT = 37861
    SHARE = False

    demo = create_app(
        model_key=MODEL_KEY,
        checkpoint_path=CHECKPOINT_PATH,
        device=DEVICE,
    )
    demo.queue().launch(
        server_name=SERVER_NAME,
        server_port=SERVER_PORT,
        share=SHARE,
    )
