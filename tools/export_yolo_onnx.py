#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Ultralytics YOLO .pt → ONNX。

由 ``train/train_models/yolo_quantize.py`` 的 export 路径迁入。
检测 / 分类 / 分割均走 ``YOLO.export(format="onnx")``。
改底部变量后直接运行，不解析命令行。
"""
from __future__ import annotations

import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, Union

import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import strip_optimizer

logger = logging.getLogger(__name__)

TaskName = Literal["detect", "classify", "segment"]
QuantMode = Literal["fp16_pt", "export"]

_CUSTOM_LOSS_STUBS = ("BCEDiceLoss", "MultiChannelDiceLoss")

TASK_ALIASES: dict[str, TaskName] = {
    "detect": "detect",
    "detection": "detect",
    "det": "detect",
    "框选": "detect",
    "cls": "classify",
    "classify": "classify",
    "classification": "classify",
    "分类": "classify",
    "seg": "segment",
    "segment": "segment",
    "segmentation": "segment",
    "分割": "segment",
}

INT8_EXPORT_FORMATS = frozenset(
    {"engine", "openvino", "coreml", "tflite", "saved_model", "tfjs", "edgetpu", "imx", "axelera", "deepx", "mnn"}
)
HALF_EXPORT_FORMATS = frozenset(
    {
        "torchscript",
        "onnx",
        "openvino",
        "engine",
        "coreml",
        "tflite",
        "tfjs",
        "ncnn",
        "mnn",
    }
)


def _register_custom_ultralytics_loss_stubs() -> None:
    import ultralytics.utils.loss as ul_loss

    class _PickleStub:
        def __init__(self, *args, **kwargs) -> None:
            pass

    for name in _CUSTOM_LOSS_STUBS:
        if not hasattr(ul_loss, name):
            setattr(ul_loss, name, type(name, (_PickleStub,), {}))


def _normalize_task(task: str | None) -> TaskName | None:
    if task is None or not str(task).strip():
        return None
    key = str(task).strip().lower()
    if key not in TASK_ALIASES:
        raise ValueError(f"未知 task={task!r}，可选: {sorted(set(TASK_ALIASES.values()))} 或别名 {sorted(TASK_ALIASES)}")
    return TASK_ALIASES[key]


def _guess_task_from_path(weights: Path) -> TaskName | None:
    name = weights.name.lower()
    if "-cls" in name or "cls" in name.split("_") or "classify" in name:
        return "classify"
    if "-seg" in name or "segment" in name or name.endswith("-seg.pt"):
        return "segment"
    return None


def _guess_task_from_checkpoint(weights: Path) -> TaskName | None:
    _register_custom_ultralytics_loss_stubs()
    try:
        ckpt = torch.load(str(weights), map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(str(weights), map_location="cpu")
    except Exception:
        return None
    if not isinstance(ckpt, dict):
        return None
    args = ckpt.get("train_args") or {}
    if isinstance(args, dict):
        t = args.get("task")
        if t in ("detect", "classify", "segment"):
            return t  # type: ignore[return-value]
    model = ckpt.get("model")
    if model is not None:
        cls_name = model.__class__.__name__.lower()
        if "classify" in cls_name or "classification" in cls_name:
            return "classify"
        if "segment" in cls_name:
            return "segment"
        if "detect" in cls_name:
            return "detect"
    return None


def resolve_yolo_task(weights: str | Path, task: str | None = None) -> TaskName:
    """解析 YOLO 任务类型：显式 task > ckpt train_args > 文件名启发，默认 detect。"""
    explicit = _normalize_task(task)
    if explicit is not None:
        return explicit
    path = Path(weights).expanduser().resolve()
    from_ckpt = _guess_task_from_checkpoint(path)
    if from_ckpt is not None:
        return from_ckpt
    from_name = _guess_task_from_path(path)
    if from_name is not None:
        return from_name
    return "detect"


def strip_checkpoint(src: str | Path, dst: str | Path | None = None) -> Path:
    """按 Ultralytics best.pt 逻辑去掉 optimizer，EMA 替换 model 并 FP16 固化。"""
    _register_custom_ultralytics_loss_stubs()
    src_p = Path(src).expanduser().resolve()
    if not src_p.is_file():
        raise FileNotFoundError(src_p)
    dst_p = Path(dst).expanduser().resolve() if dst else src_p.with_name(f"{src_p.stem}_strip.pt")
    dst_p.parent.mkdir(parents=True, exist_ok=True)
    strip_optimizer(str(src_p), str(dst_p))
    mb = dst_p.stat().st_size / 1e6
    logger.info("strip 完成: %s (%.1f MB)", dst_p, mb)
    return dst_p


def _default_fp16_output(weights: Path, output: str | Path | None) -> Path:
    if output:
        return Path(output).expanduser().resolve()
    return weights.parent / f"{weights.stem}_fp16.pt"


def _yolo_load_task(task: TaskName) -> str:
    if task == "classify":
        return "classify"
    return task


def quantize_fp16_pt(
    weights: str | Path,
    *,
    task: str | None = None,
    output: str | Path | None = None,
    strip_first: bool = True,
    verify_load: bool = True,
) -> Path:
    """生成可直接 YOLO() 加载的 FP16 推理权重 .pt。"""
    task_resolved = resolve_yolo_task(weights, task)
    src = Path(weights).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(src)

    out = _default_fp16_output(src, output)
    out.parent.mkdir(parents=True, exist_ok=True)

    if strip_first:
        strip_checkpoint(src, out)
    elif src.resolve() != out.resolve():
        shutil.copy2(src, out)
    else:
        out = src

    if verify_load:
        YOLO(str(out), task=_yolo_load_task(task_resolved))

    mb = out.stat().st_size / 1e6
    logger.info("FP16 PT 已写出: %s (%.1f MB), task=%s", out, mb, task_resolved)
    return out


def build_calib_data_yaml(
    images_dir: str | Path,
    *,
    output_yaml: str | Path | None = None,
    class_names: Sequence[str] | None = None,
) -> Path:
    img_root = Path(images_dir).expanduser().resolve()
    if not img_root.is_dir():
        raise NotADirectoryError(img_root)

    names = list(class_names) if class_names else ["object"]
    lines = [
        f"path: {img_root.parent.as_posix()}",
        f"train: {img_root.name}",
        f"val: {img_root.name}",
        "names:",
    ]
    for i, n in enumerate(names):
        lines.append(f"  {i}: {n}")

    if output_yaml:
        yaml_path = Path(output_yaml).expanduser().resolve()
    else:
        yaml_path = Path(tempfile.gettempdir()) / f"insect_calib_{img_root.name}.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("校准 data.yaml: %s", yaml_path)
    return yaml_path


def _resolve_calib_data(data: str, calib_images_dir: str) -> str:
    data = (data or "").strip()
    calib_images_dir = (calib_images_dir or "").strip()
    if data:
        p = Path(data).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"校准 data yaml 不存在: {p}")
        return str(p)
    if calib_images_dir:
        return str(build_calib_data_yaml(calib_images_dir))
    return "coco8.yaml"


def _validate_export_flags(export_format: str, half: bool, int8: bool) -> None:
    fmt = export_format.lower()
    if half and int8:
        raise ValueError("half 与 int8 不能同时为 True（Ultralytics 互斥）")
    if int8 and fmt not in INT8_EXPORT_FORMATS:
        raise ValueError(f"format={fmt} 不支持 int8，可选: {sorted(INT8_EXPORT_FORMATS)}")
    if half and fmt not in HALF_EXPORT_FORMATS:
        raise ValueError(f"format={fmt} 不支持 half，可选: {sorted(HALF_EXPORT_FORMATS)}")


def export_quantized(
    weights: str | Path,
    *,
    task: str | None = None,
    export_format: str = "onnx",
    imgsz: Union[int, tuple[int, int]] = 640,
    half: bool = False,
    int8: bool = False,
    data: str = "",
    calib_images_dir: str = "",
    fraction: float = 0.2,
    dynamic: bool = False,
    simplify: bool = True,
    workspace: float | None = 4.0,
    batch: int = 1,
    device: str | None = None,
    strip_first: bool = True,
    opset: int | None = 17,
    **export_kwargs: Any,
) -> Path:
    """导出量化/半精度模型（默认 ONNX；也可 engine / openvino 等）。"""
    fmt = export_format.lower().strip()
    _validate_export_flags(fmt, half, int8)
    _register_custom_ultralytics_loss_stubs()

    task_resolved = resolve_yolo_task(weights, task)
    src = Path(weights).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(src)

    if strip_first:
        tmp = src.with_name(f"{src.stem}_export_strip.pt")
        load_path = strip_checkpoint(src, tmp)
    else:
        load_path = src
    model = YOLO(str(load_path), task=_yolo_load_task(task_resolved))

    export_args: dict[str, Any] = {
        "format": fmt,
        "imgsz": imgsz,
        "half": half,
        "int8": int8,
        "dynamic": dynamic,
        "simplify": simplify,
        "batch": batch,
    }
    if opset is not None and fmt == "onnx":
        export_args["opset"] = int(opset)
    if workspace is not None and fmt == "engine":
        export_args["workspace"] = workspace
    if device:
        export_args["device"] = device
    if int8:
        export_args["data"] = _resolve_calib_data(data, calib_images_dir)
        export_args["fraction"] = fraction
    if export_kwargs:
        export_args.update(export_kwargs)

    logger.info(
        "开始 export: format=%s task=%s half=%s int8=%s imgsz=%s opset=%s",
        fmt,
        task_resolved,
        half,
        int8,
        imgsz,
        export_args.get("opset"),
    )
    t0 = time.time()
    exported: str | Path | None = None
    try:
        exported = model.export(**export_args)
    finally:
        if strip_first and load_path != src and load_path.name.endswith("_export_strip.pt"):
            try:
                load_path.unlink()
            except OSError:
                pass
    if exported is None:
        raise RuntimeError("model.export 未返回路径")
    logger.info("export 完成 (%.2fs): %s", time.time() - t0, exported)
    return Path(exported)


def export_yolo_onnx(
    weights: str | Path,
    *,
    task: str | None = None,
    output: str | Path | None = None,
    imgsz: Union[int, tuple[int, int]] = 640,
    opset: int = 17,
    half: bool = False,
    dynamic: bool = False,
    simplify: bool = True,
    batch: int = 1,
    device: str | None = None,
    strip_first: bool = True,
    **export_kwargs: Any,
) -> Path:
    """YOLO .pt → ONNX。默认写在权重同目录同主文件名。"""
    out = export_quantized(
        weights,
        task=task,
        export_format="onnx",
        imgsz=imgsz,
        half=half,
        int8=False,
        dynamic=dynamic,
        simplify=simplify,
        batch=batch,
        device=device,
        strip_first=strip_first,
        opset=opset,
        **export_kwargs,
    )
    if not output:
        return out
    dest = Path(output).expanduser().resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if out.resolve() == dest.resolve():
        return out
    if dest.is_file():
        dest.unlink()
    shutil.copy2(out, dest)
    logger.info("已复制 ONNX 到: %s", dest)
    return dest


def benchmark_pt_inference(
    weights: str | Path,
    *,
    task: str | None = None,
    imgsz: int = 640,
    device: str = "cpu",
    warmup: int = 2,
    repeats: int = 10,
) -> dict[str, float]:
    task_resolved = resolve_yolo_task(weights, task)
    path = Path(weights).expanduser().resolve()
    model = YOLO(str(path), task=_yolo_load_task(task_resolved))

    x = torch.randn(1, 3, imgsz, imgsz)

    def _time_run(use_half: bool) -> float:
        with torch.no_grad():
            for _ in range(warmup):
                model.predict(x, imgsz=imgsz, device=device, half=use_half, verbose=False)
        t0 = time.time()
        with torch.no_grad():
            for _ in range(repeats):
                model.predict(x, imgsz=imgsz, device=device, half=use_half, verbose=False)
        return (time.time() - t0) / repeats

    fp32_t = _time_run(False)
    try:
        fp16_t = _time_run(True)
    except Exception:
        fp16_t = float("nan")

    return {"fp32_s": fp32_t, "fp16_s": fp16_t, "speedup": fp32_t / fp16_t if fp16_t else float("nan")}


def run_quantize(
    weights: str | Path,
    *,
    task: str | None = None,
    mode: QuantMode = "export",
    output: str | Path | None = None,
    export_format: str = "onnx",
    imgsz: Union[int, tuple[int, int]] = 640,
    half: bool = False,
    int8: bool = False,
    data: str = "",
    calib_images_dir: str = "",
    fraction: float = 0.2,
    strip_first: bool = True,
    dynamic: bool = False,
    device: str | None = None,
    benchmark: bool = False,
    opset: int | None = 17,
    **export_kwargs: Any,
) -> Path | Mapping[str, float]:
    if mode == "fp16_pt":
        out = quantize_fp16_pt(
            weights,
            task=task,
            output=output,
            strip_first=strip_first,
        )
        if benchmark:
            stats = benchmark_pt_inference(out, task=task, imgsz=int(imgsz) if isinstance(imgsz, int) else imgsz[0])
            logger.info("benchmark: %s", stats)
        return out

    if mode == "export":
        if int8 and not data and not calib_images_dir:
            logger.warning("INT8 未指定 data/calib_images_dir，将使用 coco8.yaml 作校准（可能与昆虫数据分布不符）")
        if str(export_format).lower().strip() == "onnx" and not int8:
            extra = dict(export_kwargs)
            batch = int(extra.pop("batch", 1))
            simplify = bool(extra.pop("simplify", True))
            return export_yolo_onnx(
                weights,
                task=task,
                output=output,
                imgsz=imgsz,
                opset=int(opset or 17),
                half=half,
                dynamic=dynamic,
                batch=batch,
                device=device,
                strip_first=strip_first,
                simplify=simplify,
                **extra,
            )
        out = export_quantized(
            weights,
            task=task,
            export_format=export_format,
            imgsz=imgsz,
            half=half,
            int8=int8,
            data=data,
            calib_images_dir=calib_images_dir,
            fraction=fraction,
            dynamic=dynamic,
            device=device,
            strip_first=strip_first,
            opset=opset,
            **export_kwargs,
        )
        if output:
            dest = Path(output).expanduser().resolve()
            dest.parent.mkdir(parents=True, exist_ok=True)
            if out.is_file():
                shutil.copy2(out, dest)
                logger.info("已复制导出产物到: %s", dest)
                return dest
            if out.is_dir() and not dest.exists():
                shutil.copytree(out, dest)
                logger.info("已复制导出目录到: %s", dest)
                return dest
        return out

    raise ValueError(f"未知 mode={mode!r}，可选: fp16_pt, export")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # 输入：YOLO 检测 / 分类 / 分割 .pt（best.pt 或中间 ckpt）
    INPUT_PATH = "/Users/shunyaoyin/Documents/code/models/daofeishi-detect-0415.pt"
    TASK = None  # None=自动识别；detect | classify | segment
    OUTPUT_ONNX = ""  # 空则与权重同目录同名 .onnx

    IMGSZ = 640  # 分类常用 224/256/512；检测/分割与训练 imgsz 一致
    OPSET = 17
    HALF = False
    DYNAMIC = False
    SIMPLIFY = True
    BATCH = 1
    DEVICE = ""  # 如 "0" / "cpu" / "mps"；空则 Ultralytics 默认
    STRIP_FIRST = True

    out = export_yolo_onnx(
        INPUT_PATH,
        task=TASK,
        output=OUTPUT_ONNX or None,
        imgsz=IMGSZ,
        opset=OPSET,
        half=HALF,
        dynamic=DYNAMIC,
        simplify=SIMPLIFY,
        batch=BATCH,
        device=DEVICE or None,
        strip_first=STRIP_FIRST,
    )
    print(f"完成: {out}")
