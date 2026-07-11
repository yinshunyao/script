#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将 YOLO 推理用 .pt 导出为 TensorRT .engine，用于 NVIDIA GPU 部署加速。

依赖：CUDA、TensorRT、ultralytics；须在带 NVIDIA GPU 的机器上执行（Linux / Windows）。
macOS 无官方 TensorRT，本地仅可编辑脚本；实际导出请在部署/训练 GPU 机运行。

支持任务：detect（框选）、segment（分割）、classify（分类）；由 checkpoint 自动识别。
可多 imgsz 各导出一颗 engine，与 insect_alg JSON 中 seg_imgsz / clip_profiles 对齐。

用法：改文件底部 ``if __name__ == "__main__"`` 中的配置后在 IDE 直接运行。
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch


def _preflight_torch_torchvision() -> None:
    """导入 ultralytics 前校验 torch/torchvision 成对匹配，避免 torchvision::nms 注册失败。"""
    try:
        import torchvision
        from torchvision.ops import nms

        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]], dtype=torch.float32)
        scores = torch.tensor([0.9], dtype=torch.float32)
        nms(boxes, scores, 0.5)
    except Exception as exc:
        tv_ver = "未安装"
        try:
            import torchvision as tv

            tv_ver = tv.__version__
        except Exception:
            pass
        raise RuntimeError(
            "torch 与 torchvision 版本/CUDA 构建不匹配，无法加载 ultralytics。\n"
            f"  当前: torch {torch.__version__}, torchvision {tv_ver}\n"
            "  请先成对重装（勿单独升级其一），再执行 acc_tensorRT.py。\n"
            "  CUDA 12.8 示例：\n"
            "    pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 "
            "--find-links https://mirrors.aliyun.com/pytorch-wheels/cu128/ --force-reinstall\n"
            "  CUDA 13 示例：\n"
            "    pip install torch==2.10.0+cu130 torchvision==0.25.0+cu130 "
            "--extra-index-url https://download.pytorch.org/whl/cu130 --force-reinstall\n"
            f"  原始错误: {exc}"
        ) from exc


_preflight_torch_torchvision()

from ultralytics import YOLO
from ultralytics.utils.torch_utils import strip_optimizer

_FILE = Path(__file__).resolve()
_INSECT_ROOT = _FILE.parents[2]
if str(_INSECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_INSECT_ROOT))

# 与 gen_best_pt 一致：含自定义 criterion 的中间 ckpt 反序列化占位
_CUSTOM_LOSS_STUBS = ("BCEDiceLoss", "MultiChannelDiceLoss")

_DEFAULT_PT_GLOBS = ("*.pt",)
_DEFAULT_INTERMEDIATE_GLOBS = ("*-t.pt", "last.pt", "epoch*.pt")
_TRT_EXPORT_PATCHED = False
# dynamic 导出时 ONNX 示踪用 batch=1，TRT profile / metadata 仍用此 max batch
_DYNAMIC_EXPORT_MAX_BATCH: int | None = None


def _register_custom_ultralytics_loss_stubs() -> None:
    import ultralytics.utils.loss as ul_loss

    class _PickleStub:
        def __init__(self, *args, **kwargs) -> None:
            pass

    for name in _CUSTOM_LOSS_STUBS:
        if not hasattr(ul_loss, name):
            setattr(ul_loss, name, type(name, (_PickleStub,), {}))


def _is_macos_resource_fork(path: Path) -> bool:
    return path.name.startswith("._")


def _is_intermediate_name(name: str) -> bool:
    lower = name.lower()
    if lower in {"best.pt"}:
        return False
    if lower.endswith("-best.pt") or lower.endswith("_strip.pt") or lower.endswith("_fp16.pt"):
        return False
    return (
        lower.endswith("-t.pt")
        or lower == "last.pt"
        or (lower.startswith("epoch") and lower.endswith(".pt"))
    )


def _cuda_total_mem_gb(device: int | str | None = 0) -> float | None:
    """当前 CUDA 设备总显存（GiB），不可用则 None。"""
    if not torch.cuda.is_available():
        return None
    try:
        idx = int(device) if device is not None else 0
        props = torch.cuda.get_device_properties(idx)
        return float(props.total_memory) / (1024**3)
    except Exception:
        return None


def _resolve_export_batches(
    *,
    task: str,
    imgsz: int,
    dynamic: bool,
    batch: int,
    device: int | str | None = 0,
) -> tuple[int, int]:
    """
  解析导出用 batch。

  - ``max_batch``：写入 engine metadata / TRT dynamic profile 的最大 batch
  - ``trace_batch``：ONNX 示踪与导出前 warmup 用的 batch（dynamic 且 max>1 时固定为 1，避免 OOM）
    """
    max_batch = max(1, int(batch))
    trace_batch = 1 if (dynamic and max_batch > 1) else max_batch

    mem_gb = _cuda_total_mem_gb(device)
    task_l = str(task or "").lower()
    if task_l == "classify" and int(imgsz) >= 1280 and max_batch > 16:
        hint = 8 if (mem_gb is not None and mem_gb <= 12.5) else 16
        print(
            f"提示: classify imgsz={imgsz} 在 "
            f"{mem_gb:.1f}GB 显存上 max batch={max_batch} 可能过大；"
            f"建议 cls_batch_size / BATCH={hint}~16。"
        )
    if max_batch > 64:
        print(f"警告: BATCH={max_batch} 异常偏大，请确认是否误输入（常见为 8/16/32）。")
    return trace_batch, max_batch


def _require_cuda_device(device: int | str | None) -> int | str:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "未检测到 CUDA。TensorRT 导出须在 NVIDIA GPU 机器上运行；"
            "macOS 请把本脚本拷到 Linux 部署机后修改 INPUT_PATH 再执行。"
        )
    if device is None:
        return 0
    return device


def _normalize_imgsz_list(imgsz_list: list[int] | int | tuple[int, ...]) -> list[int]:
    if isinstance(imgsz_list, int):
        return [int(imgsz_list)]
    out = [int(x) for x in imgsz_list if int(x) > 0]
    if not out:
        raise ValueError("imgsz_list 不能为空，且每项须 > 0")
    return out


def _task_label(yolo_model: YOLO) -> str:
    task = str(getattr(yolo_model, "task", "") or "unknown")
    names = getattr(yolo_model, "names", None)
    n_cls = len(names) if names else "?"
    return f"{task} (classes={n_cls})"


def _patch_ultralytics_tensorrt_export() -> None:
    """修补 ultralytics TensorRT 导出，兼容 TRT 10/11 API 变更。"""
    global _TRT_EXPORT_PATCHED
    if _TRT_EXPORT_PATCHED:
        return

    import json
    from typing import Dict, List, Optional, Tuple

    from ultralytics.utils import IS_JETSON, LOGGER

    def _modelopt_quantize_onnx(
        onnx_path: str,
        *,
        quantize: int,
        dataset=None,
        shape: Tuple[int, int, int, int],
        dynamic: bool,
        prefix: str,
    ) -> str:
        """TRT11：用 ModelOpt 将精度写入 ONNX（FP16 AutoCast / INT8 QDQ）。"""
        if quantize == 8 and dataset is None:
            raise ValueError("INT8 ModelOpt 量化需要校准数据集 (INT8_DATA)。")

        try:
            from ultralytics.utils.checks import check_requirements
        except ImportError:
            check_requirements = None

        if check_requirements is not None:
            check_requirements("nvidia-modelopt[onnx]>=0.44")
        else:
            import importlib.util

            if importlib.util.find_spec("modelopt") is None:
                raise ImportError(
                    'TensorRT 11 导出 FP16/INT8 需安装: pip install "nvidia-modelopt[onnx]>=0.44"'
                )

        import onnx
        import torch

        input_name = onnx.load(onnx_path, load_external_data=False).graph.input[0].name
        src = Path(onnx_path)

        if quantize == 8:
            from modelopt.onnx.quantization import quantize as modelopt_quantize

            out_file = str(src.with_suffix(".int8.onnx"))
            images, n = [], 0
            for batch in dataset:
                images.append(batch["img"])
                n += images[-1].shape[0]
                if n >= 512:
                    break
            calib = torch.cat(images).to(torch.float32) / 255.0
            LOGGER.info(
                f"{prefix} ModelOpt INT8 量化 ONNX，校准图 {calib.shape[0]} 张..."
            )
            kwargs = (
                {"calibration_shapes": f"{input_name}:{'x'.join(str(d) for d in shape)}"}
                if dynamic
                else {}
            )
            modelopt_quantize(
                onnx_path,
                quantize_mode="int8",
                calibration_data={input_name: calib.cpu().numpy()},
                calibration_method="max",
                calibration_eps=["cpu"],
                output_path=out_file,
                **kwargs,
            )
            return out_file

        import modelopt.onnx.autocast as autocast

        out_file = str(src.with_suffix(".fp16.onnx"))
        LOGGER.info(f"{prefix} ModelOpt FP16 混合精度转换 ONNX...")
        onnx.save(
            autocast.convert_to_mixed_precision(
                onnx_path,
                low_precision_type="fp16",
                keep_io_types=True,
                calibration_data={input_name: torch.randn(*shape).cpu().numpy()},
            ),
            out_file,
        )
        return out_file

    def _onnx2engine_compat(
        onnx_file: str,
        output_file: str | Path | None = None,
        workspace: int | None = None,
        half: bool = False,
        int8: bool = False,
        dynamic: bool = False,
        shape: Tuple[int, int, int, int] = (1, 3, 640, 640),
        dla: int | None = None,
        dataset=None,
        metadata: Dict | None = None,
        verbose: bool = False,
        prefix: str = "",
        engine_file: str | Path | None = None,
        quantize: int | str | None = None,
        **_kwargs: Any,
    ) -> str:
        import tensorrt as trt  # noqa

        global _DYNAMIC_EXPORT_MAX_BATCH
        if dynamic and _DYNAMIC_EXPORT_MAX_BATCH and int(_DYNAMIC_EXPORT_MAX_BATCH) > 1:
            max_b = int(_DYNAMIC_EXPORT_MAX_BATCH)
            if int(shape[0]) < max_b:
                shape = (max_b, int(shape[1]), int(shape[2]), int(shape[3]))
            if metadata is None:
                metadata = {}
            else:
                metadata = dict(metadata)
            metadata["batch"] = max_b
            LOGGER.info(
                f"{prefix} dynamic 低显存导出: ONNX 示踪 batch=1，engine max batch={max_b} shape={shape}"
            )

        out_path = Path(output_file or engine_file or Path(onnx_file).with_suffix(".engine"))
        onnx_path = onnx_file

        logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            logger.min_severity = trt.Logger.Severity.VERBOSE

        builder = trt.Builder(logger)
        config = builder.create_builder_config()
        workspace_bytes = int((workspace or 0) * (1 << 30))
        trt_major = int(str(trt.__version__).split(".", 1)[0])
        is_trt10 = trt_major >= 10
        is_trt11 = trt_major >= 11

        if workspace_bytes > 0:
            if hasattr(config, "set_memory_pool_limit"):
                config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
            else:
                config.max_workspace_size = workspace_bytes

        if is_trt10:
            network = builder.create_network()
        else:
            flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            network = builder.create_network(flag)

        if quantize in (8, "8"):
            use_int8, use_fp16 = True, False
        elif quantize in (16, "16"):
            use_int8, use_fp16 = False, True
        else:
            use_fp16 = getattr(builder, "platform_has_fast_fp16", True) and half
            use_int8 = getattr(builder, "platform_has_fast_int8", True) and int8

        if use_int8 and dataset is None:
            raise ValueError("INT8 TensorRT export requires a calibration dataset (data=...).")

        if dla is not None:
            if not IS_JETSON:
                raise ValueError("DLA is only available on NVIDIA Jetson devices")
            if is_trt11:
                raise ValueError("TensorRT 11.0 不支持 DLA，请使用 TensorRT 10.x 导出。")
            LOGGER.info(f"{prefix} enabling DLA on core {dla}...")
            if not use_fp16 and not use_int8:
                raise ValueError(
                    "DLA requires either 'half=True' (FP16) or 'int8=True' (INT8) to be enabled."
                )
            config.default_device_type = trt.DeviceType.DLA
            config.DLA_core = int(dla)
            config.set_flag(trt.BuilderFlag.GPU_FALLBACK)

        # TRT11：精度须先写入 ONNX，不能再 set_flag(FP16/INT8)
        if is_trt11 and (use_fp16 or use_int8):
            q = 8 if use_int8 else 16
            onnx_path = _modelopt_quantize_onnx(
                onnx_path,
                quantize=q,
                dataset=dataset,
                shape=shape,
                dynamic=dynamic,
                prefix=prefix,
            )

        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(onnx_path):
            raise RuntimeError(f"failed to load ONNX file: {onnx_path}")

        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]
        for inp in inputs:
            LOGGER.info(f'{prefix} input "{inp.name}" with shape{inp.shape} {inp.dtype}')
        for out in outputs:
            LOGGER.info(f'{prefix} output "{out.name}" with shape{out.shape} {out.dtype}')

        if dynamic:
            if shape[0] <= 1:
                LOGGER.warning(f"{prefix} 'dynamic=True' model requires max batch size, i.e. 'batch=16'")
            profile = builder.create_optimization_profile()
            min_shape = (1, shape[1], 32, 32)
            max_shape = (*shape[:2], *(int(max(2, workspace or 2) * d) for d in shape[2:]))
            for inp in inputs:
                profile.set_shape(inp.name, min=min_shape, opt=shape, max=max_shape)
            config.add_optimization_profile(profile)
            if use_int8 and not is_trt10:
                config.set_calibration_profile(profile)

        LOGGER.info(
            f"{prefix} building {'INT8' if use_int8 else 'FP' + ('16' if use_fp16 else '32')} "
            f"engine as {out_path} (TensorRT {trt.__version__})"
        )

        has_int8_calibrator = hasattr(trt, "IInt8Calibrator") and hasattr(trt.BuilderFlag, "INT8")
        has_fp16_flag = hasattr(trt.BuilderFlag, "FP16")

        if use_int8 and not is_trt11 and has_int8_calibrator:
            config.set_flag(trt.BuilderFlag.INT8)
            config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

            class EngineCalibrator(trt.IInt8Calibrator):
                def __init__(self, calib_dataset, cache: str = "") -> None:
                    trt.IInt8Calibrator.__init__(self)
                    self.dataset = calib_dataset
                    self.data_iter = iter(calib_dataset)
                    self.algo = (
                        trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2
                        if dla is not None
                        else trt.CalibrationAlgoType.MINMAX_CALIBRATION
                    )
                    self.batch = calib_dataset.batch_size
                    self.cache = Path(cache)

                def get_algorithm(self) -> trt.CalibrationAlgoType:
                    return self.algo

                def get_batch_size(self) -> int:
                    return self.batch or 1

                def get_batch(self, names) -> Optional[List[int]]:
                    try:
                        im0s = next(self.data_iter)["img"] / 255.0
                        im0s = im0s.to("cuda") if im0s.device.type == "cpu" else im0s
                        return [int(im0s.data_ptr())]
                    except StopIteration:
                        return None

                def read_calibration_cache(self) -> Optional[bytes]:
                    if self.cache.exists() and self.cache.suffix == ".cache":
                        return self.cache.read_bytes()
                    return None

                def write_calibration_cache(self, cache: bytes) -> None:
                    _ = self.cache.write_bytes(cache)

            config.int8_calibrator = EngineCalibrator(
                dataset=dataset,
                cache=str(Path(onnx_path).with_suffix(".cache")),
            )
        elif use_fp16 and not is_trt11 and has_fp16_flag:
            config.set_flag(trt.BuilderFlag.FP16)

        if hasattr(builder, "build_serialized_network"):
            engine = builder.build_serialized_network(network, config)
            if engine is None:
                raise RuntimeError("TensorRT engine build failed, check logs for errors")
            with open(out_path, "wb") as t:
                if metadata is not None:
                    meta = json.dumps(metadata)
                    t.write(len(meta).to_bytes(4, byteorder="little", signed=True))
                    t.write(meta.encode())
                t.write(engine)
        else:
            with builder.build_engine(network, config) as engine, open(out_path, "wb") as t:
                if metadata is not None:
                    meta = json.dumps(metadata)
                    t.write(len(meta).to_bytes(4, byteorder="little", signed=True))
                    t.write(meta.encode())
                t.write(engine.serialize())
        return str(out_path)

    targets: list[Any] = []
    try:
        import ultralytics.utils.export.engine as engine_mod

        targets.append(engine_mod)
    except ImportError:
        pass
    try:
        import ultralytics.utils.export as export_mod

        targets.append(export_mod)
    except ImportError:
        pass

    for mod in targets:
        if hasattr(mod, "onnx2engine"):
            mod.onnx2engine = _onnx2engine_compat
        if hasattr(mod, "export_engine"):
            mod.export_engine = _onnx2engine_compat

    try:
        import ultralytics.engine.exporter as exporter_mod

        exporter_mod.export_engine = _onnx2engine_compat
    except ImportError:
        pass

    _TRT_EXPORT_PATCHED = True
    print("已应用 TensorRT 导出兼容补丁（TRT 10/11：EXPLICIT_BATCH、FP16/INT8）。")


def _build_export_kwargs(
    *,
    imgsz: int,
    device: int | str,
    half: bool,
    int8: bool,
    dynamic: bool,
    batch: int,
    workspace: int,
    simplify: bool,
    int8_data: str | Path | None,
    int8_split: str,
    int8_fraction: float,
) -> dict[str, Any]:
    if int8 and half:
        half = False
        print("INT8 与 FP16 互斥，已自动设置 half=False。")

    export_kwargs: dict[str, Any] = {
        "format": "engine",
        "imgsz": int(imgsz),
        "device": device,
        "half": bool(half),
        "int8": bool(int8),
        "dynamic": bool(dynamic),
        "batch": int(batch),
        "workspace": int(workspace),
        "simplify": bool(simplify),
    }

    if int8:
        if not int8_data:
            raise ValueError(
                "INT8=True 时必须配置 INT8_DATA（YOLO 数据集 yaml，含 train/val 图片路径）。"
                " 不建议使用 ultralytics 默认 coco8，会与虫情分布不符。"
            )
        data_path = Path(int8_data).expanduser().resolve()
        if not data_path.is_file():
            raise FileNotFoundError(f"INT8 校准数据集不存在: {data_path}")
        export_kwargs["data"] = str(data_path)
        export_kwargs["split"] = str(int8_split or "val")
        export_kwargs["fraction"] = float(int8_fraction)
        print(
            f"INT8 校准: data={data_path}, split={export_kwargs['split']}, "
            f"fraction={export_kwargs['fraction']}"
        )

    return export_kwargs


def collect_pt_sources(
    root: Path,
    *,
    pt_globs: tuple[str, ...] = _DEFAULT_PT_GLOBS,
    intermediate_globs: tuple[str, ...] = _DEFAULT_INTERMEDIATE_GLOBS,
    include_inference_pt: bool = True,
    inference_only: bool = False,
) -> list[Path]:
    root = Path(root).expanduser().resolve()
    if root.is_file():
        return [root]

    found: dict[str, Path] = {}
    patterns = list(intermediate_globs)
    if include_inference_pt:
        patterns = list(dict.fromkeys([*pt_globs, *intermediate_globs]))

    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if not path.is_file() or _is_macos_resource_fork(path):
                continue
            if inference_only and _is_intermediate_name(path.name):
                continue
            if pattern in pt_globs and pattern not in intermediate_globs:
                if _is_intermediate_name(path.name) and not include_inference_pt:
                    continue
            found[str(path.resolve())] = path
    return sorted(found.values(), key=lambda p: p.name)


def resolve_engine_path(
    src_pt: Path,
    imgsz: int,
    *,
    imgsz_suffix: bool,
    multi_imgsz: bool,
) -> Path:
    stem = src_pt.stem
    if multi_imgsz or imgsz_suffix:
        return src_pt.with_name(f"{stem}-i{imgsz}.engine")
    return src_pt.with_suffix(".engine")


def should_skip(
    src: Path,
    dst: Path,
    *,
    skip_if_up_to_date: bool,
    overwrite: bool,
) -> bool:
    if overwrite:
        return False
    if not dst.is_file():
        return False
    if not skip_if_up_to_date:
        return False
    return dst.stat().st_mtime >= src.stat().st_mtime


def _prepare_export_pt(src: Path, *, strip_intermediate: bool) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    src = src.expanduser().resolve()
    if not strip_intermediate or not _is_intermediate_name(src.name):
        return src, None

    tmp = tempfile.TemporaryDirectory(prefix="insect_trt_strip_")
    dst = Path(tmp.name) / f"{src.stem}_strip.pt"
    _register_custom_ultralytics_loss_stubs()
    ckpt = strip_optimizer(str(src), str(dst))
    if not ckpt or not dst.is_file():
        raise RuntimeError(f"strip_optimizer 失败: {src}")
    return dst, tmp


def export_pt_to_tensorrt(
    src_pt: str | Path,
    *,
    imgsz: int = 640,
    output_path: str | Path | None = None,
    device: int | str | None = 0,
    half: bool = True,
    int8: bool = False,
    dynamic: bool = False,
    batch: int = 1,
    workspace: int = 4,
    simplify: bool = True,
    int8_data: str | Path | None = None,
    int8_split: str = "val",
    int8_fraction: float = 1.0,
    strip_intermediate: bool = True,
    verify_load: bool = True,
    overwrite: bool = False,
) -> Path:
    """
    将单个 YOLO .pt 导出为 TensorRT .engine。

    :return: 最终 .engine 路径（若指定 output_path 则可能从默认导出路径移动/复制过去）
    """
    src = Path(src_pt).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(src)

    dev = _require_cuda_device(device)
    _patch_ultralytics_tensorrt_export()
    export_pt, tmp_holder = _prepare_export_pt(src, strip_intermediate=strip_intermediate)
    try:
        model = YOLO(str(export_pt))
        print(f"加载: {src.name} -> {_task_label(model)}")

        default_engine = src.with_suffix(".engine")
        final = Path(output_path).expanduser().resolve() if output_path else default_engine
        if not overwrite and final.is_file() and final.stat().st_mtime >= src.stat().st_mtime:
            print(f"跳过（已存在且较新）: {final}")
            return final

        task = str(getattr(model, "task", "") or "")
        trace_batch, max_batch = _resolve_export_batches(
            task=task,
            imgsz=int(imgsz),
            dynamic=bool(dynamic),
            batch=int(batch),
            device=dev,
        )
        global _DYNAMIC_EXPORT_MAX_BATCH
        if dynamic and max_batch > 1:
            _DYNAMIC_EXPORT_MAX_BATCH = max_batch
            print(
                f"dynamic 导出: ONNX 示踪/warmup batch={trace_batch}，"
                f"engine max batch={max_batch}（避免 {max_batch}×{imgsz}² 导出 OOM）"
            )
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            export_kwargs = _build_export_kwargs(
                imgsz=imgsz,
                device=dev,
                half=half,
                int8=int8,
                dynamic=dynamic,
                batch=trace_batch,
                workspace=workspace,
                simplify=simplify,
                int8_data=int8_data,
                int8_split=int8_split,
                int8_fraction=int8_fraction,
            )
            print(
                f"导出 TensorRT: imgsz={imgsz}, half={export_kwargs['half']}, "
                f"int8={export_kwargs['int8']}, dynamic={dynamic}, "
                f"trace_batch={trace_batch}, max_batch={max_batch}, device={dev}"
            )
            out = Path(model.export(**export_kwargs))
        finally:
            _DYNAMIC_EXPORT_MAX_BATCH = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if not out.is_file():
            raise FileNotFoundError(f"export 未生成 engine: {out}")

        # ultralytics 固定写在 export 所用权重同目录、同 stem；需重命名时移动
        if out.resolve() != final.resolve():
            final.parent.mkdir(parents=True, exist_ok=True)
            if final.is_file():
                final.unlink()
            shutil.move(str(out), str(final))
            out = final

        src_mb = src.stat().st_size / 1e6
        dst_mb = out.stat().st_size / 1e6
        print(f"已写出: {out} ({dst_mb:.1f} MB, 源 pt {src_mb:.1f} MB)")

        if verify_load:
            YOLO(str(out))
            print(f"试加载通过: {out.name}")

        return out
    finally:
        if tmp_holder is not None:
            tmp_holder.cleanup()


def export_pt_to_tensorrt_batch(
    input_path: str | Path,
    *,
    imgsz_list: list[int] | int | tuple[int, ...] = (640,),
    imgsz_suffix: bool = True,
    device: int | str | None = 0,
    half: bool = True,
    int8: bool = False,
    dynamic: bool = False,
    batch: int = 1,
    workspace: int = 4,
    simplify: bool = True,
    int8_data: str | Path | None = None,
    int8_split: str = "val",
    int8_fraction: float = 1.0,
    strip_intermediate: bool = True,
    verify_load: bool = True,
    skip_if_up_to_date: bool = True,
    overwrite: bool = False,
    pt_globs: tuple[str, ...] = _DEFAULT_PT_GLOBS,
    intermediate_globs: tuple[str, ...] = _DEFAULT_INTERMEDIATE_GLOBS,
    include_inference_pt: bool = True,
    inference_only: bool = False,
) -> list[Path]:
    sizes = _normalize_imgsz_list(imgsz_list)
    multi = len(sizes) > 1
    sources = collect_pt_sources(
        Path(input_path),
        pt_globs=pt_globs,
        intermediate_globs=intermediate_globs,
        include_inference_pt=include_inference_pt,
        inference_only=inference_only,
    )
    if not sources:
        raise FileNotFoundError(f"未找到可导出的 .pt: {input_path}")

    _require_cuda_device(device)
    written: list[Path] = []
    for src in sources:
        for imgsz in sizes:
            dst = resolve_engine_path(src, imgsz, imgsz_suffix=imgsz_suffix, multi_imgsz=multi)
            if should_skip(src, dst, skip_if_up_to_date=skip_if_up_to_date, overwrite=overwrite):
                print(f"跳过（已存在且较新）: {dst}")
                written.append(dst)
                continue
            out = export_pt_to_tensorrt(
                src,
                imgsz=imgsz,
                output_path=dst,
                device=device,
                half=half,
                int8=int8,
                dynamic=dynamic,
                batch=batch,
                workspace=workspace,
                simplify=simplify,
                int8_data=int8_data,
                int8_split=int8_split,
                int8_fraction=int8_fraction,
                strip_intermediate=strip_intermediate,
                verify_load=verify_load,
                overwrite=True,
            )
            written.append(out)
    return written


def main(
    *,
    input_path: str | Path,
    imgsz_list: list[int] | int | tuple[int, ...],
    imgsz_suffix: bool = True,
    device: int | str | None = 0,
    half: bool = True,
    int8: bool = False,
    dynamic: bool = False,
    batch: int = 1,
    workspace_gb: int = 4,
    simplify: bool = True,
    int8_data: str | Path | None = None,
    int8_split: str = "val",
    int8_fraction: float = 1.0,
    strip_intermediate: bool = True,
    verify_load: bool = True,
    skip_if_up_to_date: bool = True,
    overwrite: bool = False,
    inference_only: bool = True,
    include_inference_pt: bool = True,
    pt_globs: tuple[str, ...] = _DEFAULT_PT_GLOBS,
    intermediate_globs: tuple[str, ...] = _DEFAULT_INTERMEDIATE_GLOBS,
) -> list[Path]:
    outputs = export_pt_to_tensorrt_batch(
        input_path,
        imgsz_list=imgsz_list,
        imgsz_suffix=imgsz_suffix,
        device=device,
        half=half,
        int8=int8,
        dynamic=dynamic,
        batch=batch,
        workspace=workspace_gb,
        simplify=simplify,
        int8_data=int8_data,
        int8_split=int8_split,
        int8_fraction=int8_fraction,
        strip_intermediate=strip_intermediate,
        verify_load=verify_load,
        skip_if_up_to_date=skip_if_up_to_date,
        overwrite=overwrite,
        pt_globs=pt_globs,
        intermediate_globs=intermediate_globs,
        include_inference_pt=include_inference_pt,
        inference_only=inference_only,
    )
    print(f"完成，共 {len(outputs)} 个 engine。")
    for p in outputs:
        print(f"  {p}")
    return outputs


if __name__ == "__main__":
    # ------------------------------------------------------------------ #
    #  按需修改：以下为本脚本全部入口配置（集中维护）
    # /home/shunyao/miniconda310/envs/yolo11/bin/python3 acc_tensorRT.py
    # 
    # ------------------------------------------------------------------ #

    # 3060解决依赖冲突问题
    """
    /home/shunyao/miniconda310/envs/yolo11/bin/python3 -m pip install \
  torch==2.7.0+cu128 torchvision==0.22.0+cu128 \
  --find-links https://mirrors.aliyun.com/pytorch-wheels/cu128/ \
  --force-reinstall
    """
    # 输入：单个 .pt 文件，或含多个 .pt 的目录
    # INPUT_PATH = "/data/models/seg-3.12.1.pt"
    INPUT_PATH = "/data/models/daofeishi-detect-0415.pt"
    # INPUT_PATH = "/data/models/daofeishi-cls-3.0.3.pt"
    INPUT_PATH = "/data/models/cls-3.8.1.pt"

    # 多尺寸时输出 seg-3.12.1-i1280.engine；仅一个尺寸且 False 时写 seg-3.12.1.engine
    IMGSZ_SUFFIX = False

    # TensorRT 导出参数
    DEVICE: int | str | None = 0  # CUDA 设备，如 0；None 表示有 GPU 时用 0
    HALF = True
    INT8 = False
    # 导出输入尺寸；与 JSON 中 seg_imgsz / clip_profiles[].seg_imgsz 一致，可写多个
    # 例：detect_big 多尺度 -> [640, 1024, 1280]
    IMGSZ_LIST = [512]
    # 分类 crop 批量推理：DYNAMIC=True + BATCH=期望最大批量（写入 engine metadata）
    # 3060 12GB + classify 1280：建议 BATCH=8~16；导出示踪自动用 batch=1 省显存
    DYNAMIC = True
    BATCH = 16
    WORKSPACE_GB = 4
    SIMPLIFY = True

    # TensorRT 11 导出 FP16/INT8 需先安装（首次导出前执行一次）：
    # pip install "nvidia-modelopt[onnx]>=0.44"

    # INT8 校准（仅 INT8=True 时生效；HALF 会自动关闭）
    # yaml 示例：path/train/val 指向图片目录，建议 val 至少 300 张与线上一致的虫情图
    INT8_DATA = ""  # 如 /data/datasets/seg_calib.yaml
    INT8_SPLIT = "val"  # val / train
    INT8_FRACTION = 1.0  # 校准集抽样比例 0~1

    # 中间 ckpt（*-t.pt / last.pt / epoch*.pt）导出前先 strip 为临时推理权重
    STRIP_INTERMEDIATE = True
    # 写出后用 YOLO() 试加载 engine
    VERIFY_LOAD = True
    # 已存在且比源 .pt 新则跳过
    SKIP_IF_UP_TO_DATE = True
    OVERWRITE = False

    # 目录批量扫描
    INFERENCE_ONLY = True  # True：跳过 *-t.pt / last.pt / epoch*.pt
    INCLUDE_INFERENCE_PT = True
    PT_GLOBS = ("*.pt",)
    INTERMEDIATE_GLOBS = ("*-t.pt", "last.pt", "epoch*.pt")

    # ------------------------------------------------------------------ #

    main(
        input_path=INPUT_PATH,
        imgsz_list=IMGSZ_LIST,
        imgsz_suffix=IMGSZ_SUFFIX,
        device=DEVICE,
        half=HALF,
        int8=INT8,
        dynamic=DYNAMIC,
        batch=BATCH,
        workspace_gb=WORKSPACE_GB,
        simplify=SIMPLIFY,
        int8_data=INT8_DATA or None,
        int8_split=INT8_SPLIT,
        int8_fraction=INT8_FRACTION,
        strip_intermediate=STRIP_INTERMEDIATE,
        verify_load=VERIFY_LOAD,
        skip_if_up_to_date=SKIP_IF_UP_TO_DATE,
        overwrite=OVERWRITE,
        inference_only=INFERENCE_ONLY,
        include_inference_pt=INCLUDE_INFERENCE_PT,
        pt_globs=PT_GLOBS,
        intermediate_globs=INTERMEDIATE_GLOBS,
    )
