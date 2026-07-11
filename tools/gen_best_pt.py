#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将 YOLO 训练中间 checkpoint（含 optimizer / EMA 等大对象）瘦身为推理用 best.pt。

Ultralytics 在 save_model 里把整颗 EMA 网络、optimizer 等写入 ckpt，体积常为 best.pt 的
2～3 倍。本脚本调用官方 strip_optimizer：去 optimizer、用 EMA 替换 model、FP16 固化，产出可
直接 YOLO(path) 加载的权重。

支持输入：
- 单个 .pt 文件（如 .../3.8.1-t.pt）
- 目录：扫描 *-t.pt / last.pt / epoch*.pt 等中间权重

用法：改 __main__ 中变量后在 IDE 直接运行（本仓库 demo 约定不用 argparse）。
"""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils.torch_utils import strip_optimizer

# 训练侧曾向 ultralytics.utils.loss 注入的自定义 loss；checkpoint 反序列化需要占位类
_CUSTOM_LOSS_STUBS = ("BCEDiceLoss", "MultiChannelDiceLoss")

# --- 按需修改 ---
# 文件或目录；目录时会扫描 INTERMEDIATE_GLOBS 下的中间权重
INPUT_PATH = "/Volumes/shunyao-h1/models-test/cls-v3.8/3.8.3/temp.pt"
# 单文件时可指定输出路径；留空则与源同目录写 OUTPUT_NAME
OUTPUT_PATH = ""
# 目录批量时默认输出文件名；多文件同目录时会改为 {版本}.pt 避免覆盖
OUTPUT_NAME = "cls-3.8.3.pt"
# 中间权重 glob（相对 INPUT_PATH 为目录时）
INTERMEDIATE_GLOBS = ("*-t.pt", "last.pt", "epoch*.pt")
# 写出前用 YOLO() 试加载
VERIFY_LOAD = True
# 已存在且比源文件新则跳过
SKIP_IF_UP_TO_DATE = True
# 强制覆盖已有输出
OVERWRITE = False


def _is_macos_resource_fork(path: Path) -> bool:
    return path.name.startswith("._")


def _is_intermediate_name(name: str) -> bool:
    lower = name.lower()
    if lower == "best.pt":
        return False
    if lower.endswith("-best.pt") or lower.endswith("_strip.pt") or lower.endswith("_fp16.pt"):
        return False
    return True


def _register_custom_ultralytics_loss_stubs() -> None:
    """为含自定义 criterion 的中间 ckpt 注册占位类，便于 torch.load / strip_optimizer。"""
    import ultralytics.utils.loss as ul_loss

    class _PickleStub:
        def __init__(self, *args, **kwargs) -> None:
            pass

    for name in _CUSTOM_LOSS_STUBS:
        if not hasattr(ul_loss, name):
            setattr(ul_loss, name, type(name, (_PickleStub,), {}))


def collect_intermediate_weights(root: Path) -> list[Path]:
    if not root.is_dir():
        raise NotADirectoryError(root)

    found: dict[str, Path] = {}
    for pattern in INTERMEDIATE_GLOBS:
        for path in sorted(root.glob(pattern)):
            if not path.is_file() or _is_macos_resource_fork(path):
                continue
            if not _is_intermediate_name(path.name):
                continue
            found[str(path.resolve())] = path
    return sorted(found.values(), key=lambda p: p.name)


def resolve_output_path(
    src: Path,
    *,
    output_path: str | Path | None,
    output_name: str,
    multi_in_dir: bool,
) -> Path:
    if output_path:
        return Path(output_path).expanduser().resolve()

    parent = src.parent
    name = (output_name or "best.pt").strip() or "best.pt"

    if multi_in_dir and src.name.endswith("-t.pt"):
        # 3.8.1-t.pt -> 3.8.1.pt，避免同目录多个 -t 覆盖同一个 best.pt
        return parent / f"{src.stem[:-2]}.pt"

    return parent / name


def strip_to_best(src: Path, dst: Path, *, verify_load: bool) -> tuple[Path, float, float]:
    src = src.expanduser().resolve()
    dst = dst.expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(src)

    dst.parent.mkdir(parents=True, exist_ok=True)
    src_mb = src.stat().st_size / 1e6
    _register_custom_ultralytics_loss_stubs()
    ckpt = strip_optimizer(str(src), str(dst))
    if not ckpt:
        raise RuntimeError(
            f"strip_optimizer 失败，未写出 {dst}。"
            f" 常见原因：checkpoint 含当前 ultralytics 不存在的自定义类；"
            f" 若为新类，请补充到 _CUSTOM_LOSS_STUBS。"
        )
    if not dst.is_file():
        raise FileNotFoundError(f"strip_optimizer 未生成输出文件: {dst}")
    dst_mb = dst.stat().st_size / 1e6

    if verify_load:
        YOLO(str(dst))

    return dst, src_mb, dst_mb


def should_skip(src: Path, dst: Path, *, skip_if_up_to_date: bool, overwrite: bool) -> bool:
    if overwrite:
        return False
    if not dst.is_file():
        return False
    if not skip_if_up_to_date:
        return False
    return dst.stat().st_mtime >= src.stat().st_mtime


def gen_best_pt(
    input_path: str | Path,
    *,
    output_path: str | Path | None = None,
    output_name: str = "best.pt",
    intermediate_globs: tuple[str, ...] = INTERMEDIATE_GLOBS,
    verify_load: bool = True,
    skip_if_up_to_date: bool = True,
    overwrite: bool = False,
) -> list[Path]:
    root = Path(input_path).expanduser().resolve()
    if root.is_file():
        sources = [root]
    elif root.is_dir():
        sources = collect_intermediate_weights(root)
        if not sources:
            patterns = ", ".join(intermediate_globs)
            raise FileNotFoundError(f"目录内未找到中间权重（{patterns}）: {root}")
    else:
        raise FileNotFoundError(root)

    multi = len(sources) > 1
    written: list[Path] = []
    for src in sources:
        dst = resolve_output_path(
            src,
            output_path=output_path if len(sources) == 1 else None,
            output_name=output_name,
            multi_in_dir=multi,
        )
        if should_skip(src, dst, skip_if_up_to_date=skip_if_up_to_date, overwrite=overwrite):
            print(f"跳过（已存在且较新）: {dst}")
            written.append(dst)
            continue

        out, src_mb, dst_mb = strip_to_best(src, dst, verify_load=verify_load)
        ratio = (1.0 - dst_mb / src_mb) * 100.0 if src_mb else 0.0
        print(f"已写出: {out}")
        print(f"  源: {src} ({src_mb:.1f} MB)")
        print(f"  目标: {out} ({dst_mb:.1f} MB, 瘦身约 {ratio:.0f}%)")
        written.append(out)
    return written


def main() -> None:
    outputs = gen_best_pt(
        INPUT_PATH,
        output_path=OUTPUT_PATH or None,
        output_name=OUTPUT_NAME,
        intermediate_globs=INTERMEDIATE_GLOBS,
        verify_load=VERIFY_LOAD,
        skip_if_up_to_date=SKIP_IF_UP_TO_DATE,
        overwrite=OVERWRITE,
    )
    print(f"完成，共 {len(outputs)} 个文件。")


if __name__ == "__main__":
    main()
