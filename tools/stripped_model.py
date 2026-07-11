"""
将训练过程中的 *.pt（含 optimizer / ema 等大对象）压缩为接近 best.pt 的推理用权重。

说明：
- Ultralytics 在 save_model 里把「整颗 EMA 网络」放在 ckpt['ema']，且常为 FP16；
  与「手写 YOLO('yolo11n.yaml') 再 load_state_dict」相比，**架构必须与训练时一致**，
  否则会报 channel / shape mismatch（例如 n 与 s 首层 16 vs 32）。
- **不必**借助预训练模型做「优化」；用官方 strip_optimizer 即可按 best 逻辑去掉 optimizer、
  用 EMA 替换 model、再 FP16 固化，显著减小体积且仍可直接 YOLO(path) 加载。

用法：改下面三个变量后直接在 IDE 里运行即可（本仓库约定 demo 不用 argparse）。
"""

from pathlib import Path

import torch
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.utils.torch_utils import strip_optimizer

# --- 按需修改 ---
ckpt_path = '/Users/shunyaoyin/Documents/code/ai-company/insect/doc/测试结果/大虫框选/20260426-large-02-insect/epoch129.pt'
output_path = '/Users/shunyaoyin/Documents/code/ai-company/insect/doc/测试结果/大虫框选/20260426-large-02-insect/best.pt'
# 与 best.pt 一致：True=官方 strip（推荐）；False=仅导出 state_dict 等小体积（需自管架构）
USE_STRIP_OPTIMIZER = True


def main() -> None:
    src = Path(ckpt_path)
    if not src.is_file():
        raise FileNotFoundError(src)

    if USE_STRIP_OPTIMIZER:
        strip_optimizer(str(src), str(output_path))
        mb = Path(output_path).stat().st_size / 1e6
        print(f'已按 best.pt 逻辑写出: {output_path} ({mb:.1f} MB)')
        return

    # --- 备选：从 ckpt 直接实例化「与训练一致」的模型，再只存 state_dict（体积通常更小，
    # 但 ultralytics 的 YOLO('x.pt') 默认期望 ckpt["model"] 为 Module；若要用此文件做 YOLO 入口，
    # 更稳妥仍用上面 strip_optimizer。）---
    model, ckpt = attempt_load_one_weight(str(src), device='cpu', fuse=False)
    names = getattr(model, 'names', ckpt.get('names'))
    nc = getattr(model, 'nc', ckpt.get('nc'))
    torch.save(
        {
            'model': model.state_dict(),
            'names': names,
            'nc': nc,
            'train_args': ckpt.get('train_args'),
        },
        output_path,
    )
    mb = Path(output_path).stat().st_size / 1e6
    print(f'已写出 state_dict 版: {output_path} ({mb:.1f} MB)')
    print('提示：该格式可能无法被 YOLO(path) 直接加载；推理请优先用 strip 输出。')


if __name__ == '__main__':
    main()
