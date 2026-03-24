# script

虫情识别推理脚本目录，主要用于离线图片推理与多模型结果融合。

## 目录说明

- `predict_merge.py`：融合推理总入口，组合 `orig`、`daofeishi`、`yumiming`、`cls_12` 多路结果并做去重合并。
- `predict_orig.py`：大虫模型推理入口，包含检测 + 分类及部分规则修正逻辑。
- `predict_size_daofeishi.py`：基于尺寸过滤的检测/分类推理器（`PredictSize`），用于稻飞虱等按尺寸约束的流程。
- `predict_size_yumiming.py`：玉米螟推理示例脚本，复用 `PredictSize`。
- `predict/`：底层模型封装（检测、分类、框合并等公共逻辑）。
- `size.json`：尺寸过滤配置。

## 推理输出（统一格式）

融合入口 `predict_merge.predict(...)` 输出列表，每条结果格式如下：

- `name`：类别名
- `score`：置信度
- `location`：框坐标 `[x1, y1, x2, y2]`
- `msg`：规则修正信息（可空）
- `source`：结果来源（如 `orig` / `daofeishi` / `yumiming` / `cls_12`）

## 说明

- 设备支持 `cuda` / `mps` / `cpu`，不指定时自动选择。
- 模型文件默认从项目下 `models/20260123` 读取（部分脚本包含回退路径逻辑）。
- 文件中的 `__main__` 示例主要用于本地批量验证与结果图导出。
