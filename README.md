# script

虫情识别推理脚本目录：离线图片推理、多模型融合，以及标注验证与可视化辅助。

## 许可与使用

本目录代码为**商用专有代码**，公开仅供**技术参考**，**不构成使用授权**；未经权利人书面许可，**禁止复制、整合或用于商业用途**。若需商业应用或授权合作，请**联系著作权人**。完整条款见同目录 [`LICENSE`](LICENSE)。

## 目录说明

| 文件 / 目录 | 说明 |
|-------------|------|
| `predict_merge.py` | **主融合入口**：`orig`（大虫）+ `daofeishi`（稻飞虱）+ `yumiming`（玉米螟，可选）+ `cls_12`（12 类专项）；统一输出格式，内置 IoR 去重与跨源合并。 |
| `predict_orig.py` | 大虫单路推理：检测 `kuangxuan_0209.pt` + 多级分类与规则修正；`predict(..., device=...)`。 |
| `predict_size_daofeishi.py` | 通用 **`PredictSize`**：切片检测 → `size.json` 尺寸过滤 → 可选分类；供稻飞虱、玉米螟、`cls_12` 等复用。 |
| `predict_size_yumiming.py` | 玉米螟批处理示例：基于 `PredictSize`，含 **`_write_pascal_voc_xml`** 将结果写成 Pascal VOC XML。 |
| `predict_size_yumiming_validate.py` | 在玉米螟流程上与 **Pascal VOC 标注** 比对验证：几何用 **IoR**（与 `model_detect.ior` 一致），类别经配置归并后比较。 |
| `merge_beyond.py` | **另一套融合**：稻飞虱 + 大检测框上的 **beyond** 二级推理（`beyond_insect` 集合内走 `beyond_predict`）+ 其余走 **cls_12**；与大虫框/稻飞虱框做 IoR 合并。`demo.py` 调用此入口。 |
| `demo.py` | 批量读图 → `merge_beyond.predict` → 按 `source` 着色绘制保存。 |
| `draw_inference_clip_grid.py` | **不加载模型**：按与 `ModelDetector.predict` 相同的切片逻辑在图上画分片网格（调 `get_clip`）。 |
| `predict/` | `model_detect.py`（检测、切片、IoR/合并等）、`model_cls.py`（分类）。 |
| `size.json` | 各类别宽高像素范围，供 `PredictSize` 尺寸过滤。 |

## `predict_merge.predict`（推荐主入口）

```python
from script.predict_merge import predict, draw_results

results = predict(image_bgr, device=None, parts=None)
```

- **`image`**：`numpy` BGR 图。
- **`device`**：`"cuda"` / `"mps"` / `"cpu"`；`None` 时自动选择。
- **`parts`**：启用的子模型列表。`None` 时默认为 **`["orig", "daofeishi", "cls_12"]`（默认不包含 `yumiming`）**。可显式传入例如 `["orig", "daofeishi", "yumiming", "cls_12"]`。

模型根目录为项目下 **`models/20260123`**（稻飞虱/玉米螟检测分类权重在该目录；缺失时在代码中可回退到脚本旁同名 `*.pt`，见各 `_get_predictor_*`）。

### 融合后处理（实现摘要）

1. **稻飞虱与 orig/yumiming**：对 `baibeifeishi`、`huifeishi`、`hefeishi`，若与任一 `orig` 或 `yumiming` 框 **IoR > 0.8**，则丢弃该稻飞虱框。
2. **orig 与 yumiming 同标 `yumiming`**：仅当结果中存在 `yumiming` 来源时执行；若 orig 框与某 `yumiming` 框 **IoR > 0.8**，丢弃 orig 那条。
3. **`cls_12` 与 orig/yumiming**：对 `source in ("cls_12", "orig", "yumiming")` 的框按分数降序，若两框 **IoR > 0.8** 则去掉分数较低者（减轻大虫多模型重复检出）。

`daofeishi` 路径上 **`name == "other"`** 的结果会在写入列表前过滤；`cls_12` 上对 **`other` 且分类置信度 < 0.05** 的条目也会跳过。

### 统一输出格式

每条为字典：

- `name`：类别名  
- `score`：置信度  
- `location`：`[x1, y1, x2, y2]`  
- `msg`：规则信息（可为空字符串）  
- `source`：`"orig"` / `"daofeishi"` / `"yumiming"` / `"cls_12"`

同文件提供 **`draw_results(image, results, output_path=None)`** 用于可视化（按来源配色）。

## `merge_beyond.predict`（扩展品类融合）

```python
from script.merge_beyond import predict

results = predict(image_bgr, merage=False)
```

- **`merage`**：为 `True` 时检测框一律走 `beyond_predict` 分支；为 `False` 时先经二级模型，仅在 `beyond_insect` 命中时再走 beyond 逻辑，否则走 **cls_12** 分类（结果里 `source` 多为 **`"cls-12"`** 或 **`"beyond"`**）。
- 稻飞虱使用 **`PredictSize`**，`conf_thresh=0.65`，切片 `clip_size=640`、`overlap_size=120`，并带 **`edge_reject_distance=5`** 等边缘过滤参数（见 `predict_daofeishi`）。
- 与大虫路结果通过 **`_merge_big_with_small`**：大框与小框（稻飞虱）IoR > 0.8 时去掉小框重叠者。

大检测模型权重来自 **`models/20260123`**（或 `model/20260123` 回退）；部分权重另从 **`MODEL_ROOT`（`项目/model`）** 加载，与 `predict_merge` 路径约定不完全相同，部署时需对照脚本内路径。

## `PredictSize`（`predict_size_daofeishi.py`）

构造参数包括：`detect_model_path`、`size_config_path`（可为 `None` 关闭尺寸过滤）、`cls_list`、`cls_model_path`（`None` 则只做检测+尺寸过滤）、`offset_rate`、`conf_thresh` / `conf_merge` / `iou_threshold` / `ior_threshold`、`device`、`cls_pad_square`、`cls_gray_binarize` 等。

**`predict(...)`** 支持：`clip_size`、`overlap_size`、`edge_reject_distance`、`edge_reject_conf_threshold`、`cls_top1_conf_threshold`、`detect_pad_square`、`return_full_final` 等（详见方法 docstring）。返回列表元素含 `x1,y1,x2,y2,conf,cls_name,cls_conf,...`。

## 其它说明

- **`predict_orig`** 中模型目录优先 **`models/20260123`**，不存在时回退 **`model/20260123`**。
- 各脚本在作为包导入时会将 **`insect` 项目根** 插入 `sys.path`（`Path(__file__).parents[1]`），请从项目根以包方式运行或设置 `PYTHONPATH`。
- `predict_merge.py` 与 `merge_beyond.py` 的 `__main__` 块多为本地批量跑图与保存路径示例，使用前请改为自己的数据目录。
