# script

虫情识别推理脚本目录：**配置驱动的统一多根推理**（`predict_all.py`）、**Gradio 本地测试与 REST 服务**（`predict_all_gradio.py`），以及标注校验、数据 ingest 等辅助脚本。

一套 JSON 配置即可驱动「检测 + 分割 + 多级分类 + 多场景部署 + 在线服务」，覆盖田间设备、摆拍评测与算法迭代全链路。

## 亮点功能

### 统一推理架构

- **多根模型并行**：大虫检测、小虫专项、实例分割等多路模型在同一管线内独立跑图，结果统一格式、带来源 `source` 可追溯。
- **递归路由子图**：`out` → `models.cls` → 嵌套 `out` 形成可配置的分类决策树，新增物种/专项模型**改 JSON 即可**，无需改代码。
- **检测 × 分割协同**：detect 与 segment 根可并发执行；分割输出 polygon + bbox，支持 `poly_merge` 轮廓去重、`mask_rate` 填充率过滤。
- **场景一键切换**：`run_model` 在摆拍 / 生产 / 其他三套 profile 间切换，差异项覆盖合并，同一套代码适配评测与上线。

### 工程化与性能

- **TensorRT 加速**：`trt_switch` 一键切换 `.engine` 推理，YOLO 检测/分割/分类全链路可 TRT 化。
- **多级批推理**：滑窗 detect/seg 批处理、分类 crop 批推理、GPU 裁切流水线（`use_gpu_crop`），充分压榨 GPU 吞吐。
- **多进程推理池**：`run_count>1` 时独立 worker 进程各持一套模型，HTTP 并发场景下线性扩展吞吐。
- **分阶段 profiling**：内置 detect / seg / cls-batch / post 耗时采集，定位瓶颈无需外接 profiler。

### 现场适配能力

- **智能 ROI 预处理**：粘虫板圆盘自动定位与裁切（`roi_switch` + 插件），减少背景干扰。
- **多尺度滑窗**：`clip_profiles` 多套切片策略按图幅/场景切换，兼顾大图检出与小目标召回。
- **in_big 小虫恢复**：大虫框内小虫误滤时可按检测/分类置信度二次恢复，降低「大虫挡小虫」漏报。
- **尺寸 / 形态过滤**：对角线 `dia`、Otsu 暗区占比 `bin_dark_ratio`、边缘拒识等门限均可 JSON 配置，现场调参不改代码。

### 服务化交付

- **Gradio + REST 一体**：同一进程同时提供可视化测试页与 `POST /insect_3_predict` 生产接口，兼容历史 Flask 响应格式。
- **热切换运行模式**：Web 端切换摆拍/生产/其他后自动重启并重新加载模型，就绪探测 `/health/ready` 保障流量安全。
- **页面级调参**：各根模型开关与关键门限可在 Gradio 临时调整，**仅影响本次运行**，便于现场 A/B 对比而不污染配置仓库。
- **启动预加载**：服务启动即 warmup 默认管线，首请求无冷启动延迟。

### 质量闭环

- **内置标注校验**：批量跑图时对照 Pascal VOC xml 或文件名伪 GT，输出 TP/FP/FN/类型错统计与混淆矩阵 CSV。
- **过滤透明化**：`collect_filtered=True` 返回被滤实例及 `filter_reason`（门限 / 分类 other / dia / mask_rate 等），问题定位可解释。
- **增量续跑**：预测 xml 已存在则跳过，支持大规模评测断点续算。
- **Label Studio 回流**：分类框图一键上报 LS 或本地分目录导出，加速难例收集与迭代标注。

---

## 许可与使用

本目录代码为**商用专有代码**，公开仅供**技术参考**，**不构成使用授权**；未经权利人书面许可，**禁止复制、整合或用于商业用途**。若需商业应用或授权合作，请**联系著作权人**。完整条款见同目录 [`LICENSE`](LICENSE)。

---

## 主入口

| 入口 | 用途 |
|:---|:---|
| **`predict_all.py`** | 统一推理核心：加载 `config/insect_alg_all.json`，递归 `out` / `models.cls` 路由；支持编程调用与 `__main__` 批量跑图、VOC 导出、标注校验 |
| **`predict_all_gradio.py`** | 基于 `predict_all` 的 **Gradio 图片测试页** + **FastAPI REST**（`/insect_3_predict`）；可切换运行模式、临时调参、多进程并发 |

算法 JSON 字段、`out` 路由、门限与裁切参数见 **[算法配置说明.md](./算法配置说明.md)**。物种元数据与类别合并见 `config/`（`config/AGENTS.md`）。

---

## `predict_all.py`

### 编程接口

```python
from script.predict_all import create_pipeline, predict, draw_results, load_insect_alg_all

# 方式一：便捷函数（每次新建管线并在 finally 中 release）
results = predict(image_bgr, config_path="config/insect_alg_all.json", device="cuda:0")

# 方式二：复用管线（服务/批处理推荐）
pipe = create_pipeline("config/insect_alg_all.json", device="cuda:0")
try:
    results = pipe.predict(image_bgr)
    # collect_filtered=True 时返回 (results, filtered)
finally:
    pipe.release()

# 可视化（PIL 中文字体；支持 polygon、校验框、filtered 灰框等）
annotated = draw_results(image_bgr, results, output_path="out.jpg", draw_polygon=True)
```

| 符号 | 说明 |
|:---|:---|
| `load_insect_alg_all(path)` | 加载并合并配置（`run_model` 场景 profile、按 OS 解析 `model_dir`） |
| `InsectPredictAll` | 多根推理管线类；构造时可传 `root_ids` 限定根模型、`enable_mask_rate_filter` |
| `create_pipeline(...)` | `InsectPredictAll` 工厂 |
| `predict(...)` | 单次推理便捷函数；传入已构建的 `pipeline=` 可复用权重 |
| `draw_results(...)` | 将结果绘制到 BGR 图；校验模式可叠加 TP/FP/FN 配色 |

**配置路径**：`config_path` 为相对路径时，一律相对于 **`insect/script/`**（见 `config_paths.resolve_insect_alg_all_path`），与进程 `cwd` 无关。

**运行模式**：启动文件 `config/insect_alg_all.json` 顶层 `run_model` 取值 `baipai`（摆拍）/ `shengchan`（生产）/ `other`（其他），分别合并 `insect_alg_shengchan.json`、`insect_alg_other.json` 中的场景差异项。

### 推理行为摘要

- 每个启用的 `models.<id>` 为独立**根模型**，结果带 `source=<id>`。
- `model_type`：`detect`（`PredictSize`：切片检测 → 尺寸/路由 → 可选分类）或 `segment`（`PredictSeg`：实例分割 → `out` 路由）。
- `out` / 嵌套 `models.cls` 形成递归子图；后置 `postprocess` 做报出类映射、去重等。
- `predict_cfg.parallel_detect_seg=true` 时，detect 与 segment 根可并发执行。

### 统一输出格式

每条保留结果大致为：

```json
{
  "name": "baitiaoyee",
  "score": 0.87,
  "location": [x1, y1, x2, y2],
  "source": "detect_big",
  "cn_name": "白条夜蛾",
  "polygon": [[x, y], ...]
}
```

- **`source`**：根模型 ID。
- **`collect_filtered=True`** 时另返回被过滤项，含 `filtered`、`filter_reason`（如 `threshold` / `cls` / `dia` / `mask_rate` 等）。
- 完整字段与过滤语义见 [算法配置说明.md §8](./算法配置说明.md#8-统一输出格式predict_all)。

### 批量运行（`__main__`）

在 IDE 中直接运行 `predict_all.py`，于文件末尾 `if __name__ == "__main__":` 修改变量即可（**不使用命令行参数**）：

| 变量 | 含义 |
|:---|:---|
| `CONFIG_PATH` | 算法 JSON，默认 `config/insect_alg_all.json` |
| `INPUT_PATH` | 输入图片目录或单图路径 |
| `OUTPUT_DIR` | 输出目录（结果图、xml、校验统计等） |
| `ROOT_IDS` | `None` 跑全部启用根；或 `["detect_big", ...]` 限定根 |
| `SAVE_IMAGE` / `DRAW_*` | 可视化开关（bbox、polygon、标签、filtered 等） |
| `OUTPUT_XML` / `OUTPUT_LABELME` | 导出 Pascal VOC bbox xml / LabelMe polygon json |
| `ENABLE_VALIDATION` | 对照同目录 `.xml` 或文件名伪 GT 做 IoU/IoR 校验与按类汇总 |
| `SKIP_IF_OUTPUT_EXISTS` | 增量续跑（须 `OUTPUT_XML=True` 且 `CLEAN_OUTPUT_BEFORE_RUN=False`） |
| `POSTPROCESS_PIPELINE` | 推理（GPU）与校验/画图/导出（CPU）流水线并行 |
| `ENABLE_LS_INGEST` | 将分类框图上报 LS 或落盘本地目录 |

示例（Linux 后台）：

```bash
nohup /path/to/python insect/script/predict_all.py > predict.log 2>&1 &
```

---

## `predict_all_gradio.py`

Gradio 本地测试服务：上传图片、切换 **运行模式**、临时调整各根模型开关与关键参数，复用 `predict_all` 推理与 `draw_results` 绘图。

### 启动

在 `__main__` 中修改 `CONFIG_PATH`、`SERVER_NAME`、`SERVER_PORT`、`INFER_DEVICE` 等变量后运行：

```bash
# 推荐：stdout+stderr 重定向到同一日志文件
nohup /path/to/python insect/script/predict_all_gradio.py > web.log 2>&1 &
```

默认监听 **`0.0.0.0:37860`**，同时提供 Gradio 页面与 REST 路由。

依赖：`gradio`、`uvicorn`、`fastapi`（见 `requirments.txt`）。Gradio **≥ 5.29** 可避免拖拽替换图片时新开浏览器标签；建议 `gradio>=5.31,<6`。

### Gradio 页面

- **运行模式** `run_model`：摆拍 / 生产 / 其他；切换后写回 `insect_alg_all.json` 并**自动重启**后台进程重新加载模型。
- **重启服务**：不改运行模式，仅重新加载权重。
- **推理设备**：自动 / cpu / cuda:0 / mps（仅影响本次会话默认设备）。
- **算法开关与参数**：按 `models` 下各根模型动态生成控件；修改仅作用于**本次运行**，不写回 JSON（运行模式除外）。
- 就绪检测：页面加载时轮询 `GET /health/ready`，模型未就绪时显示遮罩。

### REST API

由 `create_serving_app()` 挂载到 FastAPI（`gr.mount_gradio_app`，避免 `demo.launch()` 覆盖自定义路由）。

| 方法 | 路径 | 说明 |
|:---|:---|:---|
| `GET` | `/health/ready` | 就绪探测；未就绪返回 503 |
| `POST` | `/insect_3_predict?input_type=url` | 虫情识别；body `{"url": "<http(s)://... 或本地路径>"}` |

**成功响应**（HTTP 200）：

```json
{
  "results": [
    {
      "name": "baitiaoyee",
      "score": 0.87,
      "location": { "left": 100, "top": 50, "width": 80, "height": 60 }
    }
  ]
}
```

**错误响应**：HTTP 200 + `{"code": 500, "msg": "..."}`（与历史 Flask 接口一致）。

### 编程挂载

```python
from script.predict_all_gradio import create_app, create_serving_app

# 仅 Gradio Blocks（测试或外部挂载）
demo = create_app(config_path=None, infer_device="cuda:0")

# Gradio + REST（生产推荐）
app = create_serving_app(config_path=None, cache_size=3, infer_device="cuda:0")
# uvicorn.run(app, host="0.0.0.0", port=37860)
```

### 并发与性能

由 `predict_cfg` 控制（详见 [算法配置说明.md](./算法配置说明.md)）：

| 键 | 说明 |
|:---|:---|
| `run_count` | `>1` 时启动多**推理子进程**（`predict_worker_pool.InferenceProcessPool`），各进程独立加载模型 |
| `api_concurrency` | 进程内 HTTP/Gradio 并发线程数；`run_count=1` 时 GPU 推理按模型权重加锁 |
| `use_gpu_crop` / `cls_batch_size` / `detect_seg_batch_size` | 裁切与批推理优化 |
| `trt_switch` | TensorRT `.engine` 开关 |

`run_count>1` 仅在**并发请求**时有吞吐收益；单 GPU 上多进程通常非线性加速，压测无并发时建议 `run_count=1`。

---

## 目录说明

| 文件 / 目录 | 说明 |
|:---|:---|
| **`predict_all.py`** | 统一推理主入口（detect/segment 多根、递归 cls、`draw_results`、批量校验） |
| **`predict_all_gradio.py`** | Gradio 测试服务 + `/insect_3_predict` REST |
| **`predict_all_xingneng.py`** | 性能压测脚本（基于 `predict_all`） |
| `config/` | 静态配置与加载模块：JSON、`cls_merge.py`、`insect_info.py` 等（见 `config/AGENTS.md`） |
| `config_paths.py` | 配置路径、`run_model` profile 合并、`resolve_effective_insect_alg_path()` |
| `predict/` | 底层模型：`model_detect.py`（检测、切片、IoR/合并）、`model_cls*.py`（分类/批推理/GPU 裁切）、`model_seg.py`、`model_trt.py` 等 |
| `predict_size.py` | **`PredictSize`**：切片检测 → `size.json` 尺寸过滤 → 可选分类 |
| `predict_seg.py` / `predict_seg_lib.py` | **`PredictSeg`** 与分割路由、多边形后处理、VOC/校验工具函数 |
| `predict_size_validate_lib.py` | 标注校验、混淆矩阵、评估绘图共用库 |
| `predict_worker_pool.py` | `run_count>1` 时的多进程推理池 |
| `predict_cls_validate_from_xml.py` | 基于 XML 的分类校验 |
| `predict_mark.py` | 标注/标记辅助 |
| `ls_classification_ingest.py` / `ls_seg_classification_ingest.py` | 分类框图上报 Label Studio 或本地导出 |
| `test_api.py` / `test_api_yin.py` | REST 接口测试 |
| `tools/` | ROI 预处理、配置工具等 |
| `算法配置说明.md` | **`insect_alg_all.json` 字段说明**（运维必读） |

---

## 运行约定

- 各脚本作为包导入时会将 **`insect` 项目根**（`script` 的上一级）插入 `sys.path`。
- 请从项目根运行，或设置 `PYTHONPATH` 包含 `insect/` 目录。
- 修改 JSON 配置或切换 `run_model` 后，需 **`release()` 并重建管线**或**重启 Gradio/uvicorn 进程**，进程内会缓存模型权重。
- 模型路径：配置中 `model_dir` 按操作系统解析；单模型 `model` 字段建议使用部署机上的绝对路径。
