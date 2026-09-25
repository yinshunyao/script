# script/tools

推理与评估辅助脚本。改 `__main__` 里的路径变量后直接运行（不走命令行参数）。生产推理会 import 的只有 `roi_preprocess.py`。

| 文件 | 用途 |
|:---|:---|
| `roi_preprocess.py` | 推理前 ROI 插件：识别黑框圆盘，圆外填白。`predict_all` / `model_seg` 调用。 |
| `export_yolo_best_pt.py` | 训练 ckpt（含 optimizer/EMA）瘦身为可 `YOLO(path)` 加载的推理权重；支持单文件或目录批量。 |
| `export_yolo_best_pt_simple.py` | 单文件瘦身试跑；可选只写 `state_dict`（不一定能被 `YOLO(path)` 直接加载）。 |
| `export_yolo_onnx.py` | YOLO 检测/分类/分割 `.pt` 导出 ONNX；改底部 `INPUT_PATH` 后运行。 |
| `export_tensorrt_engine.py` | YOLO `.pt` 导出 TensorRT `.engine`（须 NVIDIA GPU）。 |
| `analyze_pred_xml_conf.py` | GT XML 与预测 XML 比对，按类统计 det/cls 置信度分布。 |
| `analyze_cls_topn_scenes.py` | 已检出 VOC（含 `cls_topn`）按正确 / 近种 / 其他做 top1–top3 场景分析。 |
| `optimize_size_conf_thresholds.py` | 用已有预测 XML 离线网格搜索大小虫 det/cls 门限，最大化生产计数总分。 |
| `analyze_pred_xml_diag.py` | 同上，统计框对角线长度分布。 |
| `draw_inference_clip_grid.py` | 按推理切片逻辑画分片网格，不加载模型。 |
| `preview_gray_contrast_enhance.py` | 预览 detect/seg 前 gray CLAHE，与 `model_channel` 一致。 |
| `add_missing_insect_info_entries.py` | 为 `config/insect_info.json` 补缺物种条目。 |
| `update_insect_info_imaging_extent_mm.py` | 按体长规则更新 `insect_info.json` 顶层成像跨度 mm。 |

XML 分析默认仍写出 `xml_fenxi_*.csv`（与已有结果目录一致）。
