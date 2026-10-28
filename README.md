# script

> English (default). Chinese: **[README-cn.md](./README-cn.md)**.

Pest-recognition inference scripts: **config-driven multi-root inference** (`predict_all.py`), **Gradio local testing + REST serving** (`predict_all_gradio.py`), plus helpers for label validation, data ingest, export, and evaluation.

One JSON config drives detection + segmentation + nested classification + multi-scene deploy + online serving—covering field devices, staged evaluation, and algorithm iteration. Field-level keys: **[算法配置说明.md](./算法配置说明.md)** (Chinese); on-site toggles: `config/算法配置重点.md`.

## Highlights

### Unified inference

- **Parallel multi-root models**: large-insect detect, small-insect specialists, instance seg, and more run in one pipeline; results share a common schema with traceable `source`.
- **Recursive routing subgraph**: `out` → `models.cls` → nested `out` forms a configurable classification tree; keys support class names, regex, diagonal size bands (e.g. `"[50,300)"`), and `*` wildcards—add species / specialist models by **editing JSON only**.
- **Detect × segment co-op**: detect and segment roots can run concurrently (`predict_cfg.parallel_detect_seg`); segment emits polygon + bbox, with `poly_merge` and `mask_rate` fill-ratio filters.
- **One-switch scenes**: `run_model` switches among staged / production / other profiles with overlay merge—same code for eval and production.

### Models & backends

- **Detect**: Ultralytics **YOLO** (`.pt`), **TensorRT** (`.engine`, `trt_switch`), **ONNX Runtime** (`.onnx`; `model_type: onnx`/`ort`, or auto when weight suffix is `.onnx`; same pipeline as detect).
- **Segment**: YOLO segment (`.pt` / `.engine`), sliding windows, cross-tile IoR merge, polygon remap.
- **Nested classify**: YOLO classify (`.pt` / `.engine`) and **timm ConvNeXt** (and similar) checkpoints auto-detected; YOLO cls supports crop **batching** (`cls_batch_size`).
- **Export tools**: `tools/export_yolo_onnx.py`, `export_tensorrt_engine.py`, `export_yolo_best_pt*.py`—train weights → deploy weights.

### Confidence & filter thresholds (all JSON-configurable)

- **Detect/seg conf**: root `detect_conf`; optional `detect_conf_small` / `detect_conf_big`; per-class `out.*.detect_conf`. At YOLO time, conf uses the **min** of root and per-class floors so low-threshold classes are not missed; after routing, each class’s own threshold applies.
- **Cls conf**: `out.*.cls_conf` on top1; optional `skip_dia_conf` (skip size gate when cls conf is high).
- **Size filter**: bbox diagonal `dia` (px) or `dia_mm` + `px_per_mm`; root and `out` levels; master `dia_switch`.
- **Relative-size re-rank** (`postprocess.relative_size`): estimate scale from top1/top2/top3 reported classes in-image; if top cls size is clearly wrong, re-pick within `cls_topk` by size (reduces “large pest reported as small”).
- **Geometry post** (`postprocess.geometry`): same-class IoU merge; big-in-big / small-in-small / small-in-big (IoR) suppression.
- **Morphology & ROI**: `bin_dark_ratio_min` (Otsu dark-area ratio), seg `mask_rate`, disk-edge `roi_edge`, multi-scale `split_by_scale`, etc.
- **Cls OOD / blacklist**: `black_classes` fallback & dump-bin, `p_dump` / `p_near` / `min_near` / `near_cls` near-species reject; `only_classes` whitelist; `report_tier` for report level.
- **Explainable filters**: `collect_filtered=True` returns dropped instances with `filter_reason` (e.g. `threshold` / `cls` / `dia` / `mask_rate` / `relative_size` / `big_in_big_ior`).

### Engineering & performance

- **TensorRT**: `predict_cfg.trt_switch` flips `.engine` for detect/seg/cls.
- **Multi-level batching**: sliding-window detect/seg batches, cls crop batches, GPU crop path (`use_gpu_crop`).
- **Multi-process pool**: `run_count>1` → each worker process holds its own model set for HTTP concurrency.
- **Phase profiling**: detect / seg / cls-batch / post timings (`predict_phase_profile`).

### Field adaptation

- **Smart ROI**: sticky-board disk locate & crop (`roi_switch` + plugin).
- **Multi-scale windows**: `clip_profiles` switch tile strategies by image size / scene; optional gray CLAHE.
- **in_big recover / reject**: restore small pests wrongly dropped inside large boxes by det/cls conf; rice planthopper-style cases can use `in_big_conf` + large-mask intersection ratio.
- **Report mapping**: leaf `infer_name` / `cn_name` / `report_tier` / `size_class`—display decoupled from body-size geometry rules.

### Serving

- **Gradio + REST in one process**: visual test UI + `POST /insect_3_predict`, Flask-compatible response shape.
- **Hot run-mode switch**: Web switch among staged/production/other restarts and reloads models; `/health/ready` probe.
- **Page-level knobs**: root on/off and key thresholds in Gradio apply to **this run only** (except run mode), without dirtying the config repo.
- **Startup warmup**: default pipeline warmed at boot to cut first-request cold start.

### Quality loop

- **Built-in label check**: batch vs Pascal VOC xml or filename pseudo-GT → TP/FP/FN/type-error + confusion CSV.
- **Incremental resume**: skip when prediction xml already exists.
- **Label Studio ingest**: cls crops to LS or local folders.
- **Offline threshold search**: `tools/optimize_size_conf_thresholds.py` etc. grid-search det/cls thresholds from existing prediction XML.

---

## License & use

This directory is **commercial proprietary** code. Public access is for **technical reference only** and **does not grant a license**. Without written permission from the rights holder, **copying, integrating, or commercial use is prohibited**. For commercial use or licensing, **contact the copyright holder**. Full terms: [`LICENSE`](LICENSE).

---

## Main entry points

| Entry | Purpose |
|:---|:---|
| **`predict_all.py`** | Unified inference: load `config/insect_alg_all.json`, recursive `out` / `models.cls` routing; library API + `__main__` batch, VOC export, label check |
| **`predict_all_gradio.py`** | **Gradio image UI** + **FastAPI REST** (`/insect_3_predict`) on top of `predict_all`; run-mode switch, temp knobs, multi-process concurrency |

JSON fields, `out` routing, thresholds, and crop params: **[算法配置说明.md](./算法配置说明.md)**. Species metadata & class merge: `config/` (`config/AGENTS.md`).

---

## `predict_all.py`

### API

```python
from script.predict_all import create_pipeline, predict, draw_results, load_insect_alg_all

# Option A: convenience (new pipeline each call; release in finally)
results = predict(image_bgr, config_path="config/insect_alg_all.json", device="cuda:0")

# Option B: reuse pipeline (preferred for serving / batch)
pipe = create_pipeline("config/insect_alg_all.json", device="cuda:0")
try:
    results = pipe.predict(image_bgr)
    # collect_filtered=True → (results, filtered)
finally:
    pipe.release()

# Viz (PIL CJK fonts; polygon, validation boxes, filtered gray boxes, …)
annotated = draw_results(image_bgr, results, output_path="out.jpg", draw_polygon=True)
```

| Symbol | Notes |
|:---|:---|
| `load_insect_alg_all(path)` | Load & merge config (`run_model` profiles; OS-aware `model_dir`) |
| `InsectPredictAll` | Multi-root pipeline; ctor may take `root_ids`, `enable_mask_rate_filter` |
| `create_pipeline(...)` | Factory for `InsectPredictAll` |
| `predict(...)` | One-shot helper; pass built `pipeline=` to reuse weights |
| `draw_results(...)` | Draw on BGR; validation mode can overlay TP/FP/FN colors |

**Config path**: relative `config_path` is always resolved against **`insect/script/`** (`config_paths.resolve_insect_alg_all_path`), independent of process `cwd`.

**Run mode**: top-level `run_model` in `config/insect_alg_all.json` is `baipai` (staged) / `shengchan` (production) / `other`, merging diffs from `insect_alg_shengchan.json` / `insect_alg_other.json`.

### Inference summary

- Each enabled `models.<id>` is an independent **root**; results carry `source=<id>`.
- `model_type`: `detect` / `yolo` (YOLO detect), `onnx` / `ort` (same pipeline, ONNX backend; `.onnx` weights auto), `segment` (instance seg → `out` routing).
- Detect roots use `PredictSize`: tiled detect → size/route → optional nested cls; segment roots use `PredictSeg`.
- `out` / nested `models.cls` form a recursive subgraph; `postprocess` does geometry merge, relative size, report tier, dedupe, etc.
- With `predict_cfg.parallel_detect_seg=true`, detect and segment roots may run concurrently.

### Unified output

Kept instances look like:

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

- **`source`**: root model id.
- **`collect_filtered=True`**: also returns filtered items with `filtered` / `filter_reason` (e.g. `threshold` / `cls` / `dia` / `mask_rate` / `relative_size`).
- Full fields & filter semantics: [算法配置说明.md §8](./算法配置说明.md#8-统一输出格式predict_all).

### Batch (`__main__`)

Run `predict_all.py` in the IDE; edit variables under `if __name__ == "__main__":` at the file bottom (**no CLI args**):

| Variable | Meaning |
|:---|:---|
| `CONFIG_PATH` | Algorithm JSON; default `config/insect_alg_all.json` |
| `INPUT_PATH` | Image dir or single image |
| `OUTPUT_DIR` | Output (viz, xml, validation stats, …) |
| `ROOT_IDS` | `None` = all enabled roots; or e.g. `["detect_big", ...]` |
| `SAVE_IMAGE` / `DRAW_*` | Viz switches (bbox, polygon, labels, filtered, …) |
| `OUTPUT_XML` / `OUTPUT_LABELME` | Export Pascal VOC bbox xml / LabelMe polygon json |
| `ENABLE_VALIDATION` | IoU/IoR check vs same-dir `.xml` or filename pseudo-GT; per-class summary |
| `SKIP_IF_OUTPUT_EXISTS` | Incremental resume (needs `OUTPUT_XML=True` and `CLEAN_OUTPUT_BEFORE_RUN=False`) |
| `POSTPROCESS_PIPELINE` | Pipeline parallelism: infer (GPU) vs validate/draw/export (CPU) |
| `ENABLE_LS_INGEST` | Push cls crops to LS or local folders |

Example (Linux background):

```bash
nohup /path/to/python insect/script/predict_all.py > predict.log 2>&1 &
```

---

## `predict_all_gradio.py`

Gradio local test service: upload images, switch **run mode**, temporarily tweak root switches and key params; reuses `predict_all` infer + `draw_results`.

### Start

Edit `CONFIG_PATH`, `SERVER_NAME`, `SERVER_PORT`, `INFER_DEVICE`, etc. in `__main__`, then:

```bash
# Prefer: stdout+stderr to one log file
nohup /path/to/python insect/script/predict_all_gradio.py > web.log 2>&1 &
```

Default listen **`0.0.0.0:37860`** with Gradio UI and REST routes.

Deps: `gradio`, `uvicorn`, `fastapi` (see `requirments.txt`). Gradio **≥ 5.29** avoids opening a new browser tab when drag-replacing images; recommend `gradio>=5.31,<6`.

### Gradio UI

- **Run mode** `run_model`: staged / production / other; writes `insect_alg_all.json` and **auto-restarts** the backend to reload models.
- **Restart service**: reload weights without changing run mode.
- **Device**: auto / cpu / cuda:0 / mps (session default only).
- **Alg switches & params**: controls generated from each root under `models`; changes apply to **this run only** and are not written back (except run mode).
- Ready check: page polls `GET /health/ready`; overlay while models are not ready.

### REST API

Mounted via `create_serving_app()` on FastAPI (`gr.mount_gradio_app` so `demo.launch()` does not clobber custom routes).

| Method | Path | Notes |
|:---|:---|:---|
| `GET` | `/health/ready` | Ready probe; 503 if not ready |
| `POST` | `/insect_3_predict?input_type=url` | Pest infer; body `{"url": "<http(s)://... or local path>"}` |

**Success** (HTTP 200):

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

**Error**: HTTP 200 + `{"code": 500, "msg": "..."}` (same as historical Flask API).

### Programmatic mount

```python
from script.predict_all_gradio import create_app, create_serving_app

# Gradio Blocks only (test or external mount)
demo = create_app(config_path=None, infer_device="cuda:0")

# Gradio + REST (production)
app = create_serving_app(config_path=None, cache_size=3, infer_device="cuda:0")
# uvicorn.run(app, host="0.0.0.0", port=37860)
```

### Concurrency & performance

Controlled by `predict_cfg` (see [算法配置说明.md](./算法配置说明.md)):

| Key | Notes |
|:---|:---|
| `run_count` | `>1` starts multiple **infer worker processes** (`predict_worker_pool.InferenceProcessPool`), each with its own models |
| `api_concurrency` | In-process HTTP/Gradio threads; with `run_count=1`, GPU infer locks per weight |
| `use_gpu_crop` / `cls_batch_size` / `detect_seg_batch_size` | Crop & batching |
| `trt_switch` | TensorRT `.engine` |

`run_count>1` helps under **concurrent requests**; multi-process on one GPU is usually sublinear—use `run_count=1` for single-stream bench.

---

## Directory layout

| Path | Notes |
|:---|:---|
| **`predict_all.py`** | Unified infer entry (detect/onnx/segment multi-root, recursive cls, `draw_results`, batch validation) |
| **`predict_all_gradio.py`** | Gradio test + `/insect_3_predict` REST |
| **`predict_all_xingneng.py`** | Perf stress (on `predict_all`) |
| `config/` | Static config & loaders: JSON, `cls_merge.py`, `insect_info.py`, … (`config/AGENTS.md`) |
| `config_paths.py` | Paths, `run_model` profile merge, `resolve_effective_insect_alg_path()` |
| `predict/` | Low-level models: `model_detect.py`, `model_detect_onnx.py`, `model_cls*.py` / `model_cls_timm.py`, `model_seg.py`, `model_trt.py`, … |
| `predict_size.py` | **`PredictSize`**: tiled detect → size/route → optional cls (YOLO/ONNX shared) |
| `predict_seg.py` / `predict_seg_lib.py` | **`PredictSeg`**, seg routing, polygon post, VOC/validation helpers |
| `predict_size_validate_lib.py` | Label check, confusion matrix, eval drawing |
| `predict_worker_pool.py` | Multi-process pool when `run_count>1` |
| `predict_cls_validate_from_xml.py` | Cls validation from XML |
| `predict_mark.py` | Annotation helpers |
| `ls_classification_ingest.py` / `ls_seg_classification_ingest.py` | Cls crops → Label Studio or local export |
| `test_api.py` / `test_api_yin.py` | REST smoke tests |
| `tools/` | ONNX/TRT export, ROI, XML analysis, threshold grid search (`tools/README.md`) |
| `算法配置说明.md` | **`insect_alg_all.json` field guide** (ops; Chinese) |
| `README-cn.md` | Chinese overview (this doc’s Chinese twin) |

---

## Runtime notes

- Package imports insert the **`insect` project root** (parent of `script`) into `sys.path`.
- Run from project root, or set `PYTHONPATH` to include `insect/`.
- After JSON edits or `run_model` switches: **`release()` and rebuild the pipeline**, or **restart Gradio/uvicorn**—weights are cached in-process.
- Model paths: `model_dir` is OS-resolved; prefer absolute paths on the deploy host for per-model `model`.
- Deps: base `requirments.txt`; ONNX / TRT / 5090 envs in `requirments-export-onnx.txt`, `requirments-trt.txt`, `requirments-wsl-5090.txt`, etc. Prod A10-6Q (driver CUDA 11.4): `requirments-prod-cuda114.txt`.
