# script/config

虫情推理、训练与标注共用的**静态配置**及配置加载模块。

## 文档范围

| 文件 | 用途 |
|:---|:---|
| `insect_alg_all.json` | 统一多根推理（detect/segment 根、`out` 路由、嵌套分类）；**启动入口**，顶层 `run_model` 切换摆拍/生产/其他 |
| `insect_alg_shengchan.json` | 生产环境算法配置（`run_model=shengchan` 时加载）；不含 `model_dir` 与 tier 清单 |
| `insect_alg_other.json` | 其他场景算法配置（`run_model=other` 时加载）；不含 `model_dir` 与 tier 清单 |
| `insect_info.json` | 物种元数据（中文名、体长、区域索引等） |
| `cls_merge.py` | 训练/评估类别层级与合并（`cls_merge` 字典） |
| `insect_info.py` | 加载 `insect_info.json`，区域索引与 c1/c2 派生 |
| `cls_hierarchy_util.py` | 加载 `cls_merge.py` / 分级 JSON，评估合并组派生 |

## 路径约定

- JSON 与 `cls_merge.py` 与本目录同级；代码通过 `script.config.*` 或 `script.config_paths` 引用。
- 相对路径（如 `config/insect_alg_all.json`）相对于 `insect/script/`。

## 编辑规则

- 修改门限与推理路由：优先改 `insect_alg_all.json` 的 `out`。
- 修改训练输出类与成员物种：改 `cls_merge.py` 的 `cls_merge`。
- 新增物种元数据：改 `insect_info.json`；批量补齐可用 `tools/add_missing_insect_info_entries.py`。
