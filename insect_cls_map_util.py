# -*- coding: utf-8 -*-
"""
``insect_cls_map.py`` 中 ``cls_map`` 的配置解析工具。

支持两种顶层结构：

1) **新版（推荐）**：分类 / 检测增强目标系数分轨
   ```python
   {
     "cls": { "foo": 1, "bar": { "bar-a": 0.6, "bar-b": 0.4 }, ... },
     "detect": { ... 可与 cls 不同 ... }
   }
   ```
   - 省略 ``detect`` 或设为 ``None``：检测流水线沿用 ``cls`` 的配置。
   - ``cls`` 下标量系数允许 **> 1**（目标张数 ≈ ``expect_count * coef * group_coef``）。
   - 检测训练里「标注叶子名 → 输出组名」的 **dict/list 合并拓扑** 应使用 ``unwrap_insect_cls_map_for_merge(..., pipeline="detect")``
     或 ``extract_merge_groups_from_insect_cls_map_raw(..., pipeline="detect")``，与 ``detect`` 轨一致。

2) **旧版（兼容）**：整表即单层「类 → 系数 / dict / list」；分类与检测共用。
"""

from __future__ import annotations

import math
from typing import Any, Literal

PipelineBranch = Literal["cls", "detect"]


def get_default_insect_cls_map_raw() -> dict[str, Any]:
    """加载 ``cls_map``：优先 ``from script.insect_cls_map``（``insect/`` 在 ``sys.path``），否则 ``import insect_cls_map``（``script/`` 在 ``sys.path``）。"""
    raw: Any
    try:
        from script.insect_cls_map import cls_map as raw  # noqa: PLC0415
    except ImportError:
        import insect_cls_map as _m  # noqa: PLC0415

        raw = getattr(_m, "cls_map", None)
    if not isinstance(raw, dict) or not raw:
        raise ValueError("insect_cls_map.cls_map 须为非空 dict")
    return raw


def extract_merge_groups_from_insect_cls_map_raw(
    raw: dict[str, Any] | None = None,
    *,
    only_other_prefix: bool = False,
    pipeline: PipelineBranch = "cls",
) -> dict[str, list[str]]:
    """
    从 cls_map 的根（或 unwrap 后的 cls / detect 轨）提取「合并组」：仅 **value 为 dict 或 list** 的条目。

    - ``only_other_prefix=True``：仅保留 key 以 ``other`` 开头的组（评估脚本混淆矩阵等价合并）。
    - ``only_other_prefix=False``：全部 dict/list 组（检测训练合并等）。
    - ``pipeline``：新版 ``cls`` / ``detect`` 双轨时，从哪条轨读 dict/list 拓扑；``"detect"`` 与检测数据管线、
      ``load_aug_map_from_insect_cls_map(..., pipeline="detect")`` 对齐。缺省 ``"cls"`` 保持历史调用兼容。
    """
    if raw is None:
        raw = get_default_insect_cls_map_raw()
    if not isinstance(raw, dict):
        return {}

    body = unwrap_insect_cls_map_for_merge(raw, pipeline=pipeline)
    out: dict[str, list[str]] = {}
    for k, v in body.items():
        group = str(k or "").strip()
        if not group:
            continue
        if only_other_prefix and not group.startswith("other"):
            continue

        aliases: list[str] = []
        if isinstance(v, dict):
            aliases = [str(x).strip() for x in v.keys() if str(x).strip()]
        elif isinstance(v, list):
            aliases = [str(x).strip() for x in v if str(x).strip()]
        else:
            continue

        seen: set[str] = set()
        uniq: list[str] = []
        for a in aliases:
            if a not in seen:
                seen.add(a)
                uniq.append(a)
        if uniq:
            out[group] = uniq
    return out


def unwrap_insect_cls_map_for_merge(
    raw: dict[str, Any],
    *,
    pipeline: PipelineBranch = "cls",
) -> dict[str, Any]:
    """
    返回用于解析「合并组」（dict/list 拓扑）的配置体。

    - 旧版平面 ``cls_map``（无 ``cls`` 键）：直接返回 ``raw``。
    - 新版含 ``cls`` 对象：
      - ``pipeline="cls"``：返回 ``cls`` 轨（分类/历史行为）。
      - ``pipeline="detect"``：若存在非空 ``detect`` 对象则返回之，否则回退到 ``cls``
        （与 ``get_aug_coeff_map_body(..., pipeline="detect")`` 在 detect 省略时的行为一致）。
    """
    cls_body = raw.get("cls")
    if isinstance(cls_body, dict) and cls_body:
        if pipeline == "detect":
            det = raw.get("detect")
            if isinstance(det, dict) and det:
                return det
            return cls_body
        return cls_body
    return raw


def get_aug_coeff_map_body(raw: dict[str, Any], *, pipeline: PipelineBranch) -> dict[str, Any]:
    """返回用于展平 aug_map 的配置体（cls 或 detect 分支）。"""
    cls_body = raw.get("cls")
    if isinstance(cls_body, dict) and cls_body:
        if pipeline == "cls":
            return cls_body
        det = raw.get("detect")
        if det is None:
            return cls_body
        if not isinstance(det, dict):
            raise ValueError(f'insect_cls_map["detect"] 须为对象或省略/null，当前: {type(det).__name__}')
        if not det:
            raise ValueError('insect_cls_map["detect"] 为空对象；若需与 cls 相同请省略 detect 键')
        return det
    if pipeline == "detect" and ("detect" in raw or "cls" in raw):
        raise ValueError(
            "使用了新版键 cls/detect 但其中 cls 为空或缺失；请提供非空的 \"cls\" 对象，或改回旧版平面结构"
        )
    return raw


def load_aug_map_from_insect_cls_map(
    raw: dict[str, Any] | None = None,
    *,
    expect_count: int,
    group_coef: float = 1.0,
    pipeline: PipelineBranch = "detect",
) -> dict[str, Any]:
    """
    将 ``cls_map`` 展平为流水线 ``aug_map``（叶子文件夹名 → 目标张数或 0）。

    - ``raw is None``：使用 ``get_default_insect_cls_map_raw()``（即 ``insect_cls_map.cls_map``）。
    - ``pipeline="cls"``：分类流水线（train_cls/09）
    - ``pipeline="detect"``：检测流水线（train_detect/09）

    **分类轨（``pipeline="cls"``）** 标量 ``c``：目标 ``round(expect_count * c * group_coef)``；``c==0`` 写入 ``0``。
    dict：组内权重归一化后分配 ``expect_count * group_coef``；list：均分该预算。

    **检测轨（``pipeline="detect"``）** 数值语义：``v <= 10`` 视为**系数**（目标 ``round(expect_count * v * group_coef)``）；``v > 10`` 视为**预期张数**
    （目标 ``round(v * group_coef)``）。dict 内若存在 ``>10`` 的项，则逐项：大于 10 为预期张数，否则为系数；若**全部** ``<=10``，仍按组内权重归一化分配 ``expect_count * group_coef``。
    """
    if raw is None:
        raw = get_default_insect_cls_map_raw()
    if not isinstance(raw, dict) or not raw:
        raise ValueError("insect_cls_map 配置须为非空 dict")

    body = get_aug_coeff_map_body(raw, pipeline=pipeline)
    return flatten_insect_cls_coeff_map(
        body, expect_count=expect_count, group_coef=group_coef, pipeline=pipeline
    )


def flatten_insect_cls_coeff_map(
    map_body: dict[str, Any],
    *,
    expect_count: int,
    group_coef: float = 1.0,
    pipeline: PipelineBranch = "cls",
) -> dict[str, Any]:
    """将单层「分组配置」展平为 aug_map（供测试或外部传入 dict 子对象后调用）。

    ``pipeline="detect"`` 时标量/dict 数值采用「≤10 系数、>10 预期张数」语义，见 ``load_aug_map_from_insect_cls_map`` 文档。
    """
    ec = float(expect_count)
    gc = float(group_coef)
    if not math.isfinite(ec) or ec < 0:
        raise ValueError(f"expect_count 须为非负有限数，当前: {expect_count!r}")
    if not math.isfinite(gc) or gc < 0:
        raise ValueError(f"group_coef 须为非负有限数，当前: {group_coef!r}")

    out: dict[str, Any] = {}
    budget = ec * gc

    def _put_leaf(name: str, target_f: float, *, ctx: str) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{ctx}: 叶子类名须为非空字符串，当前: {name!r}")
        key = name.strip()
        if key in out:
            raise ValueError(f"{ctx}: 叶子类名与配置中其他条目重复: {key!r}")
        if target_f < 0:
            raise ValueError(f"{ctx}[{key!r}] 目标张数不能为负: {target_f!r}")
        if math.isclose(target_f, 0.0, abs_tol=0.0, rel_tol=0.0):
            out[key] = 0
            return
        iv = int(round(target_f))
        out[key] = iv if math.isclose(target_f, float(iv), rel_tol=0.0, abs_tol=1e-6) else float(target_f)

    for group_key, v in map_body.items():
        if not isinstance(group_key, str) or not group_key.strip():
            raise ValueError(f"顶层 key 须为非空字符串，当前: {group_key!r}")
        gk = group_key.strip()
        ctx = f"insect_cls_map[{gk!r}]"

        if v is None:
            continue
        if isinstance(v, bool):
            raise TypeError(f"{ctx}: 禁止 bool，请用数字系数或 dict/list")
        if isinstance(v, (int, float)):
            if isinstance(v, float) and not math.isfinite(v):
                raise ValueError(f"{ctx}: 系数须为有限数，当前: {v!r}")
            coef = float(v)
            if math.isclose(coef, 0.0, abs_tol=0.0, rel_tol=0.0):
                _put_leaf(gk, 0.0, ctx=ctx)
            elif pipeline == "detect" and coef > 10.0:
                _put_leaf(gk, coef * gc, ctx=ctx)
            else:
                _put_leaf(gk, ec * coef * gc, ctx=ctx)
            continue

        if isinstance(v, dict):
            pairs: list[tuple[str, float]] = []
            for lk, w in v.items():
                if not isinstance(lk, str) or not lk.strip():
                    raise ValueError(f"{ctx}: dict key 须为非空字符串，当前: {lk!r}")
                if isinstance(w, bool) or not isinstance(w, (int, float)):
                    raise TypeError(f"{ctx}[{lk!r}]: 权重须为数字，当前 {type(w).__name__}")
                wf = float(w)
                if not math.isfinite(wf) or wf < 0:
                    raise ValueError(f"{ctx}[{lk!r}]: 权重须为非负有限数，当前: {w!r}")
                pairs.append((lk.strip(), wf))

            if pipeline == "detect" and pairs and max(w for _, w in pairs) > 10.0:
                for lk, w in pairs:
                    if math.isclose(w, 0.0, abs_tol=0.0, rel_tol=0.0):
                        continue
                    if w > 10.0:
                        _put_leaf(lk, w * gc, ctx=ctx)
                    else:
                        _put_leaf(lk, ec * w * gc, ctx=ctx)
                continue

            sum_w = sum(w for _, w in pairs)
            if sum_w <= 0:
                raise ValueError(f"{ctx}: dict 权重和须 > 0")
            for lk, w in pairs:
                if w == 0:
                    continue
                _put_leaf(lk, budget * (w / sum_w), ctx=ctx)
            continue

        if isinstance(v, list):
            leaves = [str(x).strip() for x in v if str(x).strip()]
            if not leaves:
                raise ValueError(f"{ctx}: list 不能为空")
            share = budget / float(len(leaves))
            for lk in leaves:
                _put_leaf(lk, share, ctx=ctx)
            continue

        raise TypeError(f"{ctx}: value 须为数字 / dict / list，当前: {type(v).__name__}")

    return out


def normalize_aug_map_targets(aug_map: dict[str, Any], *, expect_count: int) -> dict[str, Any]:
    """
    将 aug_map 中 (0,1] 的 float 视为占 expect_count 的比例并换算为整数目标；
    其它写法（含系数 >1 的 float、int）原样保留。
    """
    out: dict[str, Any] = {}
    ec = int(expect_count)
    for k, v in aug_map.items():
        if v is None or v == "":
            out[k] = v
            continue
        if isinstance(v, bool):
            raise TypeError(f"aug_map[{k!r}] 禁止 bool，请用 0/正整数/None")
        if isinstance(v, float) and 0.0 < float(v) <= 1.0:
            out[k] = max(1, int(round(ec * float(v))))
            continue
        out[k] = v
    return out
