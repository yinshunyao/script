#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Detail  : 虫情 Gradio 客户端：本地图片上传后调用 /predict。
#           依赖：pip install 'gradio_client>=1.5'
#           任意目录可运行，只需改下方 BASE_URL / LOCAL_IMAGE。

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

try:
    from gradio_client import Client, handle_file
except ImportError:
    print("缺少依赖：pip install 'gradio_client>=1.5'", file=sys.stderr)
    sys.exit(2)


def predict_local_image(
    base_url: str,
    image_path: str | Path,
    *,
    api_name: str = "/predict",
) -> dict[str, Any]:
    """上传本地图片并调用 Gradio ``/predict``，返回结果 dict。"""
    path = Path(image_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"本地图片不存在: {path}")

    client = Client(base_url.rstrip("/") + "/")
    raw = client.predict(handle_file(str(path)), api_name=api_name)
    if isinstance(raw, str):
        return json.loads(raw)
    if not isinstance(raw, dict):
        raise TypeError(f"/predict 返回应为 dict，实际为 {type(raw)!r}")
    return raw


def print_summary(data: dict[str, Any]) -> None:
    if data.get("error"):
        raise RuntimeError(f"业务错误: {data['error']}")
    results = data.get("results") or []
    count = int(data.get("count") or len(results))
    print(f"OK: count={count}")
    print(json.dumps(data, ensure_ascii=False, indent=2))
    if results:
        first = results[0]
        print(
            f"首条: name={first.get('name')!r} cn={first.get('cn_name')!r} "
            f"det={first.get('det_conf')} cls={first.get('cls_conf')} "
            f"location={first.get('location')}"
        )


if __name__ == "__main__":
    BASE_URL = "http://117.172.230.60:37860"
    LOCAL_IMAGE = (
        "/Users/shunyaoyin/Documents/code/ai-company/insect/data/"
        "大图标注范例/202605082122_2011264891516174343.jpg"
    )
    API_NAME = "/predict"

    print(f"Gradio Client → {BASE_URL.rstrip('/')}/  api_name={API_NAME}")
    print(f"本地上传: {LOCAL_IMAGE}")
    try:
        result = predict_local_image(BASE_URL, LOCAL_IMAGE, api_name=API_NAME)
        print_summary(result)
    except Exception as exc:
        print(f"失败: {exc}", file=sys.stderr)
        sys.exit(1)
