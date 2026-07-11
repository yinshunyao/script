#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Detail  : 虫情识别 REST API 测试（纯 JSON HTTP，非 Gradio Client）。
#           客户端只发 JSON、收 JSON；url 原样透传，不做路径改写。
#
#           POST /insect_3_predict?input_type=url
#           Body: {"url": "/data/models/模拟试卷/Image_20260615163459586.jpg"}

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any
from urllib import error, parse, request

logger = logging.getLogger(__name__)

# DEFAULT_BASE_URL = "http://192.168.0.52:37860"
DEFAULT_BASE_URL = "http://192.168.1.123:37860"
PREDICT_PATH = "/insect_3_predict"


def post_json(
    url: str,
    body: dict[str, Any],
    *,
    timeout: float = 120.0,
) -> tuple[int, dict[str, Any]]:
    """发送 POST JSON 请求，返回 (HTTP 状态码, 解析后的 body)。"""
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json; charset=utf-8"}
    req = request.Request(url, data=data, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            status = resp.status
    except error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        status = exc.code
    except error.URLError as exc:
        raise ConnectionError(f"无法连接服务: {exc}") from exc

    try:
        parsed: dict[str, Any] = json.loads(raw) if raw else {}
    except json.JSONDecodeError as exc:
        raise ValueError(f"响应非 JSON (HTTP {status}): {raw[:500]}") from exc
    return status, parsed


def build_predict_url(base_url: str, *, input_type: str = "url") -> str:
    endpoint = f"{base_url.rstrip('/')}{PREDICT_PATH}"
    query = parse.urlencode({"input_type": input_type})
    return f"{endpoint}?{query}"


def post_insect_predict(
    base_url: str,
    image_url: str,
    *,
    input_type: str = "url",
    timeout: float = 120.0,
) -> tuple[int, dict[str, Any]]:
    """POST /insect_3_predict?input_type=url，Body {"url": ...}（url 原样透传）。"""
    url = build_predict_url(base_url, input_type=input_type)
    return post_json(url, {"url": image_url}, timeout=timeout)


def validate_success_response(body: dict[str, Any]) -> list[str]:
    """校验成功响应结构，返回问题列表（空表示通过）。"""
    issues: list[str] = []
    if "code" in body:
        issues.append(f"业务错误: code={body.get('code')} msg={body.get('msg')}")
        return issues
    if "results" not in body:
        issues.append("缺少 results 字段")
        return issues
    results = body["results"]
    if not isinstance(results, list):
        issues.append("results 不是数组")
        return issues
    for i, item in enumerate(results):
        if not isinstance(item, dict):
            issues.append(f"results[{i}] 不是对象")
            continue
        for key in ("name", "score", "location"):
            if key not in item:
                issues.append(f"results[{i}] 缺少 {key}")
        loc = item.get("location")
        if isinstance(loc, dict):
            for lk in ("left", "top", "width", "height"):
                if lk not in loc:
                    issues.append(f"results[{i}].location 缺少 {lk}")
        else:
            issues.append(f"results[{i}].location 不是对象")
    return issues


def _print_json(body: dict[str, Any]) -> None:
    print(json.dumps(body, ensure_ascii=False, indent=2))


_print_lock = threading.Lock()


@dataclass
class PredictResult:
    index: int
    url: str
    name: str
    ok: bool
    http_status: int
    body: dict[str, Any] | None
    elapsed: float
    error: str = ""


def predict_one(
    index: int,
    base_url: str,
    image_url: str,
    *,
    timeout: float = 120.0,
) -> PredictResult:
    """请求单张图片，返回结构化结果（线程安全，不直接打印）。"""
    name = image_url.rsplit("/", 1)[-1]
    t0 = time.perf_counter()
    try:
        status, body = post_insect_predict(base_url, image_url, timeout=timeout)
    except ConnectionError as exc:
        return PredictResult(
            index=index,
            url=image_url,
            name=name,
            ok=False,
            http_status=0,
            body=None,
            elapsed=time.perf_counter() - t0,
            error=str(exc),
        )
    elapsed = time.perf_counter() - t0
    issues = validate_success_response(body) if status == 200 else [f"HTTP {status}"]
    ok = status == 200 and not issues
    error = ""
    if not ok:
        if body and body.get("msg"):
            error = str(body.get("msg"))
        elif issues:
            error = "; ".join(issues)
    return PredictResult(
        index=index,
        url=image_url,
        name=name,
        ok=ok,
        http_status=status,
        body=body,
        elapsed=elapsed,
        error=error,
    )


def _print_predict_result(
    result: PredictResult,
    *,
    total: int,
    verbose: bool,
) -> None:
    label = f"[{result.index + 1}/{total}] {result.name}"
    with _print_lock:
        print(f"\n{'=' * 60}")
        print(f"用例: {label}")
        print(f"  请求 body={json.dumps({'url': result.url}, ensure_ascii=False)}")
        print(f"  HTTP {result.http_status}  耗时 {result.elapsed:.2f}s")
        if result.error and not result.ok:
            print(f"  FAIL - {result.error}")
        if verbose and result.body is not None:
            print("  响应 JSON:")
            _print_json(result.body)
        elif result.body is not None:
            if "code" in result.body:
                print(
                    f"  响应: code={result.body.get('code')} msg={result.body.get('msg')}"
                )
            else:
                n = len(result.body.get("results") or [])
                print(f"  响应: results={n} 条")
        if result.ok:
            count = len((result.body or {}).get("results") or [])
            print(f"  PASS - results 共 {count} 条")


def join_server_image_path(image_dir: str, image_name: str) -> str:
    """由配置的服务端目录与文件名拼成完整 url（仅测试配置用，请求体原样发送）。"""
    if image_name.startswith("/"):
        return image_name
    base = image_dir.rstrip("/")
    return f"{base}/{image_name}"


def resolve_test_image_urls(
    *,
    image_dir: str | None = None,
    image_names: list[str] | None = None,
    image_urls: list[str] | None = None,
    image_url: str | None = None,
) -> list[str]:
    """
    解析待测图片 url 列表（优先级从高到低）：

    1. ``image_urls``：完整服务端绝对路径列表
    2. ``image_dir`` + ``image_names``：目录 + 文件名列表
    3. ``image_url``：单张完整路径
    """
    if image_urls:
        urls = [u for u in image_urls if u]
    elif image_names and image_dir:
        urls = [join_server_image_path(image_dir, n) for n in image_names if n]
    elif image_url:
        urls = [image_url]
    else:
        urls = []
    if not urls:
        raise ValueError(
            "请设置 TEST_IMAGE_URLS，或 TEST_IMAGE_DIR + TEST_IMAGE_NAMES，或 TEST_IMAGE_URL"
        )
    return urls


def _print_summary(
    results: list[PredictResult],
    *,
    wall_elapsed: float | None = None,
    concurrency: int = 1,
) -> int:
    failed = sum(1 for r in results if not r.ok)
    print(f"\n{'=' * 60}")
    print("汇总:")
    for r in results:
        count = len((r.body or {}).get("results") or []) if r.body else 0
        status = "PASS" if r.ok else "FAIL"
        line = (
            f"  {status:4s}  {r.name:45s}  results={count}  "
            f"{r.elapsed:.2f}s"
        )
        if r.error:
            line += f"  ({r.error})"
        print(line)
    print(
        f"\n合计: {len(results) - failed} 通过, {failed} 失败 / 共 {len(results)} 张"
    )

    elapsed_list = [r.elapsed for r in results]
    if elapsed_list:
        avg = sum(elapsed_list) / len(elapsed_list)
        min_t = min(elapsed_list)
        max_t = max(elapsed_list)
        min_r = min(results, key=lambda r: r.elapsed)
        max_r = max(results, key=lambda r: r.elapsed)
        print("\n耗时统计:")
        if wall_elapsed is not None:
            print(f"  总耗时: {wall_elapsed:.2f}s")
        print(f"  平均耗时: {avg:.2f}s")
        print(f"  最小耗时: {min_t:.2f}s  ({min_r.name})")
        print(f"  最大耗时: {max_t:.2f}s  ({max_r.name})")
        sum_elapsed = sum(elapsed_list)
        if wall_elapsed is not None and sum_elapsed > 0:
            print(f"  单请求耗时合计: {sum_elapsed:.2f}s")
        print(
            "  说明: 客户端耗时 = HTTP 往返 + 服务端读图 + 推理排队 + pipeline.predict；"
            "服务端日志「pipeline.predict 耗时」仅含模型推理，不含读图与网络。"
        )
        if concurrency > 1:
            print("\n并发说明:")
            print(
                f"  客户端并发={concurrency}，服务端单 GPU + 单 pipeline 串行推理，"
                "总墙钟≈单请求耗时之和，不会因并发而缩短。"
            )
            if wall_elapsed is not None and sum_elapsed > 0:
                ratio = wall_elapsed / sum_elapsed
                print(
                    f"  本次 总耗时/单请求合计 ≈ {ratio:.2f} "
                    f"({'接近串行' if ratio < 1.15 else '存在重叠或网络抖动'})"
                )
            print(
                "  并发下单请求变慢，是排队等待其它请求完成；"
                "测单张延迟请设 CONCURRENCY=1。"
            )
            print(
                "  GPU 利用率低常见原因：滑窗多小 batch、CPU 后处理穿插、"
                "nvidia-smi 采样间隔，并非未走 GPU。"
            )

    return failed


def main(
    base_url: str,
    image_urls: list[str],
    *,
    timeout: float = 120.0,
    verbose: bool = False,
    concurrency: int = 1,
) -> int:
    """调用 API 测试图片列表；concurrency>1 时并发请求。返回失败用例数。"""
    total = len(image_urls)
    workers = max(1, int(concurrency))
    t_all = time.perf_counter()

    if workers == 1:
        results: list[PredictResult] = []
        for i, url in enumerate(image_urls):
            result = predict_one(i, base_url, url, timeout=timeout)
            _print_predict_result(result, total=total, verbose=verbose)
            results.append(result)
        return _print_summary(
            results, wall_elapsed=time.perf_counter() - t_all, concurrency=workers
        )

    print(f"并发模式: concurrency={workers}")
    results_map: dict[int, PredictResult] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(predict_one, i, base_url, url, timeout=timeout): i
            for i, url in enumerate(image_urls)
        }
        for fut in as_completed(futures):
            result = fut.result()
            results_map[result.index] = result
            if verbose:
                _print_predict_result(result, total=total, verbose=True)
            else:
                status = "PASS" if result.ok else "FAIL"
                with _print_lock:
                    print(
                        f"  完成 [{result.index + 1}/{total}] {result.name}  "
                        f"{status}  HTTP {result.http_status}  "
                        f"{result.elapsed:.2f}s  results="
                        f"{len((result.body or {}).get('results') or [])}"
                    )

    results = [results_map[i] for i in range(total)]
    return _print_summary(
        results, wall_elapsed=time.perf_counter() - t_all, concurrency=workers
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # 入口参数（IDE 直接运行即可，按需修改）
    # BASE_URL = "http://192.168.0.52:37860"
    # BASE_URL = "http://192.168.1.123:37860/"
    BASE_URL = "http://127.0.0.1:37860/"
    TIMEOUT = 120.0

    # 方式一（推荐）：服务端目录 + 文件名列表（拼成完整 url 后原样发送）
    TEST_IMAGE_DIR = "/home/beyond/桌面/模型识别/田间采集/原始图片"
    ll_result = """202607092012_1952342830425243648.jpg
202607092021_1952342830425243648.jpg
202607092022_1952342830425243648.jpg
202607092031_1952342830425243648.jpg
202607092032_1952342830425243648.jpg
202607092041_1952342830425243648.jpg
202607092042_1952342830425243648.jpg
202607092052_1952342830425243648 (1).jpg
202607092052_1952342830425243648.jpg
202607092102_1952342830425243648 (1).jpg
202607092102_1952342830425243648.jpg
202607092112_1952342830425243648 (1).jpg
202607092112_1952342830425243648.jpg
202607092122_1952342830425243648 (1).jpg
202607092122_1952342830425243648.jpg
202607092132_1952342830425243648 (1).jpg
202607092132_1952342830425243648.jpg
202607092142_1952342830425243648 (1).jpg
202607092142_1952342830425243648.jpg
202607092152_1952342830425243648 (1).jpg
202607092152_1952342830425243648.jpg
202607092202_1952342830425243648 (1).jpg
202607092202_1952342830425243648.jpg
202607092212_1952342830425243648 (1).jpg
202607092212_1952342830425243648.jpg
202607092222_1952342830425243648 (1).jpg
202607092222_1952342830425243648.jpg
202607092232_1952342830425243648 (1).jpg
202607092232_1952342830425243648.jpg
202607092242_1952342830425243648 (1).jpg
202607092242_1952342830425243648.jpg
202607092252_1952342830425243648.jpg
202607092253_1952342830425243648.jpg"""
    TEST_IMAGE_NAMES: list[str] = ll_result.split("\n")

    # 方式二：完整路径列表（非空时优先于方式一）
    TEST_IMAGE_URLS: list[str] = []

    # 方式三：单张完整路径（仅当方式一、二均未配置时使用）
    TEST_IMAGE_URL = "/data/models/模拟试卷/Image_20260615163459586.jpg"

    # 多张批量时默认精简输出；单张或需调试时可设 True 打印完整 JSON
    VERBOSE = len(TEST_IMAGE_NAMES) <= 1 and not TEST_IMAGE_URLS
    # 并发数：1=顺序请求（测单张延迟）；>1 仅压测排队，单 GPU 服务总墙钟不会明显下降
    CONCURRENCY = 4

    try:
        image_urls = resolve_test_image_urls(
            image_dir=TEST_IMAGE_DIR,
            image_names=TEST_IMAGE_NAMES,
            image_urls=TEST_IMAGE_URLS or None,
            image_url=TEST_IMAGE_URL,
        )
    except ValueError as exc:
        print(f"配置错误: {exc}")
        sys.exit(2)

    mode = f"并发={CONCURRENCY}" if CONCURRENCY > 1 else "顺序"
    print(f"共 {len(image_urls)} 张待测图片，{mode} 调用 API ...")
    fail_count = main(
        BASE_URL,
        image_urls,
        timeout=TIMEOUT,
        verbose=VERBOSE,
        concurrency=CONCURRENCY,
    )
    sys.exit(min(fail_count, 255))
