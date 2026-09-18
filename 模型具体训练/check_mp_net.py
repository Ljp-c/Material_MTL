"""检查与 Materials Project 相关网络服务的连通性 and 下载速度。

用途: 定位"脚本卡住不下载"的原因（API 不通 / AWS S3 不通 / 速度过慢）。
用法:
    python check_mp_net.py
"""
from __future__ import annotations

import time

import requests

TARGETS = [
    ("MP API", "https://api.materialsproject.org/"),
    ("AWS build bucket", "https://materialsproject-build.s3.amazonaws.com/"),
]

SPEED_URL = ("https://materialsproject-build.s3.amazonaws.com/"
             "collections/2024-11-14/summary/manifest.jsonl.gz")


def check(name, url, timeout=(8, 15)):
    start = time.time()
    try:
        response = requests.get(url, timeout=timeout)
        elapsed = time.time() - start
        print(f"[OK]   {name}  HTTP {response.status_code}  {elapsed:.2f}s  "
              f"{len(response.content) / 1000:.1f} KB")
    except Exception as exc:
        elapsed = time.time() - start
        print(f"[FAIL] {name}  {elapsed:.2f}s  {type(exc).__name__}: {exc}")


def speed_test(url, seconds=8):
    start = time.time()
    total = 0
    try:
        with requests.get(url, stream=True, timeout=(8, 20)) as response:
            response.raise_for_status()
            for chunk in response.iter_content(1 << 16):
                total += len(chunk)
                if time.time() - start > seconds:
                    break
    except Exception as exc:
        print(f"[FAIL] 下载测速  {type(exc).__name__}: {exc}")
        return
    elapsed = max(time.time() - start, 1e-6)
    rate = total / 1e6 / elapsed
    print(f"[OK]   下载测速  {total / 1e6:.2f} MB / {elapsed:.1f}s = {rate:.2f} MB/s")


def main():
    print("== 连通性检查 ==")
    for name, url in TARGETS:
        check(name, url)
    print("== S3 分片下载测速（采样 8 秒，测的是 MP AWS 数据源） ==")
    speed_test(SPEED_URL)
    print("提示: 若 S3 测速 < 0.1 MB/s 或 FAIL，全库直连下载会非常慢，")
    print("      可尝试开启代理后重跑本脚本（requests 会自动读取系统代理）。")


if __name__ == "__main__":
    main()
