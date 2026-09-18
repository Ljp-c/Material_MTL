"""探测：MP API 全库 material_id 拉取是"流式返回"还是"先下载大缓存"。

用途: S3 直连缓慢时，判断能否改走 API 路线收集全库 ID。
判读:
    计数持续增长 + mp_datasets 不涨  -> API 路线可行
    长时间无计数 + mp_datasets 在涨 -> mp-api 在从 S3 下载全量缓存，API 路线不可行
用法:
    python probe_full_ids.py
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

from mp_api.client import MPRester

CACHE_DIR = Path.home() / "mp_datasets"
PRINT_EVERY = 10000


def dir_size_mb():
    if not CACHE_DIR.exists():
        return float("nan")
    total = 0
    for path in CACHE_DIR.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            pass
    return total / 1e6


def monitor(stop, interval=10):
    while not stop.wait(interval):
        print(f"    [监控] mp_datasets 缓存: {dir_size_mb():.1f} MB")


def main():
    print(f"mp_datasets 初始大小: {dir_size_mb():.1f} MB")
    print("开始拉取全库 material_id ...（若 5 分钟无任何计数增长，Ctrl+C 停止即可）")
    stop = threading.Event()
    threading.Thread(target=monitor, args=(stop,), daemon=True).start()
    start = time.time()
    count = 0
    with MPRester() as mpr:
        for doc in mpr.materials.summary.search(fields=["material_id"]):
            count += 1
            if count % PRINT_EVERY == 0:
                print(f"  已收到 {count} 条 (耗时 {time.time() - start:.0f}s)")
    stop.set()
    print(f"完成: 共 {count} 条, 总耗时 {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
