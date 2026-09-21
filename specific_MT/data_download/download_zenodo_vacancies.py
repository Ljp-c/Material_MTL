"""下载 Zenodo 空位筛选数据集（86,259 种材料的空位形成能）。

来源: doi:10.5281/zenodo.15025795（论文 "Screening of material defects using
      universal machine-learning interatomic potentials", 2025）
文件: README.txt / Vacancies.db|Vacancies.json / 2D_layers.db|2D_layers.json
输出: data/全库_空位形成能/
特性: 从 Zenodo API 自动列文件、HTTP Range 断点续传、进度条

用法:
    python download_zenodo_vacancies.py
    python download_zenodo_vacancies.py --record 15025795
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
from tqdm import tqdm

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://zenodo.org/",
}


def list_files(record):
    url = f"https://zenodo.org/api/records/{record}"
    response = requests.get(url, headers=HEADERS, timeout=60)
    response.raise_for_status()
    data = response.json()
    files = []
    for item in data.get("files", []):
        key = item.get("key")
        size = item.get("size") or 0
        link = (item.get("links") or {}).get("self")
        if key and link:
            files.append((key, size, link))
    return data.get("metadata", {}).get("title"), files


def download(url, dest, total, retries=4):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if total and dest.exists() and dest.stat().st_size == total:
        print(f"[skip] {dest.name} 已完整 ({total / 1e6:.1f} MB)")
        return
    for attempt in range(retries):
        resume = dest.stat().st_size if dest.exists() else 0
        if total and resume >= total:
            print(f"[done] {dest.name} 已完整 ({total / 1e6:.1f} MB)")
            return
        headers = dict(HEADERS)
        if resume:
            headers["Range"] = f"bytes={resume}-"
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(20, 300)) as response:
                response.raise_for_status()
                if resume and response.status_code != 206:
                    resume = 0
                mode = "ab" if resume else "wb"
                remaining = int(response.headers.get("Content-Length") or 0)
                with open(dest, mode) as handle, tqdm(
                        total=total or (resume + remaining) or None,
                        initial=resume, unit="B", unit_scale=True,
                        desc=dest.name) as bar:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        handle.write(chunk)
                        bar.update(len(chunk))
            if not total or dest.stat().st_size >= total:
                print(f"[done] {dest.name}: {dest.stat().st_size / 1e6:.1f} MB")
                return
        except Exception as exc:
            print(f"[retry {attempt + 1}] {dest.name}: {type(exc).__name__}: {exc}")
    print(f"[fail] {dest.name} 未完成，重跑本脚本可续传")


def main():
    parser = argparse.ArgumentParser(description="下载 Zenodo 空位筛选数据集")
    parser.add_argument("--record", default="15025795")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data" / "全库_空位形成能"))
    args = parser.parse_args()

    out_dir = Path(args.out)
    title, files = list_files(args.record)
    print(f"记录: {title}")
    print(f"目标目录: {out_dir}")
    print(f"共 {len(files)} 个文件:")
    for key, size, _ in files:
        print(f"  - {key}  {size / 1e6:.1f} MB")

    for key, size, link in files:
        print("=" * 80)
        download(link, out_dir / key, size)

    readme = out_dir / "README.txt"
    if readme.exists():
        print("=" * 80)
        print("README.txt 内容:")
        print(readme.read_text(encoding="utf-8", errors="replace")[:4000])


if __name__ == "__main__":
    raise SystemExit(main())
