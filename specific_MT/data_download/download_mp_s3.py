"""从 Materials Project AWS Open Data 直接下载整份数据 collection。

数据源: s3://materialsproject-build/collections/<version>/<collection>/
        gzip JSONL 分片（与 MP API 同源），无需 API key、不受 API 限速。
用途: 阶段一预训练数据的全库一次性下载；也可补下 dielectric、magnetism 等 collection。

用法:
    python download_mp_s3.py --list-versions
    python download_mp_s3.py --collection summary --list-only
    python download_mp_s3.py --collection summary --out data/mp_full
    python download_mp_s3.py --collection dielectric --version 2025-09-25 --out data/mp_full
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests
from tqdm import tqdm

BASE = "https://materialsproject-build.s3.amazonaws.com"
BUCKET = "materialsproject-build"


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_listing(content):
    root = ET.fromstring(content)
    keys = []
    prefixes = []
    truncated = False
    token = None
    for element in root:
        tag = element.tag.split("}")[-1]
        if tag == "Contents":
            key = size = None
            for child in element:
                ctag = child.tag.split("}")[-1]
                if ctag == "Key":
                    key = child.text
                elif ctag == "Size":
                    text = child.text or "0"
                    try:
                        size = int(text)
                    except (TypeError, ValueError):
                        size = 0
            if key:
                keys.append((key, size or 0))
        elif tag == "CommonPrefixes":
            for child in element:
                if child.tag.split("}")[-1] == "Prefix":
                    prefixes.append(child.text)
        elif tag == "IsTruncated":
            truncated = (element.text or "").lower() == "true"
        elif tag == "NextContinuationToken":
            token = element.text
    return keys, prefixes, truncated, token


def s3_list(prefix, delimiter=None, timeout=(10, 30), verbose=True):
    keys = []
    prefixes = []
    token = None
    page = 0
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if delimiter:
            params["delimiter"] = delimiter
        if token:
            params["continuation-token"] = token
        response = requests.get(BASE + "/", params=params, timeout=timeout)
        response.raise_for_status()
        batch_keys, batch_prefixes, truncated, token = parse_listing(response.content)
        keys.extend(batch_keys)
        prefixes.extend(batch_prefixes)
        page += 1
        if verbose:
            print(f"  列表第 {page} 页: 累计 {len(keys)} 个对象")
        if not truncated:
            break
    return keys, prefixes


def list_versions():
    _, prefixes = s3_list("collections/", delimiter="/")
    versions = []
    for prefix in prefixes:
        name = prefix.rstrip("/").split("/")[-1]
        if re.match(r"^\d{4}-\d{2}-\d{2}", name):
            versions.append(name)
    return sorted(versions)


def human_size(n):
    value = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def download_one(key, dest, force=False, progress=None, retries=5):
    if dest.exists() and dest.stat().st_size > 0 and not force:
        if progress is not None:
            progress.update(dest.stat().st_size)
        return "skip", dest
    url = f"{BASE}/{key}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    for attempt in range(retries):
        resume_from = tmp.stat().st_size if tmp.exists() else 0
        headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
        written = 0
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(10, 60)) as response:
                response.raise_for_status()
                if resume_from and response.status_code != 206:
                    resume_from = 0
                if resume_from and progress is not None:
                    progress.update(resume_from)
                mode = "ab" if resume_from else "wb"
                with open(tmp, mode) as handle:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        handle.write(chunk)
                        written += len(chunk)
                        if progress is not None:
                            progress.update(len(chunk))
            tmp.replace(dest)
            return "ok", dest
        except Exception as exc:
            last_error = exc
            if progress is not None and written:
                progress.update(-written)
            time.sleep(1.0 + attempt)
    print(f"  [失败] {key}: {last_error}")
    return "fail", dest


def merge_shards(raw_root, merged_path):
    files = sorted(raw_root.rglob("*.jsonl.gz"))
    count = 0
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(merged_path, "wt", encoding="utf-8") as out_handle:
        for path in tqdm(files, desc="merge"):
            with gzip.open(path, "rt", encoding="utf-8") as in_handle:
                for line in in_handle:
                    if line.strip():
                        out_handle.write(line)
                        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="MP AWS Open Data collection 下载器")
    parser.add_argument("--collection", default="summary",
                        help="summary / dielectric / magnetism / electronic-structure / thermo 等")
    parser.add_argument("--version", default="auto", help="如 2025-09-25；auto=最新")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data" / "mp_full"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--list-versions", action="store_true")
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--test", type=int, default=0, help="只下载前 N 个分片（连通性/速度测试）")
    parser.add_argument("--no-merge", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    print("MP AWS Open Data 下载器")
    print(f"数据源: {BASE}")

    if args.list_versions:
        for version in list_versions():
            print(version)
        return 0

    version = args.version
    if version == "auto":
        print("查询可用版本 ...")
        versions = list_versions()
        if not versions:
            print("未发现版本目录")
            return 1
        version = versions[-1]
    print(f"数据库版本: {version}")

    prefix = f"collections/{version}/{args.collection}/"
    print(f"列出分片: {prefix}")
    keys, _ = s3_list(prefix)
    files = [(key, size) for key, size in keys if not key.endswith("/")]
    total = sum(size for _, size in files)
    print(f"{args.collection}: {len(files)} 个文件, 共 {human_size(total)}")
    if args.list_only:
        for key, size in files[:20]:
            print(f"  {key}  {human_size(size)}")
        if len(files) > 20:
            print(f"  ... 其余 {len(files) - 20} 个")
        return 0
    if not files:
        print("该版本下没有该 collection，请检查名称或换版本")
        return 1

    out_dir = Path(args.out)
    raw_root = out_dir / "raw" / "collections" / version / args.collection
    tasks = []
    for key, _ in files:
        rel = key.split(f"collections/{version}/", 1)[-1]
        tasks.append((key, raw_root / rel))

    if args.test:
        tasks = tasks[:args.test]
        args.no_merge = True
        print(f"测试模式: 只下载前 {len(tasks)} 个分片，不合并")

    size_by_key = {key: size for key, size in files}
    progress_total = sum(size_by_key.get(key, 0) for key, _ in tasks)
    print(f"开始下载: {len(tasks)} 个文件, 共 {human_size(progress_total)}")

    ok = skip = fail = 0
    progress = tqdm(total=progress_total, unit="B", unit_scale=True, desc="download")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, key, dest, args.force, progress): key
                   for key, dest in tasks}
        for future in as_completed(futures):
            status, _ = future.result()
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                fail += 1
    progress.close()
    print(f"下载完成: 新下载 {ok}, 跳过 {skip}, 失败 {fail}")

    count = None
    if not args.no_merge:
        merged = out_dir / f"{args.collection}_{version}.jsonl.gz"
        count = merge_shards(raw_root, merged)
        print(f"合并: {count} 条 -> {merged} ({human_size(merged.stat().st_size)})")

    info = {
        "bucket": BUCKET,
        "collection": args.collection,
        "version": version,
        "n_files": len(files),
        "total_bytes": total,
        "downloaded": ok,
        "skipped": skip,
        "failed": fail,
        "merged_records": count,
        "created_utc": utc_now(),
    }
    (out_dir / f"download_{args.collection}_{version}_meta.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
