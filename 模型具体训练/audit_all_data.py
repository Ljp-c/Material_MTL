"""统计 data/ 下全部数据集的规模，输出总览报告。

输出: data/数据总览.json + 控制台摘要
用法: python audit_all_data.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA = Path(__file__).resolve().parent / "data"


def count_csv(path):
    try:
        with open(path, encoding="utf-8-sig", errors="replace") as handle:
            return sum(1 for _ in handle) - 1
    except Exception:
        return None


def dir_size_mb(path):
    try:
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e6
    except Exception:
        return None


def main():
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_root": str(DATA),
        "datasets": {},
    }
    if not DATA.exists():
        print("data 目录不存在:", DATA)
        return 1

    for directory in sorted(DATA.iterdir()):
        if not directory.is_dir():
            continue
        entry = {"size_mb": round(dir_size_mb(directory) or 0, 1), "files": {}}
        for path in sorted(directory.iterdir()):
            if path.is_file():
                item = {"size_mb": round(path.stat().st_size / 1e6, 2)}
                if path.suffix.lower() == ".csv":
                    item["rows"] = count_csv(path)
                entry["files"][path.name] = item
            elif path.is_dir():
                item = {"dir_size_mb": round(dir_size_mb(path) or 0, 1)}
                csvs = sorted(path.glob("*.csv"))
                if csvs:
                    item["csv"] = {c.name: count_csv(c) for c in csvs}
                entry["files"][path.name + "/"] = item
        report["datasets"][directory.name] = entry

    out = DATA / "数据总览.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 88)
    print("data/ 数据集总览")
    print("=" * 88)
    total = 0
    for name, entry in report["datasets"].items():
        total += entry["size_mb"]
        print(f"\n[{name}]  目录大小 {entry['size_mb']:.1f} MB")
        for filename, info in entry["files"].items():
            if "rows" in info and info["rows"] is not None:
                print(f"    {filename:42s} {info['rows']:,} 行  ({info['size_mb']:.1f} MB)")
            else:
                size = info.get("size_mb", info.get("dir_size_mb"))
                print(f"    {filename:42s} {size:.1f} MB")
    print("\n" + "=" * 88)
    print(f"data/ 总占用: {total:,.1f} MB")
    print(f"总览已保存: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
