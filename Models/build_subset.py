r"""物化小数据源子集缓存：对配置里标记 subset_cache: true 的数据源全量扫描一次，
把过滤后的样本（已准备：标签/空位/全局特征齐备）写入 graph/<源>/subsets/subset_<key>.pkl。
之后训练/评估每个 epoch 直接读缓存（内存载入一次），不再重扫全部分片。

用法（工作目录 E:\Material_MTL）:
    python Models\build_subset.py --config Models\configs\finetune.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import file_digest, load_config, resolve_path
from data import SampleFilter, Source, load_pretrain_exclude_groups, materialize_subset, subset_key

MODELS_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="物化小数据源子集缓存（一次性全量扫描）")
    parser.add_argument("--config", default=str(MODELS_DIR / "configs" / "finetune.yaml"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    paths = cfg.get("paths") or {}
    stats_path = resolve_path(paths.get("stats") or "Models/stats/label_stats.json")
    global_stats = None
    stats_hash = None
    if stats_path is not None and stats_path.exists():
        with open(stats_path, "r", encoding="utf-8") as fh:
            global_stats = json.load(fh).get("global_feat")
        stats_hash = file_digest(stats_path)
    else:
        print("警告: 缺少 label_stats.json，缓存里的全局特征会按缺失处理；建议先运行 label_stats.py")

    exclude_groups = load_pretrain_exclude_groups() if cfg.get("exclude_family") else set()
    common_filters = cfg.get("filters") or {}
    built = 0
    for entry in cfg.get("sources") or []:
        if not entry.get("subset_cache"):
            continue
        name = entry["name"]
        filters_cfg = dict(common_filters)
        filters_cfg.update(entry.get("filters") or {})
        filters = SampleFilter(
            min_atoms=int(filters_cfg.get("min_atoms") or 0),
            max_atoms=filters_cfg.get("max_atoms"),
            contains_elements=tuple(filters_cfg.get("contains_elements") or ()),
            perovskite_batio3=bool(filters_cfg.get("perovskite_batio3", False)),
            exclude_formulas=tuple(filters_cfg.get("exclude_formulas") or ()),
        )
        source = Source(name, cfg["split"], filters=filters, exclude_groups=exclude_groups,
                        limit_shards=None, global_stats=global_stats,
                        require_global=global_stats is not None,
                        subset_cache=True, stats_hash=stats_hash)
        total = source.counts["train"] + source.counts["val"] + source.counts["test"]
        print(f"[build] {name}: 合格样本 {total} 条（train {source.counts['train']} / "
              f"val {source.counts['val']} / test {source.counts['test']}），开始全量扫描 ...")
        out = materialize_subset(source)
        size_mb = out.stat().st_size / 1024 / 1024
        print(f"[done] {name}: {out}（{size_mb:.0f} MB）", flush=True)
        manifest_path = out.parent / "manifest.json"
        manifest = {}
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                manifest = {}
        manifest[out.name] = {
            "source": name,
            "key": subset_key(source),
            "n": total,
            "stats_hash": stats_hash,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        built += 1
    if built == 0:
        print("配置里没有 subset_cache: true 的数据源，无需构建")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
