r"""计算标签 z-score 统计（预训练/微调两阶段共用），输出 Models/stats/label_stats.json。

用法（工作目录 E:\Material_MTL）:
    python Models\label_stats.py --config Models\configs\pretrain.yaml
    python Models\label_stats.py --config Models\configs\pretrain.yaml --limit-shards 1 --vacancy-shards 1

口径：预训练配置下 train 划分、剔除家族（batio3 / batio3_doped / 含 Ba 的 vacancy 组成）；
空位标签按位点池化（读取前 --vacancy-shards 个分片）；全局特征统计基于同一 train 划分。
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import REPO_ROOT, load_config
from data import (GRAPH_TARGETS, GLOBAL_FEATURE_COLUMNS, SampleFilter, Source,
                  load_pretrain_exclude_groups)

MODELS_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="标签 z-score 统计")
    parser.add_argument("--config", default=str(MODELS_DIR / "configs" / "pretrain.yaml"))
    parser.add_argument("--out", default=str(MODELS_DIR / "stats" / "label_stats.json"))
    parser.add_argument("--vacancy-shards", type=int, default=16, help="空位位点标签统计读取的分片数（0=全部分片）")
    parser.add_argument("--limit-shards", type=int, default=None, help="每个数据源限制分片数（冒烟）")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    if cfg.get("stage") not in (None, "pretrain"):
        print(f"警告: 建议使用预训练配置统计（当前 stage={cfg.get('stage')}）")
    split_cfg = cfg.get("split") or {"val_frac": 0.0, "test_frac": 0.0, "seed": 0}

    exclude_groups = load_pretrain_exclude_groups() if cfg.get("exclude_family", True) else set()
    print(f"[pretrain 排除] 微调留出组成 {len(exclude_groups)} 个（仅 val/test，其余家族数据回流）")

    common_filters = cfg.get("filters") or {}
    sources: dict[str, Source] = {}
    for entry in cfg.get("sources") or []:
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
        source = Source(name, split_cfg, filters=filters, exclude_groups=exclude_groups,
                        limit_shards=args.limit_shards)
        sources[name] = source
        print(f"[source] {name:26s} train={source.counts['train']:7d} "
              f"预训练排除={source.n_family:6d} 过滤={source.n_filtered:6d}")

    acc = {name: {"count": 0, "sum": 0.0, "sumsq": 0.0,
                  "loss_count": 0, "loss_sum": 0.0, "metal_count": 0} for name in GRAPH_TARGETS}
    for name, source in sources.items():
        if not source.labels:
            continue
        table = source.table
        train_mask = pd.Series(table.index.map(source.split), index=table.index) == "train"
        if not bool(train_mask.any()):
            continue
        gap_column = source.labels.get("gap")
        gap_values = None
        if gap_column is not None and gap_column in table.columns:
            gap_values = pd.to_numeric(table.loc[train_mask, gap_column], errors="coerce").to_numpy(dtype=np.float64)
        for slot, column in source.labels.items():
            if column not in table.columns:
                print(f"  警告: {name} 缺列 {column}")
                continue
            values = pd.to_numeric(table.loc[train_mask, column], errors="coerce").to_numpy(dtype=np.float64)
            valid = np.isfinite(values)
            if slot == "gap":
                valid &= values > 0
                if gap_values is not None:
                    acc["gap"]["metal_count"] += int(np.sum(np.isfinite(gap_values) & (gap_values == 0)))
            elif slot in ("cbm", "vbm") and gap_values is not None:
                valid &= np.isfinite(gap_values) & (gap_values > 0)
            values = values[valid]
            entry = acc[slot]
            entry["count"] += int(values.size)
            entry["sum"] += float(values.sum())
            entry["sumsq"] += float((values ** 2).sum())
            entry["loss_count"] += int(values.size)
            entry["loss_sum"] += float(values.sum())
            mean_text = f"{values.mean():.4f}" if values.size else "nan"
            print(f"  [{name}/{slot}] n={values.size} 均值={mean_text}")

    vacancy_record = {"mean": 0.0, "std": 1.0, "n": 0, "loss_n": 0, "loss_mean_z": 0.0}
    vac_source = sources.get("vacancy_screening")
    if vac_source is not None and vac_source.has_vacancy:
        shards = list(vac_source.crystal_shards)
        if args.vacancy_shards and args.vacancy_shards > 0:
            shards = shards[: args.vacancy_shards]
        count = 0
        total = 0.0
        total_sq = 0.0
        start = time.time()
        for index, path in enumerate(shards):
            with open(path, "rb") as fh:
                crystal = pickle.load(fh)
            for mid, sample in crystal.items():
                if vac_source.split.get(mid) != "train":
                    continue
                vacancy = getattr(sample, "vacancy", None)
                if vacancy is None:
                    continue
                values = vacancy.reshape(-1).to(torch.float64)
                values = values[torch.isfinite(values)]
                if values.numel():
                    count += int(values.numel())
                    total += float(values.sum())
                    total_sq += float((values * values).sum())
            print(f"  空位统计 {index + 1}/{len(shards)} 分片: 累计 {count} 个位点 ({time.time() - start:.0f}s)")
        if count:
            mean = total / count
            variance = max(total_sq / count - mean * mean, 0.0)
            vacancy_record = {"mean": mean, "std": math.sqrt(variance), "n": count,
                              "loss_n": count, "loss_mean_z": 0.0}
    if vacancy_record["n"] == 0:
        raise SystemExit("空位位点统计为空：确认配置含 vacancy_screening，并调大 --vacancy-shards")

    global_feat_stats = None
    require_global = int(((cfg.get("model") or {}).get("global_dim", 0)) or 0) > 0
    buckets = []
    for name, source in sources.items():
        feature_path = source.dir / "global_feat.csv"
        if not feature_path.exists():
            if require_global:
                raise SystemExit(f"缺少 {feature_path}；先运行 python specific_MT\\data_prep\\build_global_feat.py")
            continue
        frame = pd.read_csv(feature_path)
        missing_columns = [column for column in GLOBAL_FEATURE_COLUMNS if column not in frame.columns]
        if missing_columns:
            raise SystemExit(f"{feature_path} 缺列 {missing_columns}")
        frame["material_id"] = frame["material_id"].astype(str)
        frame = frame.set_index("material_id")
        train_mask = pd.Series(source.table.index.map(source.split), index=source.table.index) == "train"
        mids = source.table.index[train_mask]
        values = frame.reindex(mids)[list(GLOBAL_FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
        values = values[np.isfinite(values).all(axis=1)]
        print(f"  [全局特征] {name}: {values.shape[0]} / {len(mids)} 条")
        if values.shape[0]:
            buckets.append(values)
    if buckets:
        stacked = np.concatenate(buckets, axis=0)
        global_feat_stats = {
            "columns": list(GLOBAL_FEATURE_COLUMNS),
            "mean": stacked.mean(axis=0).tolist(),
            "std": np.clip(stacked.std(axis=0), 1e-8, None).tolist(),
            "n": int(stacked.shape[0]),
        }
    if require_global and global_feat_stats is None:
        raise SystemExit("配置 global_dim>0 但未找到任何 global_feat.csv")

    targets_out: dict[str, dict] = {}
    for name in GRAPH_TARGETS:
        entry = acc[name]
        if entry["count"] == 0:
            raise SystemExit(f"目标 {name} 统计为空：检查数据源与 --limit-shards")
        mean = entry["sum"] / entry["count"]
        variance = max(entry["sumsq"] / entry["count"] - mean * mean, 0.0)
        std = max(math.sqrt(variance), 1e-8)
        loss_mean = entry["loss_sum"] / entry["loss_count"] if entry["loss_count"] else mean
        record = {"mean": mean, "std": std, "n": entry["count"], "loss_n": entry["loss_count"],
                  "loss_mean_z": (loss_mean - mean) / std}
        if name == "gap":
            record["metal_frac"] = entry["metal_count"] / max(entry["count"] + entry["metal_count"], 1)
        targets_out[name] = record
    targets_out["vacancy"] = vacancy_record

    payload = {
        "version": 1,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "population": {
            "config": str(Path(args.config)),
            "split": "train",
            "exclude_family": bool(cfg.get("exclude_family", True)),
            "pretrain_exclude_groups": len(exclude_groups),
            "vacancy_shards": args.vacancy_shards,
            "limit_shards": args.limit_shards,
        },
        "targets": targets_out,
        "global_feat": global_feat_stats,
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"[out] {out}")
    for name, record in targets_out.items():
        print(f"  {name:10s} mean={record['mean']:9.4f} std={record['std']:8.4f} "
              f"n={record['n']:8d} loss_mean_z={record['loss_mean_z']:+.4f}")
    if global_feat_stats is not None:
        print(f"  global_feat n={global_feat_stats['n']} "
              f"mean={np.round(global_feat_stats['mean'], 3).tolist()}")
        print(f"              std={np.round(global_feat_stats['std'], 3).tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
