r"""对指定 split 做逐样本预测导出（CSV）与分层误差统计。

用法（工作目录 E:\Material_MTL）:
    python Models\predict.py --ckpt Models\artifacts\pretrain\best.pt --split test
    python Models\predict.py --config Models\configs\finetune.yaml --ckpt Models\artifacts\finetune\best.pt --split test
    python Models\predict.py --ckpt Models\artifacts\pretrain\best.pt --split test --limit-shards 2

输出 CSV 列：source, material_id, target, true, pred, residual, abs_error, site_index
（空位为位点级，site_index 为该位点在结构内的编号；其余为图级。）
同时打印：按 (source,target) 的 n/MAE/RMSE/R²，以及 gap 按真实值区间、vacancy 按 |真实值| 区间的分层 MAE。
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import file_digest, load_config, load_torch, resolve_path
from data import GRAPH_TARGETS, MultiSourceBatcher, collate
from metrics import regression_metrics
import train as T

GAP_BUCKETS = ((0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, float("inf")))
VACANCY_BUCKETS = ((0.0, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, float("inf")))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="逐样本预测导出与分层误差统计")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent / "configs" / "pretrain.yaml"))
    parser.add_argument("--ckpt", required=True, help="checkpoint 路径（best.pt / last.pt）")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out", default=None, help="CSV 输出路径（默认与 ckpt 同目录 preds_<split>.csv）")
    parser.add_argument("--limit-shards", type=int, default=None, help="每个数据源只读前 N 个分片（冒烟）")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def bucket_name(value: float, buckets) -> str:
    for low, high in buckets:
        if low <= value < high:
            return f"[{low:g},{high:g})"
    return "other"


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    device_name = str(args.device or cfg.get("device", "cuda"))
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    paths = cfg.get("paths") or {}
    stats_path = resolve_path(paths.get("stats") or "Models/stats/label_stats.json")
    stats = T.load_label_stats(stats_path)
    stats_hash = file_digest(stats_path)
    target_stats = stats["targets"]

    ckpt_path = resolve_path(args.ckpt)
    if ckpt_path is None or not ckpt_path.exists():
        raise SystemExit(f"找不到权重 {ckpt_path}")
    checkpoint = load_torch(ckpt_path)

    sources = T.build_sources(cfg, args.limit_shards, stats, stats_hash=stats_hash)
    model, model_cfg = T.build_model(cfg, stats, checkpoint)
    model = model.to(device).eval()

    train_cfg = cfg.get("train") or {}
    batcher = MultiSourceBatcher(
        sources,
        batch_size=int(args.batch_size or train_cfg.get("batch_size", 64)),
        sampling=str(cfg.get("sampling", "sqrt_inv")),
        seed=int(cfg.get("seed", 42)),
        max_atoms_per_batch=train_cfg.get("max_atoms_per_batch"),
    )

    mean = torch.tensor([float(target_stats[name]["mean"]) for name in GRAPH_TARGETS], dtype=torch.float32)
    std = torch.tensor([max(float(target_stats[name]["std"]), 1e-8) for name in GRAPH_TARGETS], dtype=torch.float32)
    vac_mean = torch.tensor(float(target_stats["vacancy"]["mean"]), dtype=torch.float32)
    vac_std = torch.tensor(max(float(target_stats["vacancy"]["std"]), 1e-8), dtype=torch.float32)

    out = resolve_path(args.out) if args.out else ckpt_path.parent / f"preds_{args.split}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    store = defaultdict(lambda: {"pred": [], "true": []})
    buckets = defaultdict(lambda: {"pred": [], "true": []})
    rows_written = 0

    with open(out, "w", newline="", encoding="utf-8") as fh, torch.no_grad():
        writer = csv.writer(fh)
        writer.writerow(["source", "material_id", "target", "true", "pred", "residual", "abs_error", "site_index"])
        for source_name, items in batcher.eval_batches(args.split):
            batch = collate(items).to(device)
            preds = model(batch)
            names = list(batch.name)
            labels = batch.labels

            for slot, target in enumerate(GRAPH_TARGETS):
                raw = labels[:, slot]
                mask = torch.isfinite(raw)
                if target == "gap":
                    mask = mask & (raw > 0)
                elif target in ("cbm", "vbm"):
                    gap_raw = labels[:, 1]
                    mask = mask & torch.isfinite(gap_raw) & (gap_raw > 0)
                if not bool(mask.any()):
                    continue
                pred_phys = (preds[target] * std[slot] + mean[slot]).cpu().numpy()
                true_np = raw.cpu().numpy()
                for i in torch.nonzero(mask, as_tuple=False).view(-1).tolist():
                    true_value = float(true_np[i])
                    pred_value = float(pred_phys[i])
                    writer.writerow([source_name, names[i], target, f"{true_value:.6f}", f"{pred_value:.6f}",
                                     f"{pred_value - true_value:.6f}", f"{abs(pred_value - true_value):.6f}", ""])
                    store[(source_name, target)]["true"].append(true_value)
                    store[(source_name, target)]["pred"].append(pred_value)
                    if target == "gap":
                        key = (target, bucket_name(true_value, GAP_BUCKETS))
                        buckets[key]["true"].append(true_value)
                        buckets[key]["pred"].append(pred_value)
                    rows_written += 1

            vacancy_raw = batch.vacancy.reshape(-1)
            mask_v = torch.isfinite(vacancy_raw)
            if bool(mask_v.any()):
                pred_phys = (preds["vacancy"] * vac_std + vac_mean).cpu().numpy()
                true_np = vacancy_raw.cpu().numpy()
                mask_np = mask_v.cpu().numpy()
                ptr = batch.ptr.tolist()
                for graph_index, name in enumerate(names):
                    start, end = ptr[graph_index], ptr[graph_index + 1]
                    for site_index in np.nonzero(mask_np[start:end])[0].tolist():
                        true_value = float(true_np[start + site_index])
                        pred_value = float(pred_phys[start + site_index])
                        writer.writerow([source_name, name, "vacancy", f"{true_value:.6f}", f"{pred_value:.6f}",
                                         f"{pred_value - true_value:.6f}", f"{abs(pred_value - true_value):.6f}", site_index])
                        store[(source_name, "vacancy")]["true"].append(true_value)
                        store[(source_name, "vacancy")]["pred"].append(pred_value)
                        key = ("vacancy", bucket_name(abs(true_value), VACANCY_BUCKETS))
                        buckets[key]["true"].append(true_value)
                        buckets[key]["pred"].append(pred_value)
                        rows_written += 1

    print(f"[out] {out}（{rows_written} 行）")
    print(f"=== {args.split} 按来源×目标 ===")
    for (source_name, target), values in sorted(store.items()):
        metric = regression_metrics(torch.tensor(values["pred"]), torch.tensor(values["true"]))
        print(f"  {source_name:26s} {target:10s} n={metric['n']:6d} MAE={metric['mae']:.4f} "
              f"RMSE={metric['rmse']:.4f} R2={metric['r2']:.3f}")
    if buckets:
        print("=== 分层 MAE ===")
        for (target, bucket), values in sorted(buckets.items()):
            pred = torch.tensor(values["pred"])
            true = torch.tensor(values["true"])
            metric = regression_metrics(pred, true)
            print(f"  {target:8s} {bucket:12s} n={metric['n']:6d} MAE={metric['mae']:.4f} RMSE={metric['rmse']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
