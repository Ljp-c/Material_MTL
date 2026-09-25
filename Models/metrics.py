"""回归指标（物理单位）：MAE / RMSE / R² / 样本数。"""
from __future__ import annotations

import torch


def regression_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    pred = pred.reshape(-1).float()
    target = target.reshape(-1).float()
    n = int(pred.numel())
    if n == 0:
        return {"n": 0, "mae": None, "rmse": None, "r2": None}
    diff = pred - target
    mae = float(diff.abs().mean())
    rmse = float(torch.sqrt((diff ** 2).mean()))
    ss_res = float((diff ** 2).sum())
    ss_tot = float(((target - target.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"n": n, "mae": mae, "rmse": rmse, "r2": r2}


def format_metric(metric: dict) -> str:
    if not metric or not metric.get("n"):
        return "n=0"
    r2 = metric.get("r2")
    r2_text = "nan" if r2 is None or r2 != r2 else f"{r2:.3f}"
    return f"n={metric['n']:6d} MAE={metric['mae']:.4f} RMSE={metric['rmse']:.4f} R2={r2_text}"
