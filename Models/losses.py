"""掩码多任务损失：z-score 空间 SmoothL1 + gap≈cbm-vbm 一致性项（ramp-up）+ 可选 metal BCE。"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

GRAPH_TARGETS = ("formation", "gap", "cbm", "vbm")
VACANCY = "vacancy"


class MultiTaskLoss(nn.Module):
    def __init__(self, weights: dict | None = None, betas: dict | None = None,
                 consistency_weight: float = 0.1, consistency_ramp_frac: float = 0.05,
                 use_metal: bool = False, label_stats: dict | None = None):
        super().__init__()
        if label_stats is None:
            raise ValueError("MultiTaskLoss 需要 label_stats（label_stats.py 生成）")
        weights = dict(weights or {})
        betas = dict(betas or {})
        self.w = {name: float(weights.get(name, 1.0)) for name in GRAPH_TARGETS + (VACANCY,)}
        self.w["metal"] = float(weights.get("metal", 0.2))
        self.beta = {name: float(betas.get(name, 0.1)) for name in GRAPH_TARGETS + (VACANCY,)}
        self.consistency_weight = float(consistency_weight)
        self.ramp_frac = float(consistency_ramp_frac)
        self.use_metal = bool(use_metal)
        targets = label_stats["targets"]
        mean = [float(targets[name]["mean"]) for name in GRAPH_TARGETS]
        std = [max(float(targets[name]["std"]), 1e-8) for name in GRAPH_TARGETS]
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))
        self.register_buffer("vac_mean", torch.tensor(float(targets[VACANCY]["mean"]), dtype=torch.float32))
        self.register_buffer("vac_std", torch.tensor(max(float(targets[VACANCY]["std"]), 1e-8), dtype=torch.float32))

    def forward(self, preds: dict, labels_raw: torch.Tensor, vacancy_raw: torch.Tensor,
                step: int = 0, total_steps: int = 0, force_ramp: float | None = None):
        raw = labels_raw
        vacancy_raw = vacancy_raw.reshape(-1)
        valid = torch.isfinite(raw)
        terms: dict[str, torch.Tensor] = {}
        counts: dict[str, int] = {}

        mask = valid[:, 0]
        if mask.any():
            z = (raw[mask, 0] - self.mean[0]) / self.std[0]
            terms["formation"] = F.smooth_l1_loss(preds["formation"][mask], z, beta=self.beta["formation"])
            counts["formation"] = int(mask.sum())

        mask = valid[:, 1] & (raw[:, 1] > 0)
        if mask.any():
            z = (raw[mask, 1] - self.mean[1]) / self.std[1]
            terms["gap"] = F.smooth_l1_loss(preds["gap"][mask], z, beta=self.beta["gap"])
            counts["gap"] = int(mask.sum())

        for slot, name in ((2, "cbm"), (3, "vbm")):
            mask = valid[:, slot] & (raw[:, 1] > 0)
            if mask.any():
                z = (raw[mask, slot] - self.mean[slot]) / self.std[slot]
                terms[name] = F.smooth_l1_loss(preds[name][mask], z, beta=self.beta[name])
                counts[name] = int(mask.sum())

        mask = valid[:, 1] & (raw[:, 1] > 0) & valid[:, 2] & valid[:, 3]
        if mask.any() and self.consistency_weight > 0:
            gap_eV = preds["gap"][mask] * self.std[1] + self.mean[1]
            cbm_eV = preds["cbm"][mask] * self.std[2] + self.mean[2]
            vbm_eV = preds["vbm"][mask] * self.std[3] + self.mean[3]
            residual = (gap_eV - (cbm_eV - vbm_eV)) / self.std[1]
            zero = torch.zeros_like(residual)
            terms["consistency"] = F.smooth_l1_loss(residual, zero, beta=0.1)
            counts["consistency"] = int(mask.sum())

        mask = torch.isfinite(vacancy_raw)
        if mask.any():
            z = (vacancy_raw[mask] - self.vac_mean) / self.vac_std
            terms["vacancy"] = F.smooth_l1_loss(preds["vacancy"][mask], z, beta=self.beta["vacancy"])
            counts["vacancy"] = int(mask.sum())

        if self.use_metal and "metal" in preds:
            mask = valid[:, 1]
            if mask.any():
                target = (raw[mask, 1] == 0).to(preds["metal"].dtype)
                terms["metal"] = F.binary_cross_entropy_with_logits(preds["metal"][mask], target)
                counts["metal"] = int(mask.sum())

        if force_ramp is not None:
            ramp = float(force_ramp)
        elif self.ramp_frac > 0 and total_steps and total_steps > 0:
            ramp = min(1.0, (step + 1) / max(1.0, self.ramp_frac * total_steps))
        else:
            ramp = 1.0

        weighted: dict[str, torch.Tensor] = {}
        for name, value in terms.items():
            if name == "consistency":
                weighted[name] = self.consistency_weight * ramp * value
            else:
                weighted[name] = self.w.get(name, 1.0) * value
        total = sum(weighted.values()) if weighted else None
        info = {"terms": terms, "weighted": weighted, "counts": counts, "ramp": ramp}
        return total, info
