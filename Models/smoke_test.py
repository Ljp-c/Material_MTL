r"""最小自检（不读数据文件）：批偏移 / 前向形状 / 掩码损失 / 反向。

用法（工作目录 E:\Material_MTL）:
    python Models\smoke_test.py
    python Models\smoke_test.py --device cuda
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import CrystalData, collate
from losses import MultiTaskLoss
from model import MultiTaskModel


def synthetic_sample(seed: int, n_atoms: int = 8, n_edges: int = 24, n_lines: int = 48,
                     with_labels: bool = True, with_vacancy: bool = True, with_angles: bool = True):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n_atoms, 149, generator=generator)
    z = torch.randint(1, 83, (n_atoms,), generator=generator)
    edge_index = torch.randint(0, n_atoms, (2, n_edges), generator=generator)
    edge_attr = torch.rand(n_edges, 48, generator=generator)
    if with_angles:
        line_edge_index = torch.randint(0, n_edges, (2, n_lines), generator=generator)
        line_edge_attr = torch.rand(n_lines, 1, generator=generator) * 180.0
    else:
        line_edge_index = torch.zeros((2, 0), dtype=torch.long)
        line_edge_attr = torch.zeros((0, 1), dtype=torch.float32)
    data = CrystalData(
        x=x, z=z, edge_index=edge_index, edge_attr=edge_attr,
        line_edge_index=line_edge_index, line_edge_attr=line_edge_attr,
    )
    labels = torch.full((1, 4), float("nan"))
    if with_labels:
        labels[0, 0] = -3.1
        labels[0, 1] = 1.4
        labels[0, 2] = 0.9
        labels[0, 3] = -0.5
    data.labels = labels
    if with_vacancy:
        data.vacancy = torch.rand(n_atoms, 1, generator=generator) * 3.0 - 1.0
    else:
        data.vacancy = torch.full((n_atoms, 1), float("nan"))
    data.global_feat = torch.randn(1, 8, generator=generator)
    data.name = f"syn-{seed}"
    data.split_group = f"syn-{seed}"
    data.n_atoms = int(n_atoms)
    return data


def synthetic_stats() -> dict:
    return {"targets": {
        "formation": {"mean": -2.0, "std": 1.2, "n": 100, "loss_n": 100, "loss_mean_z": 0.0},
        "gap": {"mean": 0.8, "std": 0.9, "n": 100, "loss_n": 60, "loss_mean_z": 0.35, "metal_frac": 0.4},
        "cbm": {"mean": 1.5, "std": 1.1, "n": 60, "loss_n": 60, "loss_mean_z": 0.0},
        "vbm": {"mean": 0.7, "std": 1.0, "n": 60, "loss_n": 60, "loss_mean_z": 0.0},
        "vacancy": {"mean": 1.0, "std": 1.5, "n": 500, "loss_n": 500, "loss_mean_z": 0.0},
    }}


def main() -> int:
    parser = argparse.ArgumentParser(description="合成数据自检")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[device] CUDA 不可用，改用 CPU")
        args.device = "cpu"
    device = torch.device(args.device)

    sample_a = synthetic_sample(1, n_atoms=8, n_edges=24, n_lines=48)
    sample_b = synthetic_sample(2, n_atoms=5, n_edges=12, n_lines=0, with_angles=False)
    sample_b.labels[0, 1] = 0.0
    sample_b.labels[0, 2] = float("nan")
    sample_b.labels[0, 3] = float("nan")

    batch = collate([sample_a, sample_b])
    edges_a = int(sample_a.edge_index.size(1))
    lines_a = int(sample_a.line_edge_index.size(1))
    shifted = batch.line_edge_index[:, lines_a:]
    assert torch.equal(shifted, sample_b.line_edge_index + edges_a), "line_edge_index 批偏移错误（不应重复偏移）"
    assert int(batch.labels.size(0)) == 2 and int(batch.labels.size(1)) == 4
    assert int(batch.vacancy.size(0)) == 8 + 5
    print("[ok] 批次拼接：line_edge_index 偏移、labels [B,4]、vacancy [ΣN,1]")

    model = MultiTaskModel(hidden=16, conv_dim=16, blocks=2, angle_k=4, angle_feat_dim=8,
                           dropout=0.0, global_dim=8, use_metal=True).to(device)
    model.init_head_biases(synthetic_stats()["targets"])
    batch = batch.to(device)
    out = model(batch)
    assert out["formation"].shape == (2,)
    assert out["gap"].shape == (2,)
    assert out["cbm"].shape == (2,)
    assert out["vbm"].shape == (2,)
    assert out["vacancy"].shape == (13,)
    assert out["metal"].shape == (2,)
    assert out["pooled"].shape == (2, 2 * 16 + 8)
    print("[ok] 前向输出形状（含单原子图/零键角图/8 维全局特征）")

    criterion = MultiTaskLoss(
        weights={"formation": 1.0, "gap": 1.0, "cbm": 0.5, "vbm": 0.5, "vacancy": 1.0, "metal": 0.2},
        betas={"formation": 0.05, "gap": 0.1, "cbm": 0.1, "vbm": 0.1, "vacancy": 0.1},
        consistency_weight=0.1, consistency_ramp_frac=0.05,
        use_metal=True, label_stats=synthetic_stats(),
    ).to(device)
    total, info = criterion(out, batch.labels, batch.vacancy, step=50, total_steps=1000)
    assert total is not None and torch.isfinite(total), "损失应为有限值"
    expected = {"formation", "gap", "cbm", "vbm", "consistency", "vacancy", "metal"}
    assert expected <= set(info["terms"]), f"缺损失项: {expected - set(info['terms'])}"
    total.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads), "梯度出现 NaN/Inf"
    assert any(g.abs().sum() > 0 for g in grads), "梯度全零"
    print("[ok] 掩码多任务损失与前向/反向")

    empty = collate([synthetic_sample(3, with_labels=False, with_vacancy=False)]).to(device)
    out_empty = model(empty)
    total_empty, info_empty = criterion(out_empty, empty.labels, empty.vacancy, step=0, total_steps=0)
    assert total_empty is None and not info_empty["terms"], "全掩码 batch 应跳过损失"
    print("[ok] 全掩码 batch 跳过")

    probe = MultiTaskModel(hidden=8, conv_dim=8, blocks=1, angle_k=2, angle_feat_dim=4, pool_norm=True)
    out_probe = probe(collate([synthetic_sample(4)]))
    assert out_probe["formation"].shape == (1,) and out_probe["vacancy"].shape == (8,)
    angle_probe = probe.backbone.angle_enc(torch.zeros(0, 1))
    assert angle_probe.shape[0] == 0
    print("[ok] pool_norm 变体与空键角输入")

    stripped = synthetic_sample(5)
    delattr(stripped, "global_feat")
    try:
        model(collate([stripped]).to(device))
        raise AssertionError("global_dim=8 且缺 global_feat 时应报错")
    except RuntimeError as error:
        assert "global_feat" in str(error)
    print("[ok] global_feat 缺失时报错")

    print("smoke test 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
