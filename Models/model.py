"""双图多任务 GNN：晶体图（节点/键）+ 键角线图，四头顶层（formation / gap / edges(cbm,vbm) / vacancy）。"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool, global_mean_pool

NODE_DIM = 149
EDGE_DIM = 48
ANGLE_IN_DIM = 1


class AngleFourier(nn.Module):
    def __init__(self, k_max: int = 8, out_dim: int = 32):
        super().__init__()
        self.register_buffer("k", torch.arange(1.0, k_max + 1.0).view(1, -1))
        self.net = nn.Sequential(
            nn.Linear(2 * k_max + 1, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, deg: torch.Tensor) -> torch.Tensor:
        deg = deg.reshape(-1, 1)
        rad = deg * math.pi / 180.0
        feats = torch.cat([deg / 180.0, torch.sin(rad * self.k), torch.cos(rad * self.k)], dim=-1)
        return self.net(feats)


class CrystalConv(nn.Module):
    def __init__(self, hidden: int = 64, conv_dim: int = 128):
        super().__init__()
        self.gate = nn.Linear(3 * hidden, conv_dim)
        self.message = nn.Linear(3 * hidden, conv_dim)
        self.update = nn.Linear(hidden + conv_dim, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        pair = torch.cat([h[src], h[dst], e], dim=-1)
        msg = torch.sigmoid(self.gate(pair)) * F.softplus(self.message(pair))
        agg = h.new_zeros(h.size(0), msg.size(-1))
        agg = agg.index_add_(0, dst, msg)
        upd = torch.cat([h, agg], dim=-1)
        return self.norm(h + F.softplus(self.update(upd)))


class LineConv(nn.Module):
    def __init__(self, hidden: int = 64, angle_feat_dim: int = 32, expand: int = 2):
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(2 * hidden + angle_feat_dim, expand * hidden),
            nn.SiLU(),
            nn.Linear(expand * hidden, hidden),
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, e: torch.Tensor, line_edge_index: torch.Tensor, angle_feat: torch.Tensor) -> torch.Tensor:
        target, source = line_edge_index[0], line_edge_index[1]
        pair = torch.cat([e[target], e[source], angle_feat], dim=-1)
        msg = self.message(pair)
        agg = e.new_zeros(e.size(0), msg.size(-1))
        agg = agg.index_add_(0, target, msg)
        ones = torch.ones_like(target, dtype=e.dtype)
        deg = e.new_zeros(e.size(0)).index_add_(0, target, ones)
        agg = agg / deg.clamp_min(1.0).unsqueeze(-1)
        return self.norm(e + agg)


class ABlock(nn.Module):
    def __init__(self, hidden: int = 64, conv_dim: int = 128, angle_feat_dim: int = 32):
        super().__init__()
        self.crystal = CrystalConv(hidden, conv_dim)
        self.line = LineConv(hidden, angle_feat_dim)


class Backbone(nn.Module):
    def __init__(self, hidden: int = 64, conv_dim: int = 128, blocks: int = 4,
                 angle_k: int = 8, angle_feat_dim: int = 32,
                 node_dim: int = NODE_DIM, edge_dim: int = EDGE_DIM):
        super().__init__()
        self.node_enc = nn.Sequential(nn.Linear(node_dim, hidden), nn.SiLU(), nn.LayerNorm(hidden))
        self.edge_enc = nn.Sequential(nn.Linear(edge_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.angle_enc = AngleFourier(angle_k, angle_feat_dim)
        self.blocks = nn.ModuleList([ABlock(hidden, conv_dim, angle_feat_dim) for _ in range(blocks)])

    def forward(self, batch) -> torch.Tensor:
        h = self.node_enc(batch.x)
        e = self.edge_enc(batch.edge_attr)
        angle_feat = self.angle_enc(batch.line_edge_attr)
        for block in self.blocks:
            h = block.crystal(h, batch.edge_index, e)
            e = block.line(e, batch.line_edge_index, angle_feat)
        return h


def _last_linear(module: nn.Module) -> nn.Linear:
    return module if isinstance(module, nn.Linear) else module[-1]


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden1: int | None = 128,
                 hidden2: int | None = 64, dropout: float = 0.0):
        super().__init__()
        if not hidden1:
            self.net = nn.Linear(in_dim, out_dim)
            return
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden1), nn.SiLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        if hidden2:
            layers += [nn.Linear(hidden1, hidden2), nn.SiLU(), nn.Linear(hidden2, out_dim)]
        else:
            layers.append(nn.Linear(hidden1, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VacancyHead(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h).view(-1)


class MultiTaskModel(nn.Module):
    def __init__(self, hidden: int = 64, conv_dim: int = 128, blocks: int = 4,
                 angle_k: int = 8, angle_feat_dim: int = 32, dropout: float = 0.05,
                 global_dim: int = 0, pool_norm: bool = False, use_metal: bool = False,
                 head_hidden1: int | None = 128, head_hidden2: int | None = 64,
                 heads=("formation", "gap", "edges", "vacancy"),
                 node_dim: int = NODE_DIM, edge_dim: int = EDGE_DIM):
        super().__init__()
        unsupported = set(heads) - {"formation", "gap", "edges", "vacancy", "metal"}
        if unsupported:
            raise ValueError(f"不支持的头: {sorted(unsupported)}")
        self.use_metal = bool(use_metal) or ("metal" in heads)
        self.backbone = Backbone(hidden, conv_dim, blocks, angle_k, angle_feat_dim, node_dim, edge_dim)
        self.global_dim = int(global_dim)
        pooled_dim = 2 * hidden + self.global_dim
        self.pooled_dim = pooled_dim
        self.pool_norm = nn.LayerNorm(pooled_dim) if pool_norm else None
        self.head_formation = MLPHead(pooled_dim, 1, head_hidden1, head_hidden2, dropout=dropout)
        self.head_gap = MLPHead(pooled_dim, 1, head_hidden1, head_hidden2, dropout=dropout)
        self.head_edges = MLPHead(pooled_dim, 2, head_hidden1, head_hidden2, dropout=dropout)
        self.head_vacancy = VacancyHead(hidden)
        self.head_metal = MLPHead(pooled_dim, 1, head_hidden1, head_hidden2, dropout=dropout) if self.use_metal else None

    def backbone_parameters(self):
        return self.backbone.parameters()

    def head_parameters(self):
        modules = [self.head_formation, self.head_gap, self.head_edges, self.head_vacancy]
        if self.head_metal is not None:
            modules.append(self.head_metal)
        if self.pool_norm is not None:
            modules.append(self.pool_norm)
        for module in modules:
            yield from module.parameters()

    def forward(self, batch) -> dict:
        h = self.backbone(batch)
        pooled = torch.cat([global_mean_pool(h, batch.batch), global_max_pool(h, batch.batch)], dim=-1)
        if self.global_dim:
            feat = getattr(batch, "global_feat", None)
            if feat is None:
                raise RuntimeError("global_dim > 0 但 batch 缺少 global_feat 属性（需先在数据侧补全并保持两阶段一致）")
            pooled = torch.cat([pooled, feat.view(batch.num_graphs, -1)], dim=-1)
        if self.pool_norm is not None:
            pooled = self.pool_norm(pooled)
        edges = self.head_edges(pooled)
        out = {
            "formation": self.head_formation(pooled).view(-1),
            "gap": self.head_gap(pooled).view(-1),
            "edges": edges,
            "cbm": edges[:, 0],
            "vbm": edges[:, 1],
            "vacancy": self.head_vacancy(h),
            "pooled": pooled,
        }
        if self.head_metal is not None:
            out["metal"] = self.head_metal(pooled).view(-1)
        return out

    @torch.no_grad()
    def init_head_biases(self, targets: dict) -> None:
        def z_mean(name: str) -> float:
            info = targets.get(name) or {}
            return float(info.get("loss_mean_z", 0.0))

        _last_linear(self.head_formation.net).bias.fill_(z_mean("formation"))
        _last_linear(self.head_gap.net).bias.fill_(z_mean("gap"))
        _last_linear(self.head_edges.net).bias[0] = z_mean("cbm")
        _last_linear(self.head_edges.net).bias[1] = z_mean("vbm")
        _last_linear(self.head_vacancy.net).bias.fill_(z_mean("vacancy"))
        if self.head_metal is not None:
            frac = float((targets.get("gap") or {}).get("metal_frac", 0.0))
            frac = min(max(frac, 1e-4), 1.0 - 1e-4)
            _last_linear(self.head_metal.net).bias.fill_(math.log(frac / (1.0 - frac)))

    def parameter_counts(self) -> dict:
        backbone = sum(p.numel() for p in self.backbone.parameters())
        heads = sum(p.numel() for p in self.head_parameters())
        total = sum(p.numel() for p in self.parameters())
        return {"backbone": backbone, "heads": heads, "total": total}
