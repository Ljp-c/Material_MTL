"""把晶体结构 pkl 转成图数据（PyG Data），供图神经网络训练使用。

图的构造（CGCNN 风格）：
    节点 = 原子，节点特征 = 原子序数（后续用 Embedding 学）
    边   = 截断半径内的近邻原子对
    边特征 = 两原子距离的高斯 RBF 展开（默认 32 维）
    标签 = e_total（默认目标），并附带 e_ionic / e_electronic

用法：
    python build_graphs.py                      # 全量转换
    python build_graphs.py --limit 300          # 冒烟测试
    python build_graphs.py --cutoff 6 --rbf-bins 64
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm


def rbf_expand(distances: torch.Tensor, centers: torch.Tensor, width: float) -> torch.Tensor:
    """把一维距离展开成多维高斯 RBF：距离靠近哪个刻度，哪一维就接近 1。"""
    d = distances.view(-1, 1)#-1是自动计算的成为列向量
    return torch.exp(-((d - centers.view(1, -1)) ** 2) / (2.0 * width ** 2))


class GraphBuilder:
    """负责把单个 pymatgen Structure 转成一个 PyG Data（图）。"""

    def __init__(self, cutoff: float = 5.0, max_neighbors: int = 12, rbf_bins: int = 32):
        self.cutoff = float(cutoff)            # 近邻截断半径（埃）
        self.max_neighbors = int(max_neighbors)  # 每个原子最多保留几个邻居
        self.rbf_bins = int(rbf_bins)          # 距离展开成多少维
        # 在 0 到 cutoff 之间均匀撒 rbf_bins 个"距离刻度"，作为高斯函数的中心
        self.centers = torch.linspace(0.0, self.cutoff, self.rbf_bins)
        # 高斯函数的宽度 = 相邻刻度之间的间距
        self.width = self.cutoff / max(self.rbf_bins - 1, 1)

    @property
    def edge_dim(self) -> int:
        # 每条边最终的特征维度 = RBF 维度
        return self.rbf_bins

    def build(self, structure, targets: dict, target_name: str = None) -> Data:
        # ── 第 1 步：节点特征 = 每个原子的原子序数（Ba=56, Ti=22, O=8）──
        atomic_numbers = [int(getattr(site.specie, "Z", 0)) for site in structure]
        x = torch.tensor(atomic_numbers, dtype=torch.long)

        # ── 第 2 步：找邻居（自动考虑周期性，跨晶胞边界的原子也算）──
        neighbors = structure.get_all_neighbors(self.cutoff, include_index=True)
        src, dst, dist = [], [], []
        for i, nlist in enumerate(neighbors):
            # 同一个原子的邻居按距离从近到远排序，只保留最近的 max_neighbors 个
            nlist = sorted(nlist, key=lambda nb: nb.nn_distance)[: self.max_neighbors]
            for nb in nlist:
                # 边方向：邻居 -> 中心原子（中心原子从邻居收集信息）
                src.append(int(nb.index))       # 邻居在晶胞里的编号
                dst.append(i)                   # 中心原子的编号
                dist.append(float(nb.nn_distance))  # 两点距离（埃）

        # ── 第 3 步：组装边特征 ──
        if dist:
            # edge_index 形状 [2, 边数]，第 0 行是起点、第 1 行是终点
            edge_index = torch.tensor([src, dst], dtype=torch.long)
            d = torch.tensor(dist, dtype=torch.float32)
            # 把每个距离从 1 个数变成 32 维的"距离指纹"
            edge_attr = rbf_expand(d, self.centers, self.width)
        else:
            # 极端情况：这个结构一个邻居都没有（比如单原子晶胞），给空的边
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, self.rbf_bins), dtype=torch.float32)

        # ── 第 4 步：打包成一张图 ──
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        # ── 第 5 步：挂上标签（e_total 等）──
        for name, value in targets.items():
            if value is None or value != value:   # 跳过空值和 NaN
                continue
            t = torch.tensor([float(value)], dtype=torch.float32)
            data[name] = t
            if name == target_name:
                data.y = t                        # 主目标存成标准的 y

        # 顺手记录原子数和边数，方便后面做索引和统计
        data.n_atoms = int(len(structure))
        data.n_edges = int(edge_index.shape[1])
        return data


def main() -> int:
    # ── 命令行参数 ──
    parser = argparse.ArgumentParser(description="晶体结构 -> 图数据(PyG Data)")
    here = Path(__file__).resolve().parent
    parser.add_argument("--data", default=str(here / "data" / "全库_介电" / "dielectric_dataset.pkl"))
    parser.add_argument("--out", default=str(here / "data" / "全库_介电" / "graphs.pkl"))
    parser.add_argument("--cutoff", type=float, default=5.0, help="近邻截断半径 (A)")
    parser.add_argument("--max-neighbors", type=int, default=12, help="每个原子的最大近邻数")
    parser.add_argument("--rbf-bins", type=int, default=32, help="距离 RBF 展开维度")
    parser.add_argument("--target", default="e_total", help="作为标签的目标列")
    parser.add_argument("--extra-targets", nargs="*", default=["e_ionic", "e_electronic"])
    parser.add_argument("--max-atoms", type=int, default=200, help="跳过的超大晶胞阈值")
    parser.add_argument("--limit", type=int, default=None, help="限制条数（冒烟测试用）")
    args = parser.parse_args()

    # ── 1. 读取数据集（价目表 + 结构）──
    with open(args.data, "rb") as f:
        dataset = pickle.load(f)
    labels = dataset["labels"]
    structures = dataset["structures"]

    # ── 2. 可选过滤：只取前 N 条 / 跳过超大的晶胞 ──
    if args.limit:
        labels = labels.head(int(args.limit)).reset_index(drop=True)

    if args.max_atoms:
        keep = labels["nsites"].astype(float) <= args.max_atoms
        dropped = int((~keep).sum())
        labels = labels[keep].reset_index(drop=True)
        if dropped:
            print(f"跳过 {dropped} 个超大晶胞 (nsites > {args.max_atoms})")

    # ── 3. 逐条把结构转成图 ──
    builder = GraphBuilder(args.cutoff, args.max_neighbors, args.rbf_bins)
    graphs: dict = {}
    index_rows = []
    failed = 0
    t0 = time.time()

    for row in tqdm(labels.itertuples(index=False), total=len(labels), desc="graphs"):
        mid = getattr(row, "material_id")
        structure = structures.get(mid)
        if structure is None:                     # 找不到结构就跳过
            failed += 1
            continue
        # 收集这条数据的所有标签（主目标 + 附带目标）
        targets = {args.target: getattr(row, args.target, None)}
        for name in args.extra_targets:
            targets[name] = getattr(row, name, None)
        try:
            data = builder.build(structure, targets, target_name=args.target)
        except Exception:                         # 个别结构转换失败不影响整体
            failed += 1
            continue
        graphs[mid] = data
        # 同时记一行索引，方便人类查看每个图的大小和标签
        index_rows.append({
            "material_id": mid,
            "formula": getattr(row, "formula", None),
            "n_atoms": data.n_atoms,
            "n_edges": data.n_edges,
            args.target: getattr(row, args.target, None),
        })

    # ── 4. 保存结果：图数据 + 索引表 ──
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "cutoff": args.cutoff,
        "max_neighbors": args.max_neighbors,
        "rbf_bins": args.rbf_bins,
        "edge_dim": builder.edge_dim,
        "target": args.target,
        "extra_targets": args.extra_targets,
        "n_graphs": len(graphs),
        "format": "PyG Data: x(long atomic number), edge_index, edge_attr(RBF)",
    }
    with open(out, "wb") as f:
        pickle.dump({"graphs": graphs, "meta": meta}, f)

    index_csv = out.parent / "graph_index.csv"
    pd.DataFrame(index_rows).to_csv(index_csv, index=False, encoding="utf-8-sig")

    # ── 5. 打印统计信息 ──
    size_mb = out.stat().st_size / 1024 / 1024
    mean_atoms = sum(g.n_atoms for g in graphs.values()) / max(len(graphs), 1)
    mean_edges = sum(g.n_edges for g in graphs.values()) / max(len(graphs), 1)
    print(f"\n完成: {len(graphs)} 个图 (失败 {failed})  耗时 {time.time() - t0:.1f}s")
    print(f"文件: {out}  ({size_mb:.1f} MB)")
    print(f"索引: {index_csv}")
    print(f"平均原子数 {mean_atoms:.1f} | 平均边数 {mean_edges:.1f} | 边特征维度 {builder.edge_dim}")

    # ── 6. 自检：随便拿两个图拼批，确认模型能直接吃 ──
    if len(graphs) >= 2:
        from torch_geometric.loader import DataLoader
        sample = list(graphs.values())[:2]
        batch = next(iter(DataLoader(sample, batch_size=2)))
        y_shape = tuple(batch.y.shape) if batch.y is not None else "无"
        print(f"自检: batch.x {tuple(batch.x.shape)}, "
              f"batch.edge_index {tuple(batch.edge_index.shape)}, "
              f"batch.edge_attr {tuple(batch.edge_attr.shape)}, "
              f"batch.y {y_shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
