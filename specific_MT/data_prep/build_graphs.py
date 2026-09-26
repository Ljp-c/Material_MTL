"""把晶体结构 pkl 转成图数据（PyG Data），供图神经网络训练使用。

节点 = 原子
    节点特征 x（float, [N, 149]）：
        [0:118)   元素 one-hot（原子序数 Z-1，固定 118 维，跨阶段不用对齐词表）
        [118:125) 周期 one-hot（1-7）
        [125:144) 族 one-hot（1-18；镧系/锕系/未知 -> 第 19 维）
        [144]     电负性（Pauling 标尺）
        [145]     平均离子半径 / Angstrom
        [146]     价电子数（最外层 s/p + 次外层 d + 次次外层 f）
        [147]     氧化态（BVA 推断，标量；失败 -> 缺失）
        [148]     氧化态有效掩码（0/1，不参与标准化）
    标量列 [144:148] 必须用预训练集统计的全局 mean/std 标准化，缺失位点归零。

边 = 截断半径（默认 8 A）内的近邻原子对（邻居 -> 中心），每原子最多保留最近 12 个
    邻居搜索是周期性的、以每个原子为中心；原点选在晶胞中心还是别处不影响图，
    因为距离与键角都是平移不变量。
    边特征 edge_attr = 距离 RBF（默认 48 维）
    键角线图：键 = 线图节点（即原图边的索引）；共享同一中心原子（同一 dst）
              的两条键 = 一条线图边，特征 = 夹角 RBF（默认 16 维，0-180 度）
        line_edge_index [2, L]、line_edge_attr [L, angle_bins]

标签 = e_total（默认目标），并附带 e_ionic / e_electronic

图对象是 graph_schema.CrystalData：它保证 line_edge_index 在多图 batch 时按 num_edges
偏移（PyG 默认对 *_index 键按 num_nodes 偏移，会把线图边指到错误边上）。
下游加载 graphs.pkl 的脚本需要能 import graph_schema（把 data_prep 目录加入 sys.path）。

路径（可以只指定文件夹）：
    --data-dir   输入数据文件夹；默认 <脚本上级目录>/data/full_db_dielectric
    --data-name  数据文件名；默认优先 dielectric_dataset.pkl，否则在文件夹里自动探测
    --out-dir    图输出文件夹；默认与 --data-dir 相同（不存在会自动创建）
    --out-name   图输出文件名；默认 graphs.pkl，索引 csv 同名派生（graphs.pkl -> graphs_index.csv）

标量特征的标准化口径（两阶段必须共用同一套统计）：
    预训练集:  python build_graphs.py --fit-stats          # 统计 + 建图
    微调集:    python build_graphs.py --data-dir <微调数据文件夹> \
                   --out-name graphs_finetune.pkl --stats <预训练的 stats.json>

用法：
    python build_graphs.py --fit-stats                     # 预训练：全量统计 + 建图
    python build_graphs.py --data-dir "D:\我的数据" --out-dir "D:\我的图" --fit-stats
    python build_graphs.py --limit 300 --fit-stats --out-name smoke_graphs.pkl
    python build_graphs.py --rbf-bins 64 --angle-bins 8
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from pymatgen.core import Element

from graph_schema import CrystalData

N_ELEMENTS = 118
N_PERIODS = 7
N_GROUPS = 19

ELEMENT_OFF = 0
PERIOD_OFF = ELEMENT_OFF + N_ELEMENTS
GROUP_OFF = PERIOD_OFF + N_PERIODS
EN_IDX = GROUP_OFF + N_GROUPS
IR_IDX = EN_IDX + 1
NV_IDX = IR_IDX + 1
OS_IDX = NV_IDX + 1
OS_MASK_IDX = OS_IDX + 1
NODE_DIM = OS_MASK_IDX + 1

SCALAR_COLUMNS = (
    ("electronegativity", EN_IDX),
    ("ionic_radius", IR_IDX),
    ("valence_electrons", NV_IDX),
    ("oxidation_state", OS_IDX),
)

X_LAYOUT = {
    "element_onehot": [ELEMENT_OFF, PERIOD_OFF],
    "period_onehot": [PERIOD_OFF, GROUP_OFF],
    "group_onehot": [GROUP_OFF, EN_IDX],
    "electronegativity": EN_IDX,
    "ionic_radius": IR_IDX,
    "valence_electrons": NV_IDX,
    "oxidation_state": OS_IDX,
    "oxidation_state_mask": OS_MASK_IDX,
}

ANGLE_MIN = 0.0
ANGLE_MAX = 180.0


def rbf_expand(values: torch.Tensor, centers: torch.Tensor, width: float) -> torch.Tensor:
    """把一维数值展开成多维高斯 RBF：数值靠近哪个刻度，哪一维就接近 1。"""
    d = values.view(-1, 1)
    return torch.exp(-((d - centers.view(1, -1)) ** 2) / (2.0 * width ** 2))


@lru_cache(maxsize=256)
def valence_electrons(symbol: str) -> float:
    """价电子数 = 最外层 s/p + 次外层 d + 次次外层 f 电子数。"""
    es = Element(symbol).full_electronic_structure
    n_max = max(n for n, _, _ in es)
    total = sum(occ for n, l, occ in es if n == n_max and l in ("s", "p"))
    total += sum(occ for n, l, occ in es if n == n_max - 1 and l == "d")
    total += sum(occ for n, l, occ in es if n == n_max - 2 and l == "f")
    return float(total)


@lru_cache(maxsize=256)
def element_features(symbol: str):
    """返回 (Z, 周期, 族号或 19, 电负性, 平均离子半径, 价电子数)。"""
    el = Element(symbol)
    z = int(el.Z)
    row = int(el.row) if el.row else 0
    if getattr(el, "is_lanthanoid", False) or getattr(el, "is_actinoid", False):
        group_idx = N_GROUPS
    else:
        group_idx = int(el.group) if el.group else N_GROUPS
    en = float(el.X)
    try:
        ir = float(el.average_ionic_radius)
        if not (ir > 0.0) or not math.isfinite(ir):
            ir = float("nan")
    except Exception:
        ir = float("nan")
    return z, row, group_idx, en, ir, valence_electrons(symbol)


_BVA = None


def _get_bva():
    global _BVA
    if _BVA is None:
        from pymatgen.analysis.bond_valence import BVAnalyzer
        _BVA = BVAnalyzer()
    return _BVA


def bva_oxidation_states(structure):
    """键价法逐位点氧化态；失败返回 None。"""
    try:
        decorated = _get_bva().get_oxi_state_decorated_structure(structure)
    except Exception:
        return None
    values = []
    for site in decorated:
        oxi = getattr(site.specie, "oxi_state", None)
        if oxi is None:
            return None
        values.append(float(oxi))
    return values


def guess_oxidation_states(structure):
    """组成法（oxi_state_guesses）逐元素氧化态；慢，仅供 --oxi-mode guess 使用。"""
    try:
        guesses = structure.composition.oxi_state_guesses()
    except Exception:
        return None
    if not guesses:
        return None
    table = guesses[0]
    values = []
    for site in structure:
        value = table.get(site.specie.symbol)
        if value is None:
            return None
        values.append(float(value))
    return values


class OxidationResolver:
    """逐结构推断氧化态并缓存；mode: bva / guess / none。"""

    def __init__(self, mode: str = "bva"):
        self.mode = mode
        self.cache: dict = {}
        self.n_bva_ok = 0
        self.n_guess_ok = 0
        self.n_missing = 0

    def get(self, mid: str, structure):
        if mid in self.cache:
            return self.cache[mid]
        values = None
        if self.mode != "none":
            values = bva_oxidation_states(structure)
            if values is not None:
                self.n_bva_ok += 1
            elif self.mode == "guess":
                values = guess_oxidation_states(structure)
                if values is not None:
                    self.n_guess_ok += 1
        if values is None:
            self.n_missing += 1
        self.cache[mid] = values
        return values

    def report(self) -> str:
        return (f"氧化态: BVA 成功 {self.n_bva_ok} | guess 成功 {self.n_guess_ok} | "
                f"缺失 {self.n_missing}")


class GraphBuilder:
    """负责把单个 pymatgen Structure 转成一个 PyG Data（图 + 键角线图）。"""

    def __init__(self, cutoff: float = 8.0, max_neighbors: int = 12, rbf_bins: int = 48,
                 angle_bins: int = 16, line_encoding: str = "rbf"):
        self.cutoff = float(cutoff)              # 近邻截断半径（埃）
        self.max_neighbors = int(max_neighbors)  # 每个原子最多保留几个邻居
        self.rbf_bins = int(rbf_bins)            # 距离展开成多少维
        self.angle_bins = int(angle_bins)        # 键角展开成多少维
        self.line_encoding = line_encoding       # 线图边特征: rbf(角度 RBF) / angle(角度标量)
        self.centers = torch.linspace(0.0, self.cutoff, self.rbf_bins)
        self.width = self.cutoff / max(self.rbf_bins - 1, 1)
        self.angle_centers = torch.linspace(ANGLE_MIN, ANGLE_MAX, self.angle_bins)
        self.angle_width = (ANGLE_MAX - ANGLE_MIN) / max(self.angle_bins - 1, 1)

    @property
    def node_dim(self) -> int:
        return NODE_DIM

    @property
    def edge_dim(self) -> int:
        return self.rbf_bins

    @property
    def line_dim(self) -> int:
        return 1 if self.line_encoding == "angle" else self.angle_bins

    def build(self, structure, targets: dict, target_name: str | None = None, oxi_states=None) -> CrystalData:
        n_sites = len(structure)
        x = np.zeros((n_sites, NODE_DIM), dtype=np.float32)
        z_list = []
        for i, site in enumerate(structure):
            z, row, group_idx, en, ir, nv = element_features(site.specie.symbol)
            z_list.append(z)
            if 1 <= z <= N_ELEMENTS:
                x[i, ELEMENT_OFF + z - 1] = 1.0
            if 1 <= row <= N_PERIODS:
                x[i, PERIOD_OFF + row - 1] = 1.0
            if 1 <= group_idx <= N_GROUPS:
                x[i, GROUP_OFF + group_idx - 1] = 1.0
            x[i, EN_IDX] = en
            x[i, IR_IDX] = ir
            x[i, NV_IDX] = nv
            value = float("nan")
            if oxi_states is not None and i < len(oxi_states):
                value = float(oxi_states[i])
            if math.isfinite(value):
                x[i, OS_IDX] = value
                x[i, OS_MASK_IDX] = 1.0
            else:
                x[i, OS_IDX] = float("nan")
                x[i, OS_MASK_IDX] = 0.0

        neighbors = structure.get_all_neighbors(self.cutoff, include_index=True)
        cart = structure.cart_coords
        src, dst, dist = [], [], []
        src_pos, dst_pos = [], []
        for i, nlist in enumerate(neighbors):
            nlist = sorted(nlist, key=lambda nb: nb.nn_distance)[: self.max_neighbors]
            for nb in nlist:
                src.append(int(nb.index))
                dst.append(i)
                dist.append(float(nb.nn_distance))
                src_pos.append(nb.coords)
                dst_pos.append(cart[i])

        if dist:
            edge_index = torch.tensor([src, dst], dtype=torch.long)
            d = torch.tensor(dist, dtype=torch.float32)
            edge_attr = rbf_expand(d, self.centers, self.width)
            src_t = torch.tensor(np.asarray(src_pos), dtype=torch.float32)
            dst_t = torch.tensor(np.asarray(dst_pos), dtype=torch.float32)
            line_index, line_attr = self._line_graph(edge_index, src_t, dst_t)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, self.rbf_bins), dtype=torch.float32)
            line_index = torch.zeros((2, 0), dtype=torch.long)
            line_attr = torch.zeros((0, self.line_dim), dtype=torch.float32)

        data = CrystalData(x=torch.from_numpy(x), edge_index=edge_index, edge_attr=edge_attr,
                           line_edge_index=line_index, line_edge_attr=line_attr)
        data.z = torch.tensor(z_list, dtype=torch.long)

        for name, value in targets.items():
            if value is None or value != value:
                continue
            t = torch.tensor([float(value)], dtype=torch.float32)
            data[name] = t
            if name == target_name:
                data.y = t

        data.n_atoms = int(n_sites)
        data.n_edges = int(edge_index.shape[1])
        data.n_line_edges = int(line_index.shape[1])
        return data

    def _line_graph(self, edge_index: torch.Tensor, src_pos: torch.Tensor, dst_pos: torch.Tensor):
        """以"共享同一 dst（中心原子）"的两条键成线图边，特征为两条键的夹角。"""
        n_edges = edge_index.shape[1]
        empty_index = torch.zeros((2, 0), dtype=torch.long)
        empty_attr = torch.zeros((0, self.line_dim), dtype=torch.float32)
        if n_edges == 0:
            return empty_index, empty_attr

        src_list = edge_index[0].tolist()
        groups = defaultdict(list)
        for e, center in enumerate(edge_index[1].tolist()):
            groups[center].append(e)

        pairs, feats = [], []
        for center, eids in groups.items():
            if len(eids) < 2:
                continue
            vec = src_pos[eids] - dst_pos[eids[0]]
            norm = vec.norm(dim=1).clamp_min(1e-6)
            cos = (vec @ vec.t()) / (norm[:, None] * norm[None, :])
            angle = torch.acos(cos.clamp(-1.0, 1.0)) * (180.0 / math.pi)
            for a in range(len(eids)):
                for b in range(a + 1, len(eids)):
                    if src_list[eids[a]] == src_list[eids[b]]:
                        continue
                    if self.line_encoding == "angle":
                        f = angle[a, b].reshape(1, 1)
                    else:
                        f = rbf_expand(angle[a, b].reshape(1), self.angle_centers, self.angle_width)
                    pairs.append([eids[a], eids[b]])
                    pairs.append([eids[b], eids[a]])
                    feats.append(f)
                    feats.append(f)

        if not pairs:
            return empty_index, empty_attr
        line_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()
        line_attr = torch.cat(feats, dim=0)
        return line_index, line_attr


def collect_scalars(structure, oxi_states):
    """取单个结构的 4 个标量特征原始值（缺失为 NaN）。"""
    n_sites = len(structure)
    out = {name: np.full(n_sites, np.nan, dtype=np.float64) for name, _ in SCALAR_COLUMNS}
    for i, site in enumerate(structure):
        _, _, _, en, ir, nv = element_features(site.specie.symbol)
        out["electronegativity"][i] = en
        out["ionic_radius"][i] = ir
        out["valence_electrons"][i] = nv
        if oxi_states is not None and i < len(oxi_states):
            value = float(oxi_states[i])
            if math.isfinite(value):
                out["oxidation_state"][i] = value
    return out


def compute_stats(chunks: dict, data_path: str, n_structures: int) -> dict:
    """按原子池化统计各标量特征的全局 mean/std（忽略缺失）。"""
    features = {}
    n_atoms = 0
    for name, _ in SCALAR_COLUMNS:
        arr = np.concatenate(chunks[name]) if chunks[name] else np.array([])
        n_atoms = max(n_atoms, int(arr.size))
        valid = arr[np.isfinite(arr)]
        mean = float(valid.mean()) if valid.size else 0.0
        std = float(valid.std()) if valid.size else 1.0
        features[name] = {
            "mean": mean,
            "std": max(std, 1e-8),
            "fill": mean,
            "n_valid": int(valid.size),
            "n_total": int(arr.size),
            "missing_ratio": float(1.0 - valid.size / arr.size) if arr.size else 0.0,
        }
    return {
        "version": 1,
        "data": str(data_path),
        "n_structures": int(n_structures),
        "n_atoms": int(n_atoms),
        "features": features,
    }


def apply_stats(graphs: dict, stats: dict) -> None:
    """用给定统计量就地标准化标量列，缺失位点填 mean 后归零。"""
    feats = stats["features"]
    for data in graphs.values():
        x = data.x
        for name, col in SCALAR_COLUMNS:
            mean = float(feats[name]["mean"])
            std = float(feats[name]["std"])
            v = torch.nan_to_num(x[:, col], nan=mean, posinf=mean, neginf=mean)
            x[:, col] = (v - mean) / std


def resolve_data_file(data_dir: Path, data_name: str | None) -> Path:
    """在数据文件夹里定位 {labels, structures} 格式的数据 pkl。"""
    if not data_dir.is_dir():
        raise SystemExit(f"数据文件夹不存在: {data_dir}")
    if data_name:
        candidate = data_dir / data_name
        if candidate.is_file():
            return candidate
        available = ", ".join(sorted(p.name for p in data_dir.glob("*.pkl"))) or "无"
        raise SystemExit(f"找不到 {candidate}；{data_dir} 里的 pkl: {available}")
    preferred = data_dir / "dielectric_dataset.pkl"
    if preferred.is_file():
        return preferred
    valid, invalid = [], []
    for candidate in sorted(data_dir.glob("*.pkl")):
        try:
            with open(candidate, "rb") as f:
                obj = pickle.load(f)
        except Exception:
            invalid.append(candidate.name)
            continue
        if isinstance(obj, dict) and "labels" in obj and "structures" in obj:
            valid.append(candidate)
        else:
            invalid.append(candidate.name)
    if len(valid) == 1:
        return valid[0]
    if not valid:
        checked = ", ".join(invalid) or "空目录"
        raise SystemExit(f"{data_dir} 里没有含 labels/structures 的数据 pkl（检查过: {checked}）；"
                         f"可用 --data-name 指定文件名")
    names = ", ".join(p.name for p in valid)
    raise SystemExit(f"{data_dir} 里有多个候选数据 pkl: {names}；请用 --data-name 指定")


def main() -> int:
    parser = argparse.ArgumentParser(description="晶体结构 -> 图数据(PyG Data) + 键角线图")
    here = Path(__file__).resolve().parents[1]
    parser.add_argument("--data-dir", default=str(here / "data" / "full_db_dielectric"),
                        help="输入数据所在文件夹（默认含 dielectric_dataset.pkl）")
    parser.add_argument("--data-name", default=None,
                        help="数据文件名；默认优先 dielectric_dataset.pkl，否则在文件夹里自动探测")
    parser.add_argument("--out-dir", default=None,
                        help="图输出文件夹；默认与 --data-dir 相同")
    parser.add_argument("--out-name", default="graphs.pkl", help="图输出文件名")
    parser.add_argument("--stats", default=str(here / "data" / "graph_feature_stats.json"),
                        help="标量特征统计 json（预训练集统计，微调集复用同一份）")
    parser.add_argument("--fit-stats", action="store_true",
                        help="在当前数据上重新统计 mean/std 写入 --stats")
    parser.add_argument("--force", action="store_true",
                        help="允许 --fit-stats 覆盖已存在的 stats 文件")
    parser.add_argument("--cutoff", type=float, default=8.0, help="近邻截断半径 (A)")
    parser.add_argument("--max-neighbors", type=int, default=12, help="每个原子的最大近邻数")
    parser.add_argument("--rbf-bins", type=int, default=48, help="距离 RBF 展开维度 (8 A 下 48 维约 0.17 A/格)")
    parser.add_argument("--angle-bins", type=int, default=16, help="键角 RBF 展开维度")
    parser.add_argument("--oxi-mode", choices=["bva", "guess", "none"], default="bva",
                        help="氧化态推断方式；guess 很慢，仅小数据集用")
    parser.add_argument("--target", default="e_total", help="作为标签的目标列")
    parser.add_argument("--extra-targets", nargs="*", default=["e_ionic", "e_electronic"])
    parser.add_argument("--max-atoms", type=int, default=200, help="跳过的超大晶胞阈值")
    parser.add_argument("--limit", type=int, default=None, help="限制条数（冒烟测试用）")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_file = resolve_data_file(data_dir, args.data_name)
    out_dir = Path(args.out_dir) if args.out_dir else data_dir
    out = out_dir / args.out_name
    print(f"[data] {data_file}")
    print(f"[out ] {out}")

    with open(data_file, "rb") as f:
        dataset = pickle.load(f)
    labels = dataset["labels"]
    structures = dataset["structures"]

    if args.limit:
        labels = labels.head(int(args.limit)).reset_index(drop=True)

    if args.max_atoms:
        keep = labels["nsites"].astype(float) <= args.max_atoms
        dropped = int((~keep).sum())
        labels = labels[keep].reset_index(drop=True)
        if dropped:
            print(f"跳过 {dropped} 个超大晶胞 (nsites > {args.max_atoms})")

    resolver = OxidationResolver(args.oxi_mode)
    stats_path = Path(args.stats)

    if args.fit_stats:
        if stats_path.exists() and not args.force:
            print(f"{stats_path} 已存在。若确认要在 {data_file.name} 上重算（不要在微调集上重算），请加 --force")
            return 2
        if args.limit and args.limit < len(structures):
            print(f"警告: --fit-stats 与 --limit {args.limit} 同用，统计量只来自前 {args.limit} 条，不要用于正式预训练。")
        print(f"[stats] 在 {len(labels)} 条结构上统计标量特征 ...")
        t0 = time.time()
        chunks = {name: [] for name, _ in SCALAR_COLUMNS}
        for row in tqdm(labels.itertuples(index=False), total=len(labels), desc="stats"):
            mid = getattr(row, "material_id")
            structure = structures.get(mid)
            if structure is None:
                continue
            scalars = collect_scalars(structure, resolver.get(mid, structure))
            for name in chunks:
                chunks[name].append(scalars[name])
        stats = compute_stats(chunks, str(data_file), len(labels))
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"      {stats_path}")
        for name, _ in SCALAR_COLUMNS:
            s = stats["features"][name]
            print(f"      {name:20s} mean={s['mean']:8.4f} std={s['std']:8.4f} 缺失={s['missing_ratio'] * 100:.2f}%")
        print(f"      耗时 {time.time() - t0:.1f}s")
    else:
        if not stats_path.exists():
            print(f"找不到特征统计文件: {stats_path}")
            print("请先在预训练集上统计: python build_graphs.py --fit-stats")
            return 2
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        print(f"[stats] 复用 {stats_path} "
              f"(统计自 {stats.get('n_structures')} 条结构 / {stats.get('n_atoms')} 个原子)")

    builder = GraphBuilder(args.cutoff, args.max_neighbors, args.rbf_bins, args.angle_bins)
    graphs: dict = {}
    index_rows = []
    failed = 0
    fail_examples = []
    t0 = time.time()

    for row in tqdm(labels.itertuples(index=False), total=len(labels), desc="graphs"):
        mid = getattr(row, "material_id")
        structure = structures.get(mid)
        if structure is None:
            failed += 1
            continue
        targets = {args.target: getattr(row, args.target, None)}
        for name in args.extra_targets:
            targets[name] = getattr(row, name, None)
        try:
            data = builder.build(structure, targets, target_name=args.target,
                                 oxi_states=resolver.get(mid, structure))
        except Exception as exc:
            failed += 1
            if len(fail_examples) < 3:
                fail_examples.append(f"{mid}: {type(exc).__name__}: {exc}")
            continue
        graphs[mid] = data
        index_rows.append({
            "material_id": mid,
            "formula": getattr(row, "formula", None),
            "n_atoms": data.n_atoms,
            "n_edges": data.n_edges,
            "n_line_edges": data.n_line_edges,
            args.target: getattr(row, args.target, None),
        })

    apply_stats(graphs, stats)

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print(f"注意: 覆盖已有文件 {out}")
    meta = {
        "cutoff": args.cutoff,
        "max_neighbors": args.max_neighbors,
        "rbf_bins": args.rbf_bins,
        "angle_bins": args.angle_bins,
        "node_dim": builder.node_dim,
        "edge_dim": builder.edge_dim,
        "line_dim": builder.line_dim,
        "x_layout": X_LAYOUT,
        "target": args.target,
        "extra_targets": args.extra_targets,
        "n_graphs": len(graphs),
        "oxi_mode": args.oxi_mode,
        "stats_file": str(stats_path),
        "stats": stats,
        "normalized": True,
        "data_class": "graph_schema.CrystalData",
        "line_graph": "线图边 = 共享同一中心原子(dst)的两条键；特征 = 夹角 RBF(0-180 度)",
        "format": ("PyG Data: x(float node_dim), z(long Z), edge_index, edge_attr(distance RBF), "
                   "line_edge_index, line_edge_attr(angle RBF)"),
    }
    with open(out, "wb") as f:
        pickle.dump({"graphs": graphs, "meta": meta}, f)

    index_csv = out.with_name(out.stem + "_index.csv")
    pd.DataFrame(index_rows).to_csv(index_csv, index=False, encoding="utf-8-sig")

    size_mb = out.stat().st_size / 1024 / 1024
    n_graphs = max(len(graphs), 1)
    mean_atoms = sum(g.n_atoms for g in graphs.values()) / n_graphs
    mean_edges = sum(g.n_edges for g in graphs.values()) / n_graphs
    mean_lines = sum(g.n_line_edges for g in graphs.values()) / n_graphs
    print(f"\n完成: {len(graphs)} 个图 (失败 {failed})  耗时 {time.time() - t0:.1f}s")
    for msg in fail_examples:
        print(f"      失败示例: {msg}")
    print(f"文件: {out}  ({size_mb:.1f} MB)")
    print(f"索引: {index_csv}")
    print(f"平均原子数 {mean_atoms:.1f} | 平均边数 {mean_edges:.1f} | 平均线图边数 {mean_lines:.1f}")
    print(f"截断 {args.cutoff} A | 最大邻居 {args.max_neighbors} | "
          f"节点维度 {builder.node_dim} | 边维度 {builder.edge_dim} | 线图维度 {builder.line_dim}")
    print(resolver.report())

    if len(graphs) >= 2:
        from torch_geometric.loader import DataLoader
        sample = list(graphs.values())[:2]
        batch = next(iter(DataLoader(sample, batch_size=2)))
        y_shape = tuple(batch.y.shape) if batch.y is not None else "无"
        print(f"自检: batch.x {tuple(batch.x.shape)}, "
              f"batch.edge_index {tuple(batch.edge_index.shape)}, "
              f"batch.edge_attr {tuple(batch.edge_attr.shape)}, "
              f"batch.line_edge_index {tuple(batch.line_edge_index.shape)}, "
              f"batch.line_edge_attr {tuple(batch.line_edge_attr.shape)}, "
              f"batch.y {y_shape}")
        if torch.isnan(batch.x).any():
            print("警告: batch.x 中存在 NaN，检查标准化流程")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
