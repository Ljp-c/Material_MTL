"""数据接口：分片流式读取、统一标签层、按组成分组划分、多源批采样。

对接 specific_MT/graph/<数据集>/：
    crystal_graph_partNNNNN.pkl   {material_id: CrystalData(x, z, edge_index, edge_attr, ...)}
    line_graph_partNNNNN.pkl      {material_id: {line_edge_index, line_edge_attr, n_edges}}
    index.csv                     每行一个 material_id（group 列 + 全部标签列）
    global_feat.csv               8 维全局特征（a,b,c,α,β,γ,体积/原子,密度；旁挂，见 build_global_feat.py）

图级四目标顺序固定为 GRAPH_TARGETS；空位形成能为位点级 vacancy [N,1]；
全局特征按 label_stats.json 的 global_feat 统计标准化（缺失样本按均值填充 → 标准化后为 0）。
"""
from __future__ import annotations

import hashlib
import math
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Composition
from torch_geometric.data import Batch

from common import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
GRAPH_ROOT = REPO_ROOT / "specific_MT" / "graph"
DATA_PREP = REPO_ROOT / "specific_MT" / "data_prep"
if str(DATA_PREP) not in sys.path:
    sys.path.insert(0, str(DATA_PREP))

from graph_schema import CrystalData

GRAPH_TARGETS = ("formation", "gap", "cbm", "vbm")
SLOT_INDEX = {name: index for index, name in enumerate(GRAPH_TARGETS)}
GLOBAL_FEATURE_COLUMNS = ("a", "b", "c", "alpha", "beta", "gamma", "volume_per_atom", "density")
GLOBAL_DIM = len(GLOBAL_FEATURE_COLUMNS)

SOURCE_REGISTRY = {
    "formation_energy_band_gap": {
        "dir": GRAPH_ROOT / "formation_energy_band_gap",
        "labels": {
            "formation": "formation_energy_per_atom_eV",
            "gap": "band_gap_eV",
            "cbm": "cbm_eV",
            "vbm": "vbm_eV",
        },
        "vacancy": False,
    },
    "batio3": {
        "dir": GRAPH_ROOT / "batio3",
        "labels": {"formation": "formation_energy_per_atom_eV", "gap": "band_gap_eV"},
        "vacancy": False,
    },
    "batio3_doped": {
        "dir": GRAPH_ROOT / "batio3_doped",
        "labels": {"formation": "formation_energy_per_atom"},
        "vacancy": False,
    },
    "vacancy_screening": {
        "dir": GRAPH_ROOT / "vacancy_screening",
        "labels": {},
        "vacancy": True,
    },
}


def seed_for(text: str, salt: str = "") -> int:
    digest = hashlib.md5(f"{text}|{salt}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def normalize_group(text) -> str | None:
    if text is None:
        return None
    text = str(text)
    try:
        return Composition(text).reduced_formula
    except Exception:
        return text


def formula_elements(text) -> set[str]:
    return set(re.findall(r"[A-Z][a-z]?", str(text)))


A_SITE_ELEMENTS = ("Ba", "Sr", "Ca", "Pb", "La", "Bi", "K", "Na", "Nd", "Sm", "Y", "Pr", "Eu", "Gd")
B_SITE_ELEMENTS = ("Ti", "Zr", "Sn", "Hf", "Nb", "Fe", "Mn", "Mg", "Ta", "Mo", "W",
                   "Al", "Ga", "In", "Cr", "Co", "Ni", "Zn")


def is_batio3_based(text) -> bool:
    try:
        amounts = Composition(str(text)).get_el_amt_dict()
    except Exception:
        return False
    n_oxygen = amounts.get("O", 0.0)
    a_site = sum(amounts.get(element, 0.0) for element in A_SITE_ELEMENTS)
    b_site = sum(amounts.get(element, 0.0) for element in B_SITE_ELEMENTS)
    if a_site <= 0 or b_site <= 0:
        return False
    if abs(n_oxygen - 3.0 * a_site) > 1e-6 or abs(a_site - b_site) > 1e-6:
        return False
    ba = amounts.get("Ba", 0.0)
    ti = amounts.get("Ti", 0.0)
    return ba / a_site >= 0.5 and ti / b_site >= 0.5


def load_family_groups(config_path=None) -> set[str]:
    if config_path is None:
        config_path = REPO_ROOT / "Models" / "configs" / "finetune.yaml"
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"找不到微调配置 {config_path}（家族剔除依赖它）")
    cfg = load_config(config_path)
    common_filters = cfg.get("filters") or {}
    family: set[str] = set()
    for entry in cfg.get("sources") or []:
        name = entry["name"]
        spec = SOURCE_REGISTRY.get(name)
        if spec is None:
            continue
        path = Path(spec["dir"]) / "index.csv"
        if not path.exists():
            continue
        filters_cfg = dict(common_filters)
        filters_cfg.update(entry.get("filters") or {})
        sample_filter = SampleFilter(
            min_atoms=int(filters_cfg.get("min_atoms") or 0),
            max_atoms=filters_cfg.get("max_atoms"),
            contains_elements=tuple(filters_cfg.get("contains_elements") or ()),
            perovskite_batio3=bool(filters_cfg.get("perovskite_batio3", False)),
            exclude_formulas=tuple(filters_cfg.get("exclude_formulas") or ()),
        )
        frame = pd.read_csv(path, usecols=lambda column: column in {"group", "formula", "n_atoms"})
        for record in frame.to_dict("records"):
            raw_group = record.get("group")
            raw_formula = record.get("formula")
            group_text = raw_group if isinstance(raw_group, str) and raw_group.strip() else None
            formula_text = raw_formula if isinstance(raw_formula, str) and raw_formula.strip() else None
            text = formula_text or group_text
            group_text = group_text or formula_text
            if text is None:
                continue
            raw_atoms = record.get("n_atoms")
            n_atoms = 0 if raw_atoms is None or pd.isna(raw_atoms) else int(raw_atoms)
            if not sample_filter.keep(n_atoms, text):
                continue
            normalized = normalize_group(group_text)
            if normalized:
                family.add(normalized)
    return family


@dataclass(frozen=True)
class SampleFilter:
    min_atoms: int = 0
    max_atoms: int | None = None
    contains_elements: tuple[str, ...] = ()
    perovskite_batio3: bool = False
    exclude_formulas: tuple[str, ...] = ()

    def __post_init__(self):
        if self.exclude_formulas:
            normalized = {normalize_group(value) for value in self.exclude_formulas}
            object.__setattr__(self, "exclude_formulas",
                               tuple(sorted(value for value in normalized if value)))

    def keep(self, n_atoms: int, formula_text: str) -> bool:
        if n_atoms and n_atoms < self.min_atoms:
            return False
        if self.max_atoms is not None and n_atoms and n_atoms > self.max_atoms:
            return False
        if self.contains_elements:
            elements = formula_elements(formula_text)
            if not all(element in elements for element in self.contains_elements):
                return False
        if self.exclude_formulas and normalize_group(formula_text) in self.exclude_formulas:
            return False
        if self.perovskite_batio3 and not is_batio3_based(formula_text):
            return False
        return True


class Source:
    def __init__(self, name: str, split_cfg: dict, filters: SampleFilter | None = None,
                 exclude_groups: set[str] | None = None, limit_shards: int | None = None,
                 global_stats: dict | None = None, require_global: bool = False):
        if name not in SOURCE_REGISTRY:
            raise KeyError(f"未知数据源 {name}；可用: {', '.join(SOURCE_REGISTRY)}")
        spec = SOURCE_REGISTRY[name]
        self.name = name
        self.dir = Path(spec["dir"])
        self.labels = dict(spec["labels"])
        self.has_vacancy = bool(spec["vacancy"])
        self.split_cfg = dict(split_cfg)

        index_path = self.dir / "index.csv"
        if not index_path.exists():
            raise FileNotFoundError(f"{name}: 缺少 index.csv（{index_path}）")
        table = pd.read_csv(index_path)
        if "material_id" not in table.columns:
            raise ValueError(f"{name}: index.csv 缺少 material_id 列")
        table["material_id"] = table["material_id"].astype(str)
        self.table = table.set_index("material_id", drop=False)

        info: dict[str, dict] = {}
        group_values: list[str] = []
        for row in table.to_dict("records"):
            mid = str(row["material_id"])
            raw_group = row.get("group")
            raw_formula = row.get("formula")
            group = raw_group if isinstance(raw_group, str) and raw_group.strip() else None
            formula = raw_formula if isinstance(raw_formula, str) and raw_formula.strip() else None
            group = group or formula or mid
            formula = formula or group
            raw_atoms = row.get("n_atoms")
            n_atoms = 0 if raw_atoms is None or pd.isna(raw_atoms) else int(raw_atoms)
            info[mid] = {"group": group, "formula": formula, "n_atoms": n_atoms}
            group_values.append(group)
        self.records = info

        self.global_feat: dict[str, np.ndarray] = {}
        feature_path = self.dir / "global_feat.csv"
        if feature_path.exists():
            frame = pd.read_csv(feature_path)
            missing_columns = [column for column in GLOBAL_FEATURE_COLUMNS if column not in frame.columns]
            if missing_columns:
                print(f"  警告: {feature_path} 缺列 {missing_columns}，忽略全局特征")
            else:
                mids = frame["material_id"].astype(str).to_numpy()
                values = frame[list(GLOBAL_FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
                self.global_feat = {mid: values[index] for index, mid in enumerate(mids)}
        self.global_stats = global_stats
        if global_stats is not None:
            self.global_mean = np.asarray(global_stats["mean"], dtype=np.float32)
            self.global_std = np.clip(np.asarray(global_stats["std"], dtype=np.float32), 1e-6, None)
        else:
            self.global_mean = None
            self.global_std = None
        self.n_global_missing = sum(1 for mid in info if mid not in self.global_feat)
        if require_global and not self.global_feat:
            raise FileNotFoundError(
                f"{name}: 缺少 {feature_path}；先运行 python specific_MT\\data_prep\\build_global_feat.py")

        shards = sorted(self.dir.glob("crystal_graph_part*.pkl"))
        if limit_shards:
            shards = shards[: int(limit_shards)]
        self.crystal_shards = shards
        self.n_shards = len(shards)
        self.line_paths = {
            path: path.with_name(path.name.replace("crystal_graph", "line_graph", 1))
            for path in shards
        }

        groups = sorted(set(group_values))
        rng = np.random.default_rng(seed_for(name, f"split-{self.split_cfg.get('seed', 0)}"))
        permutation = rng.permutation(len(groups))
        n_val = int(round(float(self.split_cfg.get("val_frac", 0.0)) * len(groups)))
        n_test = int(round(float(self.split_cfg.get("test_frac", 0.0)) * len(groups)))
        val_groups = {groups[i] for i in permutation[:n_val]}
        test_groups = {groups[i] for i in permutation[n_val:n_val + n_test]}

        family = set(exclude_groups or ())
        self.n_family = 0
        self.n_filtered = 0
        self.split: dict[str, str] = {}
        normalized_cache: dict[str, str | None] = {}
        for mid, entry in info.items():
            group_text = entry["group"]
            if group_text in normalized_cache:
                normalized = normalized_cache[group_text]
            else:
                normalized = normalize_group(group_text)
                normalized_cache[group_text] = normalized
            if family and normalized in family:
                self.split[mid] = "excluded"
                self.n_family += 1
                continue
            if filters is not None and not filters.keep(entry["n_atoms"], entry["formula"]):
                self.split[mid] = "excluded"
                self.n_filtered += 1
                continue
            if entry["group"] in val_groups:
                self.split[mid] = "val"
            elif entry["group"] in test_groups:
                self.split[mid] = "test"
            else:
                self.split[mid] = "train"

        counts = {"train": 0, "val": 0, "test": 0, "excluded": 0}
        for value in self.split.values():
            counts[value] += 1
        self.counts = counts

    def count(self, split: str) -> int:
        return self.counts.get(split, 0)

    def stream(self, split: str, epoch: int = 0, sub: int = 0, shuffle: bool = True):
        rng = np.random.default_rng(seed_for(self.name, f"stream-{split}-{epoch}-{sub}"))
        order = list(range(len(self.crystal_shards)))
        if shuffle:
            rng.shuffle(order)
        for index in order:
            crystal_path = self.crystal_shards[index]
            with open(crystal_path, "rb") as fh:
                crystal = pickle.load(fh)
            with open(self.line_paths[crystal_path], "rb") as fh:
                line = pickle.load(fh)
            mids = [mid for mid in crystal.keys() if self.split.get(mid) == split]
            if shuffle:
                rng.shuffle(mids)
            for mid in mids:
                yield prepare_sample(crystal[mid], line.get(mid), self, self.records.get(mid), mid)
            del crystal, line


def prepare_sample(data, line, source: Source, record: dict | None, mid: str):
    n_atoms = int(data.x.size(0))
    if line is None:
        line_index = torch.zeros((2, 0), dtype=torch.long)
        line_attr = torch.zeros((0, 1), dtype=torch.float32)
    else:
        line_index = line["line_edge_index"]
        line_attr = line["line_edge_attr"]
        if line_attr.dim() != 2 or line_attr.size(1) != 1:
            raise RuntimeError(
                f"{source.name}/{mid}: line_edge_attr 形状 {tuple(line_attr.shape)}，期望 [L,1]；"
                "请用 --line-mode angle 重建图数据")
        n_edges = int(line.get("n_edges", data.edge_index.size(1)))
        if n_edges != int(data.edge_index.size(1)):
            raise RuntimeError(
                f"{source.name}/{mid}: 晶体图边数 {int(data.edge_index.size(1))} 与线图 n_edges {n_edges} 不一致")

    data.line_edge_index = line_index
    data.line_edge_attr = line_attr

    labels = torch.full((1, len(GRAPH_TARGETS)), float("nan"), dtype=torch.float32)
    for slot, column in source.labels.items():
        value = getattr(data, column, None)
        if value is None:
            continue
        number = float(value.reshape(-1)[0]) if torch.is_tensor(value) else float(value)
        if math.isfinite(number):
            labels[0, SLOT_INDEX[slot]] = number
    data.labels = labels

    if source.has_vacancy:
        vacancy = getattr(data, "vacancy", None)
        if vacancy is None:
            vacancy = torch.full((n_atoms, 1), float("nan"), dtype=torch.float32)
        else:
            vacancy = vacancy.reshape(-1, 1).float()
            if vacancy.size(0) != n_atoms:
                raise RuntimeError(f"{source.name}/{mid}: vacancy 长度 {vacancy.size(0)} != 原子数 {n_atoms}")
    else:
        vacancy = torch.full((n_atoms, 1), float("nan"), dtype=torch.float32)
    data.vacancy = vacancy

    if source.global_mean is None:
        features = np.zeros(GLOBAL_DIM, dtype=np.float32)
    else:
        raw_features = source.global_feat.get(mid) if source.global_feat else None
        if raw_features is None:
            features = np.zeros(GLOBAL_DIM, dtype=np.float32)
        else:
            features = (np.asarray(raw_features, dtype=np.float32) - source.global_mean) / source.global_std
    data.global_feat = torch.from_numpy(features.astype(np.float32)).view(1, -1)

    data.name = str(mid)
    existing_group = getattr(data, "split_group", None)
    if not isinstance(existing_group, str) or not existing_group.strip() or existing_group == "nan":
        data.split_group = str(record["group"]) if record else str(mid)

    keep = {"x", "z", "edge_index", "edge_attr", "line_edge_index", "line_edge_attr",
            "labels", "vacancy", "global_feat", "name", "split_group", "n_atoms"}
    for key in list(data.keys()):
        if key not in keep:
            delattr(data, key)
    return data


def collate(items):
    return Batch.from_data_list(list(items))


def allocate_counts(probabilities: np.ndarray, steps: int) -> list[int]:
    raw = np.asarray(probabilities, dtype=np.float64) * steps
    base = np.floor(raw).astype(int)
    remainder = raw - base
    short = steps - int(base.sum())
    if short > 0:
        order = np.argsort(-remainder)[:short]
        base[order] += 1
    return base.tolist()


class MultiSourceBatcher:
    def __init__(self, sources, batch_size: int, weights: dict | None = None,
                 sampling: str = "sqrt_inv", seed: int = 0):
        self.batch_size = int(batch_size)
        self.sampling = sampling
        self.seed = int(seed)
        self.sources = {source.name: source for source in sources}
        self.names = [source.name for source in sources]
        self.sizes = {source.name: source.count("train") for source in sources}
        weights = weights or {}
        raw: dict[str, float] = {}
        for name in self.names:
            size = self.sizes[name]
            if size <= 0:
                continue
            weight = float(weights.get(name, 1.0))
            if sampling == "sqrt_inv":
                factor = 1.0 / math.sqrt(size)
            elif sampling == "proportional":
                factor = float(size)
            else:
                raise ValueError(f"未知采样模式 {sampling}")
            raw[name] = weight * factor
        total = sum(raw.values())
        self.prob = {name: value / total for name, value in raw.items()} if total > 0 else {}
        self.steps_per_epoch = max(1, math.ceil(sum(self.sizes.values()) / self.batch_size)) if self.sizes else 0

    def _make_gen(self, name: str, epoch: int, sub: int):
        return self.sources[name].stream("train", epoch=epoch, sub=sub)

    def epoch_batches(self, epoch: int):
        names = [name for name in self.names if self.sizes.get(name, 0) > 0]
        if not names:
            return
        rng = np.random.default_rng(seed_for("batcher", f"{self.seed}-{epoch}"))
        probabilities = np.array([self.prob[name] for name in names], dtype=np.float64)
        probabilities = probabilities / probabilities.sum()
        steps = max(1, math.ceil(sum(self.sizes[name] for name in names) / self.batch_size))
        counts = allocate_counts(probabilities, steps)
        sequence = [name for name, count in zip(names, counts) for _ in range(count)]
        rng.shuffle(sequence)
        generators = {name: self._make_gen(name, epoch, 0) for name in names}
        subs = {name: 0 for name in names}
        buffers: dict[str, list] = {name: [] for name in names}
        for name in sequence:
            buffer = buffers[name]
            while len(buffer) < self.batch_size:
                try:
                    buffer.append(next(generators[name]))
                except StopIteration:
                    subs[name] += 1
                    generators[name] = self._make_gen(name, epoch, subs[name])
            yield name, buffer[: self.batch_size]
            buffers[name] = buffer[self.batch_size:]

    def eval_batches(self, split: str, max_batches: int | None = None, batch_size: int | None = None):
        size = int(batch_size or self.batch_size)
        for name in self.names:
            source = self.sources[name]
            if source.count(split) <= 0:
                continue
            buffer = []
            produced = 0
            for data in source.stream(split, epoch=0, sub=0, shuffle=False):
                buffer.append(data)
                if len(buffer) == size:
                    yield name, buffer
                    buffer = []
                    produced += 1
                    if max_batches and produced >= int(max_batches):
                        break
            if buffer:
                yield name, buffer
