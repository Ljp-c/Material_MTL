r"""生成 8 维全局特征（a, b, c, α, β, γ, 体积/原子, 密度）→ graph/<数据集>/global_feat.csv。

对应 Models 的 工作思路.md §4.2.3 全局特征（池化向量 128 + 8 = 136）。
不重建图分片，仅旁挂 material_id -> 特征表；标准化由 Models/label_stats.py 统一统计。

用法（工作目录 E:\Material_MTL）:
    python specific_MT\data_prep\build_global_feat.py --jobs 8
    python specific_MT\data_prep\build_global_feat.py --only batio3 batio3_doped
    python specific_MT\data_prep\build_global_feat.py --only formation_energy_band_gap --limit 2000
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pymatgen.core import Element, Lattice, Structure
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
GRAPH_DIR = ROOT / "graph"
_JOBLIB_TMP = GRAPH_DIR / "_joblib_tmp"
_JOBLIB_TMP.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("JOBLIB_TEMP_FOLDER", str(_JOBLIB_TMP))

COLUMNS = ["a", "b", "c", "alpha", "beta", "gamma", "volume_per_atom", "density"]
AMU_TO_G = 1.66053906660
DATASETS = ("formation_energy_band_gap", "vacancy_screening", "batio3", "batio3_doped")


def lattice_features(lattice: Lattice, n_atoms: int, mass_amu: float) -> list[float]:
    a, b, c = lattice.abc
    alpha, beta, gamma = lattice.angles
    volume = lattice.volume
    density = mass_amu * AMU_TO_G / max(volume, 1e-8)
    return [a, b, c, alpha, beta, gamma, volume / max(n_atoms, 1), density]


def structure_features(structure: Structure) -> list[float]:
    mass = float(sum(site.species.weight for site in structure))
    return lattice_features(structure.lattice, len(structure), mass)


def _process_json_lines(chunk):
    rows = []
    for line in chunk:
        try:
            obj = json.loads(line)
            structure = Structure.from_dict(obj["structure"])
            rows.append((str(obj["material_id"]), structure_features(structure)))
        except Exception:
            continue
    return rows


def _process_cells(chunk):
    rows = []
    cache: dict[int, float] = {}
    for mid, numbers, cell in chunk:
        try:
            zs = np.frombuffer(numbers, dtype=np.int32)
            matrix = np.frombuffer(cell, dtype=np.float64).reshape(3, 3)
            lattice = Lattice(matrix)
            mass = 0.0
            for z in zs:
                zi = int(z)
                if zi not in cache:
                    cache[zi] = float(Element.from_Z(zi).atomic_mass)
                mass += cache[zi]
            rows.append((mid, lattice_features(lattice, len(zs), mass)))
        except Exception:
            continue
    return rows


def parallel_process(chunks, worker, jobs: int):
    rows = []
    stream = Parallel(n_jobs=int(jobs), return_as="generator")(delayed(worker)(chunk) for chunk in chunks)
    for piece in tqdm(stream, desc="global_feat", unit="chunk"):
        rows.extend(piece)
    return rows


def formation_chunks(limit, chunk_size: int):
    path = ROOT / "data" / "full_db_formation_energy_band_gap" / "structures.jsonl"
    chunk = []
    count = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            chunk.append(line)
            count += 1
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
            if limit and count >= int(limit):
                break
    if chunk:
        yield chunk


def vacancy_chunks(limit, chunk_size: int):
    path = ROOT / "data" / "full_db_vacancy_formation_energy" / "Vacancies.db"
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
    meta: dict[int, dict] = {}
    for sid, key, value in connection.execute(
            "SELECT id, key, value FROM text_key_values WHERE key IN ('mpid', 'formula_sc')"):
        meta.setdefault(sid, {})[key] = value
    chunk = []
    count = 0
    for sid, numbers, cell in connection.execute("SELECT id, numbers, cell FROM systems"):
        if numbers is None or cell is None:
            continue
        info = meta.get(sid, {})
        mid = info.get("mpid") or f"db-{sid}"
        chunk.append((mid, numbers, cell))
        count += 1
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
        if limit and count >= int(limit):
            break
    if chunk:
        yield chunk
    connection.close()


def collect_batio3(limit=None):
    path = (ROOT / "data" / "finetune_BaTiO3_doped"
            / "BaTiO3_structure_formation_energy_band_gap_stability" / "structures.pkl")
    with open(path, "rb") as fh:
        dataset = pickle.load(fh)
    structures = dataset["structures"]
    rows = []
    for mid, structure in structures.items():
        rows.append((str(mid), structure_features(structure)))
        if limit and len(rows) >= int(limit):
            break
    return rows


def collect_batio3_doped(limit=None):
    base = ROOT / "data" / "finetune_BaTiO3_doped" / "BaTiO3_doped_structure_formation_energy_convex_hull"
    frames = []
    for name in ("BaTiO3_doped_multi.pkl", "BaTiO3_doped_multi(rough).pkl"):
        with open(base / name, "rb") as fh:
            frames.append(pickle.load(fh))
    table = pd.concat(frames, ignore_index=True).drop_duplicates("material_id")
    rows = []
    for record in table.to_dict("records"):
        structure = record.get("structure")
        if structure is None:
            continue
        rows.append((str(record["material_id"]), structure_features(structure)))
        if limit and len(rows) >= int(limit):
            break
    return rows


def report_coverage(name: str, frame: pd.DataFrame) -> None:
    index_path = GRAPH_DIR / name / "index.csv"
    if not index_path.exists():
        return
    index = pd.read_csv(index_path, usecols=["material_id"])
    want = set(index["material_id"].astype(str))
    have = set(frame["material_id"].astype(str))
    missing = want - have
    print(f"   覆盖: index {len(want)} 条 | CSV {len(have)} 条 | 缺失 {len(missing)}")
    if missing:
        print(f"   缺失示例: {sorted(missing)[:5]}")


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 8 维全局特征表（旁挂，不重建图分片）")
    parser.add_argument("--only", nargs="*", default=None, help=f"数据集子集，可选: {', '.join(DATASETS)}")
    parser.add_argument("--jobs", type=int, default=8, help="并行进程数")
    parser.add_argument("--chunk", type=int, default=2000, help="每批处理条数")
    parser.add_argument("--limit", type=int, default=None, help="每个数据集限制条数（冒烟）")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的 global_feat.csv")
    args = parser.parse_args()

    names = args.only or list(DATASETS)
    for name in names:
        if name not in DATASETS:
            print(f"未知数据集: {name}")
            return 2
        out = GRAPH_DIR / name / "global_feat.csv"
        if out.exists() and not args.overwrite:
            print(f"[skip] {out} 已存在（--overwrite 可覆盖）")
            continue
        print(f"[build] {name}")
        start = time.time()
        if name == "formation_energy_band_gap":
            rows = parallel_process(formation_chunks(args.limit, args.chunk), _process_json_lines, args.jobs)
        elif name == "vacancy_screening":
            rows = parallel_process(vacancy_chunks(args.limit, args.chunk), _process_cells, args.jobs)
        elif name == "batio3":
            rows = collect_batio3(args.limit)
        else:
            rows = collect_batio3_doped(args.limit)
        frame = pd.DataFrame([(mid, *features) for mid, features in rows],
                             columns=["material_id", *COLUMNS]).drop_duplicates("material_id", keep="first")
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out, index=False, encoding="utf-8")
        print(f"   写出 {out} | {len(frame)} 行 | {time.time() - start:.1f}s")
        report_coverage(name, frame)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
