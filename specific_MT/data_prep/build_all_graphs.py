"""批量建图：data/ 下各数据集 -> graph/<名称>/{crystal_graph,line_graph}_partNNNNN.pkl

晶体图与线图分开保存（同名分片一一对应，同一 material_id 在两边都有）：
    crystal_graph_part00000.pkl   {mid: CrystalData(x, z, edge_index, edge_attr, y, ...)}
    line_graph_part00000.pkl      {mid: {line_edge_index, line_edge_attr, n_edges}}
    两边的 mid 顺序一致，训练时按分片成对加载。

线图编码 --line-mode：
    angle  每个线图边存 1 维夹角（度），省空间（推荐，训练时模型内做展开）
    rbf    16 维角度 RBF（与 build_graphs.py 默认一致，体积约 4 倍）

标量特征标准化（跨数据集共用同一套，预训练集统计一次）：
    --fit-stats-from dielectric   在介电 7327 条上统计 -> graph/feature_stats.json
    其余数据集自动复用该文件；已有 stats 时默认跳过重算（加 --refit 可强制）

用法：
    python build_all_graphs.py --list
    python build_all_graphs.py --only batio3_doped --limit 30
    python build_all_graphs.py --clean --jobs 12
    python build_all_graphs.py --only formation_energy_band_gap --jobs 12
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import time
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed
from pymatgen.core import Structure

from build_graphs import (GraphBuilder, OxidationResolver, SCALAR_COLUMNS, X_LAYOUT,
                          apply_stats, collect_scalars, compute_stats)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
GRAPH_DIR = ROOT / "graph"

_JOBLIB_TMP = GRAPH_DIR / "_joblib_tmp"
_JOBLIB_TMP.mkdir(parents=True, exist_ok=True)
os.environ["JOBLIB_TEMP_FOLDER"] = str(_JOBLIB_TMP)

SOURCES = {
    "dielectric": {
        "kind": "pkl_standard",
        "file": "data/full_db_dielectric/dielectric_dataset.pkl",
        "targets": ("e_total", "e_ionic", "e_electronic"),
        "label": "e_total",
        "group": None,
        "desc": "MP 全库 DFPT 介电 7327 条",
    },
    "formation_energy_band_gap": {
        "kind": "jsonl_parquet",
        "structures": "data/full_db_formation_energy_band_gap/structures.jsonl",
        "labels": "data/full_db_formation_energy_band_gap/materials.parquet",
        "targets": ("formation_energy_per_atom_eV", "band_gap_eV", "energy_above_hull_eV_per_atom"),
        "label": "formation_energy_per_atom_eV",
        "group": "split_group",
        "desc": "MP 全库形成能/带隙 153877 条",
    },
    "ferroelectric": {
        "kind": "ferro_jsonl",
        "structures": "data/ferroelectric_polarization/structures.jsonl",
        "labels": "data/ferroelectric_polarization/labels_unified.parquet",
        "targets": ("polarization_uC_cm2", "bandgap_polar_eV", "energy_diff_eV_per_atom"),
        "label": "polarization_uC_cm2",
        "group": "identifier",
        "desc": "铁电极化 2408 个结构 / 641 个极化标签",
    },
    "batio3": {
        "kind": "pkl_standard",
        "file": "data/finetune_BaTiO3_doped/BaTiO3_structure_formation_energy_band_gap_stability/structures.pkl",
        "targets": ("formation_energy_per_atom_eV", "band_gap_eV", "energy_above_hull_eV_per_atom"),
        "label": "formation_energy_per_atom_eV",
        "group": "reduced_formula",
        "desc": "BaTiO3 母体+掺杂 153 条（形成能/带隙）",
    },
    "batio3_doped": {
        "kind": "df_pkl",
        "files": (
            "data/finetune_BaTiO3_doped/BaTiO3_doped_structure_formation_energy_convex_hull/BaTiO3_doped_multi.pkl",
            "data/finetune_BaTiO3_doped/BaTiO3_doped_structure_formation_energy_convex_hull/BaTiO3_doped_multi(rough).pkl",
        ),
        "targets": ("formation_energy_per_atom", "energy_above_hull"),
        "label": "formation_energy_per_atom",
        "group": "formula",
        "desc": "BaTiO3 掺杂（La/Nd/Sr 等）形成能/凸包能 260 条",
    },
}


def load_records(name, cfg, limit=None):
    """生成器：yield (mid, structure, row_dict)。"""
    if cfg["kind"] == "pkl_standard":
        with open(ROOT / cfg["file"], "rb") as f:
            dataset = pickle.load(f)
        labels, structs = dataset["labels"], dataset["structures"]
        if limit:
            labels = labels.head(int(limit))
        for row in labels.to_dict("records"):
            mid = row["material_id"]
            st = structs.get(mid)
            if st is not None:
                yield str(mid), st, row
    elif cfg["kind"] == "jsonl_parquet":
        labels = pd.read_parquet(ROOT / cfg["labels"])
        keep = [c for c in labels.columns if c in cfg["targets"] or c == cfg.get("group")]
        table = labels.set_index("material_id")[keep].to_dict("index")
        n = 0
        with open(ROOT / cfg["structures"], encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                mid = obj["material_id"]
                row = table.get(mid)
                if row is None:
                    continue
                yield str(mid), Structure.from_dict(obj["structure"]), row
                n += 1
                if limit and n >= int(limit):
                    return
    elif cfg["kind"] == "ferro_jsonl":
        labels = pd.read_parquet(ROOT / cfg["labels"]).set_index("identifier")
        keep = [c for c in labels.columns if c in cfg["targets"]]
        table = labels[keep].to_dict("index")
        n = 0
        with open(ROOT / cfg["structures"], encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                row = table.get(obj["identifier"], {})
                yield str(obj["key"]), Structure.from_dict(obj["structure"]), row
                n += 1
                if limit and n >= int(limit):
                    return
    elif cfg["kind"] == "df_pkl":
        frames = []
        for rel in cfg["files"]:
            with open(ROOT / rel, "rb") as f:
                frames.append(pickle.load(f))
        df = pd.concat(frames, ignore_index=True).drop_duplicates("material_id")
        if limit:
            df = df.head(int(limit))
        for row in df.to_dict("records"):
            st = row.pop("structure", None)
            if st is None:
                continue
            row["nsites"] = len(st)
            yield str(row["material_id"]), st, row
    else:
        raise ValueError(f"未知数据源类型: {cfg['kind']}")


def prepare_shards(name, cfg, limit, shard_size, tmp_dir):
    tmp_dir.mkdir(parents=True, exist_ok=True)
    paths, buf, total = [], [], 0
    for mid, structure, row in load_records(name, cfg, limit):
        buf.append((mid, structure, row))
        total += 1
        if len(buf) >= shard_size:
            p = tmp_dir / f"{name}_{len(paths):05d}.pkl"
            with open(p, "wb") as f:
                pickle.dump(buf, f)
            paths.append(p)
            buf = []
    if buf:
        p = tmp_dir / f"{name}_{len(paths):05d}.pkl"
        with open(p, "wb") as f:
            pickle.dump(buf, f)
        paths.append(p)
    return paths, total


def _scan_shard(shard_file, oxi_mode):
    with open(shard_file, "rb") as f:
        records = pickle.load(f)
    resolver = OxidationResolver(oxi_mode)
    out = {name: [] for name, _ in SCALAR_COLUMNS}
    for mid, structure, _row in records:
        scalars = collect_scalars(structure, resolver.get(mid, structure))
        for key in out:
            out[key].append(scalars[key])
    return out, len(records)


def _process_shard(shard_file, params):
    with open(shard_file, "rb") as f:
        records = pickle.load(f)
    builder = GraphBuilder(**params["builder"])
    resolver = OxidationResolver(params["oxi_mode"])
    out_dir = Path(params["out_dir"])
    shard_id = int(Path(shard_file).stem.rsplit("_", 1)[-1])

    crystal, line, rows = {}, {}, []
    failed = 0
    for mid, structure, row in records:
        targets = {}
        for name in params["targets"]:
            value = row.get(name)
            if value is None:
                continue
            value = float(value)
            if value != value:
                continue
            targets[name] = value
        try:
            data = builder.build(structure, targets, target_name=params["label"],
                                 oxi_states=resolver.get(mid, structure))
        except Exception:
            failed += 1
            continue
        group = row.get(params["group"]) if params["group"] else None
        if group is not None:
            data.split_group = str(group)
        data.name = str(mid)
        line_index, line_attr = data.line_edge_index, data.line_edge_attr
        del data.line_edge_index
        del data.line_edge_attr
        crystal[mid] = data
        line[mid] = {"line_edge_index": line_index, "line_edge_attr": line_attr,
                     "n_edges": int(data.n_edges)}
        entry = {"material_id": mid, "formula": row.get("formula"),
                 "n_atoms": int(data.n_atoms), "n_edges": int(data.n_edges),
                 "n_line_edges": int(data.n_line_edges), "group": group}
        for name in params["targets"]:
            entry[name] = targets.get(name)
        rows.append(entry)

    apply_stats(crystal, params["stats"])
    with open(out_dir / f"crystal_graph_part{shard_id:05d}.pkl", "wb") as f:
        pickle.dump(crystal, f)
    with open(out_dir / f"line_graph_part{shard_id:05d}.pkl", "wb") as f:
        pickle.dump(line, f)
    return {"shard": shard_id, "n_graphs": len(crystal), "failed": failed, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description="批量建图（晶体图 + 线图分开保存）")
    parser.add_argument("--only", nargs="*", default=None, help="只处理这些数据集")
    parser.add_argument("--list", action="store_true", help="列出数据源后退出")
    parser.add_argument("--limit", type=int, default=None, help="每个数据集限制条数（冒烟）")
    parser.add_argument("--shard-size", type=int, default=5000, help="每个分片的图数")
    parser.add_argument("--jobs", type=int, default=12, help="并行进程数")
    parser.add_argument("--line-mode", choices=["angle", "rbf"], default="angle",
                        help="线图边特征：angle=1 维角度（省空间），rbf=16 维角度 RBF")
    parser.add_argument("--cutoff", type=float, default=8.0)
    parser.add_argument("--max-neighbors", type=int, default=12)
    parser.add_argument("--rbf-bins", type=int, default=48)
    parser.add_argument("--angle-bins", type=int, default=16)
    parser.add_argument("--oxi-mode", choices=["bva", "guess", "none"], default="bva")
    parser.add_argument("--stats", default=str(GRAPH_DIR / "feature_stats.json"))
    parser.add_argument("--fit-stats-from", default="dielectric", help="用哪个数据集统计标量特征")
    parser.add_argument("--refit", action="store_true", help="强制重算 stats")
    parser.add_argument("--clean", action="store_true", help="先删除 graph/ 与 data/graphs 里的旧图")
    parser.add_argument("--keep-tmp", action="store_true", help="保留临时分片")
    args = parser.parse_args()

    if args.list:
        for name, cfg in SOURCES.items():
            print(f"  {name:28s} {cfg['desc']}")
        return 0

    if args.clean:
        if GRAPH_DIR.exists():
            shutil.rmtree(GRAPH_DIR)
            print(f"[clean] 已删除 {GRAPH_DIR}")
        GRAPH_DIR.mkdir(parents=True, exist_ok=True)
        _JOBLIB_TMP.mkdir(parents=True, exist_ok=True)
        old = DATA_DIR / "graphs"
        if old.exists():
            shutil.rmtree(old)
            print(f"[clean] 已删除 {old}")
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = GRAPH_DIR / "_tmp"
    stats_path = Path(args.stats)

    selected = args.only or list(SOURCES)
    for name in selected:
        if name not in SOURCES:
            print(f"未知数据集: {name}（用 --list 查看）")
            return 2

    print(f"[1/3] 切分片 ...")
    shards_by_name = {}
    t0 = time.time()
    for name in selected:
        shards_by_name[name], total = prepare_shards(name, SOURCES[name], args.limit,
                                                     args.shard_size, tmp_dir)
        print(f"   {name:28s} {len(shards_by_name[name])} 片 / {total} 条")
    print(f"   切分耗时 {time.time() - t0:.1f}s")

    stats = None
    if stats_path.exists() and not args.refit:
        with open(stats_path, encoding="utf-8") as f:
            stats = json.load(f)
        print(f"[2/3] 复用 stats: {stats_path}")
    else:
        source = args.fit_stats_from
        if source not in shards_by_name:
            print(f"[2/3] 统计来源 {source} 不在本次范围内；先在 --only 里加上它，或指定已在范围内的 --fit-stats-from")
            return 2
        print(f"[2/3] 在 {source} 上统计标量特征 ...")
        t0 = time.time()
        pieces = Parallel(n_jobs=args.jobs)(
            delayed(_scan_shard)(p, args.oxi_mode) for p in shards_by_name[source])
        chunks = {key: [] for key, _ in SCALAR_COLUMNS}
        n_structures = 0
        for piece, n_rec in pieces:
            for key in chunks:
                chunks[key].extend(piece[key])
            n_structures += n_rec
        stats = compute_stats(chunks, source, n_structures)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"   {stats_path}")
        for key, _ in SCALAR_COLUMNS:
            s = stats["features"][key]
            print(f"   {key:20s} mean={s['mean']:8.4f} std={s['std']:8.4f} 缺失={s['missing_ratio'] * 100:.2f}%")
        print(f"   统计耗时 {time.time() - t0:.1f}s")

    if stats is None:
        print("[2/3] stats 未就绪，退出")
        return 2

    print(f"[3/3] 建图 ...")
    builder_args = {"cutoff": args.cutoff, "max_neighbors": args.max_neighbors,
                    "rbf_bins": args.rbf_bins, "angle_bins": args.angle_bins,
                    "line_encoding": args.line_mode}
    for name in selected:
        cfg = SOURCES[name]
        out_dir = GRAPH_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)
        params = {
            "builder": builder_args,
            "oxi_mode": args.oxi_mode,
            "targets": list(cfg["targets"]),
            "label": cfg["label"],
            "group": cfg.get("group"),
            "stats": stats,
            "out_dir": str(out_dir),
        }
        t0 = time.time()
        shards = shards_by_name[name]
        results = Parallel(n_jobs=args.jobs, verbose=0)(
            delayed(_process_shard)(p, params) for p in shards)
        rows = [r for res in sorted(results, key=lambda x: x["shard"]) for r in res["rows"]]
        failed = sum(res["failed"] for res in results)
        index_csv = out_dir / "index.csv"
        pd.DataFrame(rows).to_csv(index_csv, index=False, encoding="utf-8-sig")
        meta = {
            "source": name,
            "desc": cfg["desc"],
            "kind": cfg["kind"],
            "targets": list(cfg["targets"]),
            "label": cfg["label"],
            "group": cfg.get("group"),
            "builder": builder_args,
            "line_graph": "线图边 = 共享同一中心原子(dst)的两条键；特征 = 夹角（度）" if args.line_mode == "angle"
                          else "线图边 = 共享同一中心原子(dst)的两条键；特征 = 夹角 RBF(0-180)",
            "x_layout": X_LAYOUT,
            "stats_file": str(stats_path),
            "stats": stats,
            "normalized": True,
            "data_class": "graph_schema.CrystalData",
            "n_graphs": len(rows),
            "n_failed": failed,
            "shards": len(shards),
            "files": {
                "crystal": [p.name for p in sorted(out_dir.glob("crystal_graph_part*.pkl"))],
                "line": [p.name for p in sorted(out_dir.glob("line_graph_part*.pkl"))],
                "index": index_csv.name,
            },
        }
        with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        size_gb = sum(p.stat().st_size for p in out_dir.glob("*.pkl")) / 1024 ** 3
        print(f"   {name:28s} {len(rows):7d} 图 (失败 {failed})  {size_gb:6.2f} GB  "
              f"{time.time() - t0:6.1f}s -> {out_dir}")

    if not args.keep_tmp and tmp_dir.exists():
        shutil.rmtree(tmp_dir)
        print(f"[清理] 已删除临时分片 {tmp_dir}")

    total_gb = sum(p.stat().st_size for p in GRAPH_DIR.rglob("*.pkl")) / 1024 ** 3
    print(f"\n完成: graph/ 合计 {total_gb:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
