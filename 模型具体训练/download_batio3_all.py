"""下载「钛酸钡（BaTiO3）相关全部 MP 数据」，按性质分目录存放，供微调。

范围:
    1) Ba-Ti-O 三元体系全部材料（BaTiO3 各多型）
    2) BaTiO3 基掺杂家族 Ba-Ti-O-X（默认 30 种常见掺杂元素）
    3) 对全部材料批量拉取性质端点，并按性质分别存放到独立目录：
       BaTiO3_结构_形成能_带隙_稳定性 / BaTiO3_介电 / BaTiO3_弹性 /
       BaTiO3_磁性 / BaTiO3_热力学 / BaTiO3_声子 / BaTiO3_压电
输出（data/ 下）:
    BaTiO3_结构_形成能_带隙_稳定性/  materials.csv(.parquet)、structures.jsonl、structures.pkl
    BaTiO3_<其他性质>/               <endpoint>.csv(.parquet)
    各目录 cache/<endpoint>_raw.jsonl 为原始 JSONL 备份
特性: 分块 material_ids 查询、按端点断点续传

用法:
    python download_batio3_all.py
    python download_batio3_all.py --no-dopants
    python download_batio3_all.py --endpoints summary,dielectric
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import time
from datetime import datetime, timezone
from importlib.metadata import version as pkg_version
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from mp_api.client import MPRester

DOPANTS = ["La", "Nd", "Sm", "Pb", "Sr", "K", "Ca", "Nb", "Zr", "Sn", "Hf", "Fe",
           "Co", "Ni", "Cr", "Mn", "Mg", "Al", "Sc", "V", "Si", "Y", "Gd", "Dy",
           "Er", "Ho", "Tb", "Eu", "Li", "Bi"]

ENDPOINT_CHOICES = ["summary", "dielectric", "elasticity", "piezo",
                    "phonon", "magnetism", "thermo"]

PROPERTY_DIRS = {
    "summary": "BaTiO3_结构_形成能_带隙_稳定性",
    "dielectric": "BaTiO3_介电",
    "elasticity": "BaTiO3_弹性",
    "piezo": "BaTiO3_压电",
    "phonon": "BaTiO3_声子",
    "magnetism": "BaTiO3_磁性",
    "thermo": "BaTiO3_热力学",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def package_version(name):
    try:
        return pkg_version(name)
    except Exception:
        return None


def json_default(obj):
    as_dict = getattr(obj, "as_dict", None)
    if callable(as_dict):
        try:
            return as_dict()
        except Exception:
            pass
    value = getattr(obj, "value", None)
    if value is not None and not callable(value):
        return value
    symbol = getattr(obj, "symbol", None)
    if isinstance(symbol, str):
        return symbol
    return str(obj)


def doc_to_dict(doc):
    if isinstance(doc, dict):
        return doc
    for name in ("model_dump", "dict"):
        fn = getattr(doc, name, None)
        if callable(fn):
            try:
                data = fn()
            except Exception:
                continue
            if isinstance(data, dict):
                return data
    return {"value": str(doc)}


def get_field(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def collect_ids(mpr, include_dopants):
    ids = set()
    chemsys_list = ["Ba-O-Ti"]
    if include_dopants:
        for dopant in DOPANTS:
            chemsys_list.append("-".join(sorted({"Ba", "Ti", "O", dopant})))
    for chemsys in tqdm(chemsys_list, desc="chemsys"):
        try:
            docs = mpr.materials.summary.search(chemsys=chemsys, fields=["material_id"])
        except Exception as exc:
            print(f"  查询失败 {chemsys}: {exc}")
            continue
        for doc in docs:
            mid = get_field(doc, "material_id")
            if mid is not None:
                ids.add(str(mid))
    return sorted(ids)


def fetch_summary(mpr, ids, prop_dir, chunk_size=200, force=False):
    cache_dir = prop_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_path = cache_dir / "summary_raw.jsonl"
    if not raw_path.exists() or raw_path.stat().st_size == 0 or force:
        tmp = raw_path.with_suffix(".jsonl.tmp")
        n_chunks = math.ceil(len(ids) / chunk_size)
        with open(tmp, "w", encoding="utf-8") as handle:
            for i in tqdm(range(n_chunks), desc="summary"):
                chunk = ids[i * chunk_size:(i + 1) * chunk_size]
                docs = None
                for attempt in range(3):
                    try:
                        docs = mpr.materials.summary.search(material_ids=chunk)
                        break
                    except Exception as exc:
                        print(f"    summary 分块 {i} 第 {attempt + 1} 次失败: {exc}")
                        time.sleep(1.5 * (attempt + 1))
                if docs is None:
                    continue
                for doc in docs:
                    handle.write(json.dumps(doc_to_dict(doc), ensure_ascii=False,
                                            default=json_default) + "\n")
                time.sleep(0.1)
        tmp.replace(raw_path)

    rows = []
    structures = {}
    with open(raw_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            mid = raw.get("material_id")
            if mid is None:
                continue
            rows.append(flatten_summary_record(raw))
            struct = raw.get("structure")
            if struct is not None:
                structures[str(mid)] = struct
    df = pd.DataFrame(rows).drop_duplicates("material_id").reset_index(drop=True)
    df.to_csv(prop_dir / "materials.csv", index=False, encoding="utf-8-sig")
    try:
        df.to_parquet(prop_dir / "materials.parquet", index=False)
    except Exception as exc:
        print("  materials.parquet 未写出:", exc)

    with open(prop_dir / "structures.jsonl", "w", encoding="utf-8") as handle:
        for mid, struct in structures.items():
            handle.write(json.dumps({"material_id": mid, "structure": struct},
                                    ensure_ascii=False, default=json_default) + "\n")
    try:
        from pymatgen.core import Structure
        struct_objects = {}
        for mid, struct in structures.items():
            if isinstance(struct, dict):
                struct_objects[mid] = Structure.from_dict(struct)
        with open(prop_dir / "structures.pkl", "wb") as handle:
            pickle.dump({"structures": struct_objects, "labels": df}, handle)
        print(f"  structures.pkl: {len(struct_objects)} 个结构")
    except Exception as exc:
        print("  structures.pkl 未写出:", exc)
    return df


def flatten_summary_record(raw):
    symmetry = raw.get("symmetry")
    symmetry = symmetry if isinstance(symmetry, dict) else {}
    return {
        "material_id": raw.get("material_id"),
        "formula": raw.get("formula_pretty"),
        "chemsys": raw.get("chemsys"),
        "nsites": raw.get("nsites"),
        "nelements": raw.get("nelements"),
        "volume_A3": raw.get("volume"),
        "density_g_cm3": raw.get("density"),
        "spacegroup_number": symmetry.get("number"),
        "spacegroup_symbol": symmetry.get("symbol"),
        "crystal_system": symmetry.get("crystal_system"),
        "band_gap_eV": raw.get("band_gap"),
        "is_gap_direct": raw.get("is_gap_direct"),
        "is_metal": raw.get("is_metal"),
        "formation_energy_per_atom_eV": raw.get("formation_energy_per_atom"),
        "energy_above_hull_eV_per_atom": raw.get("energy_above_hull"),
        "is_stable": raw.get("is_stable"),
        "is_magnetic": raw.get("is_magnetic"),
        "magnetic_ordering": str(raw.get("ordering")) if raw.get("ordering") is not None else None,
        "total_magnetization": raw.get("total_magnetization"),
        "theoretical": raw.get("theoretical"),
        "deprecated": raw.get("deprecated"),
    }


def fetch_endpoint(mpr, endpoint, ids, cache_dir, chunk_size=200, force=False):
    raw_path = cache_dir / f"{endpoint}_raw.jsonl"
    if raw_path.exists() and raw_path.stat().st_size > 0 and not force:
        print(f"  [{endpoint}] 已有缓存，跳过下载")
        return
    rester = getattr(mpr.materials, endpoint, None)
    if rester is None:
        print(f"  [{endpoint}] 该端点不存在，跳过")
        return
    n_chunks = math.ceil(len(ids) / chunk_size)
    tmp = raw_path.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        for i in tqdm(range(n_chunks), desc=endpoint):
            chunk = ids[i * chunk_size:(i + 1) * chunk_size]
            docs = None
            for attempt in range(3):
                try:
                    docs = rester.search(material_ids=chunk)
                    break
                except Exception as exc:
                    print(f"    [{endpoint}] 分块 {i} 第 {attempt + 1} 次失败: {exc}")
                    time.sleep(1.5 * (attempt + 1))
            if docs is None:
                continue
            for doc in docs:
                handle.write(json.dumps(doc_to_dict(doc), ensure_ascii=False,
                                        default=json_default) + "\n")
            time.sleep(0.1)
    tmp.replace(raw_path)


def flatten_endpoint(cache_dir, endpoint, prop_dir):
    raw_path = cache_dir / f"{endpoint}_raw.jsonl"
    if not raw_path.exists():
        return None
    rows = []
    with open(raw_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            row = {}
            for key, value in record.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    row[key] = value
                elif isinstance(value, list) and (
                        not value or isinstance(value[0], (str, int, float, bool))):
                    row[key] = ",".join(str(item) for item in value)
            rows.append(row)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df.to_csv(prop_dir / f"{endpoint}.csv", index=False, encoding="utf-8-sig")
    try:
        df.to_parquet(prop_dir / f"{endpoint}.parquet", index=False)
    except Exception:
        pass
    return df


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="下载 BaTiO3 全部相关 MP 数据（按性质分目录）")
    parser.add_argument("--root", default=str(here / "data"),
                        help="数据根目录，各性质子目录创建于此")
    parser.add_argument("--endpoints", default=",".join(ENDPOINT_CHOICES))
    parser.add_argument("--no-dopants", action="store_true", help="只下 Ba-Ti-O 三元")
    parser.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    endpoints = [name.strip() for name in args.endpoints.split(",") if name.strip()]
    query_date = utc_now()
    active = [e for e in endpoints if e in ENDPOINT_CHOICES]

    with MPRester() as mpr:
        print("收集 BaTiO3 相关材料 ID ...")
        ids = collect_ids(mpr, not args.no_dopants)
        print(f"共 {len(ids)} 条材料")
        summary_df = None
        if "summary" in active:
            print("[summary] 拉取 ...")
            summary_df = fetch_summary(mpr, ids, root / PROPERTY_DIRS["summary"],
                                       chunk_size=args.chunk_size, force=args.force)
        for endpoint in active:
            if endpoint == "summary":
                continue
            prop_dir = root / PROPERTY_DIRS.get(endpoint, f"BaTiO3_{endpoint}")
            cache_dir = prop_dir / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"[{endpoint}] 拉取 -> {prop_dir.name} ...")
            fetch_endpoint(mpr, endpoint, ids, cache_dir,
                           chunk_size=args.chunk_size, force=args.force)

    counts = {}
    for endpoint in active:
        if endpoint == "summary":
            continue
        prop_dir = root / PROPERTY_DIRS.get(endpoint, f"BaTiO3_{endpoint}")
        df = flatten_endpoint(prop_dir / "cache", endpoint, prop_dir)
        counts[endpoint] = None if df is None else int(len(df))

    meta = {
        "dataset": "batio3_all_by_property",
        "created_utc": utc_now(),
        "query_date": query_date,
        "scope": "Ba-Ti-O + Ba-Ti-O-X(30 dopants)" if not args.no_dopants else "Ba-Ti-O",
        "n_materials": len(ids),
        "endpoints": active,
        "endpoint_counts": counts,
        "n_summary": None if summary_df is None else int(len(summary_df)),
        "layout": {e: PROPERTY_DIRS.get(e, f"BaTiO3_{e}") for e in active},
        "mp_api_version": package_version("mp-api"),
        "pymatgen_version": package_version("pymatgen"),
    }
    (root / "BaTiO3_数据集_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
    print("完成。各性质目录：")
    for endpoint in active:
        prop_dir = root / PROPERTY_DIRS.get(endpoint, f"BaTiO3_{endpoint}")
        print(f"  {prop_dir.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
