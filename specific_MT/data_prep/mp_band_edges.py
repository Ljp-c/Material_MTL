"""从 Materials Project electronic_structure endpoint 补充带边位置。

输出: cbm_eV / vbm_eV / efermi_eV（均相对费米能级，eV）；
      --with-band-structure 时导出完整能带 JSON 到 band_structures/。
输入: mp_gap_dataset.py 产出的 materials.csv，或任意含 material_id 的 CSV/JSONL。
策略: 字段自动探测 + 分块批处理 + 断点续传（cache/es/）。
注意: CBM/VBM 相对费米能级而非真空能级，绝对位置需功函数/电负性/实验校准。

用法:
    python mp_band_edges.py --probe
    python mp_band_edges.py --labels data/gap_dataset/materials.csv
    python mp_band_edges.py --labels data/gap_dataset/materials.csv --with-band-structure
"""
from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from importlib.metadata import version as pkg_version
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from mp_api.client import MPRester

EDGE_NAMES = ["cbm", "vbm", "efermi"]

CORE_FIELDS = ["material_id", "band_gap", "is_gap_direct", "is_metal",
               "band_gap_type"] + EDGE_NAMES

STRING_TARGETS = {"workflow_status", "workflow_category", "polar_mpid", "nonpolar_mpid"}


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


def doc_keys(doc):
    return set(doc_to_dict(doc).keys())


def get_field(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def probe_es(mpr, probe_id="mp-149"):
    docs = mpr.materials.electronic_structure.search(material_ids=[probe_id])
    if not docs:
        return None
    return docs[0]


def pick_fields(supported, with_bs):
    wanted = list(CORE_FIELDS)
    if with_bs:
        wanted.append("band_structure")
    chosen = [f for f in wanted if f in supported]
    missing = [f for f in wanted if f not in supported]
    return chosen, missing


def load_ids(labels_path):
    path = Path(labels_path)
    if path.suffix.lower() == ".jsonl":
        ids = []
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                mid = record.get("material_id")
                if mid:
                    ids.append(str(mid))
    else:
        df = pd.read_csv(path)
        if "material_id" not in df.columns:
            raise SystemExit(f"{path} 缺少 material_id 列")
        ids = df["material_id"].dropna().astype(str).tolist()
    return sorted(set(ids))


def fetch_edges(mpr, ids, cache_dir, fields, chunk_size=200, force=False, retries=3):
    chunks_dir = cache_dir / "es"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    n_chunks = math.ceil(len(ids) / chunk_size) if ids else 0
    for i in tqdm(range(n_chunks), desc="es"):
        path = chunks_dir / f"es_{i:05d}.jsonl"
        tmp = path.with_suffix(".jsonl.tmp")
        if path.exists() and path.stat().st_size > 0 and not force:
            continue
        chunk = ids[i * chunk_size:(i + 1) * chunk_size]
        kwargs = {"material_ids": chunk, "fields": fields}
        docs = None
        for attempt in range(retries):
            try:
                docs = mpr.materials.electronic_structure.search(**kwargs)
                break
            except Exception as exc:
                print(f"  分块 {i} 第 {attempt + 1} 次失败: {exc}")
                time.sleep(2.0 * (attempt + 1))
        if docs is None:
            print(f"  分块 {i} 放弃，下次运行将自动续传")
            continue
        with open(tmp, "w", encoding="utf-8") as handle:
            for doc in docs:
                handle.write(json.dumps(doc_to_dict(doc), ensure_ascii=False,
                                        default=json_default) + "\n")
        tmp.replace(path)
    return n_chunks


def iter_records(chunks_dir):
    for path in sorted(chunks_dir.glob("es_*.jsonl")):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)


def split_edge_value(value):
    if value is None:
        return None, None
    if isinstance(value, bool):
        return None, value
    if isinstance(value, (int, float)):
        return float(value), None
    if isinstance(value, dict):
        energy = value.get("energy", value.get("value"))
        if isinstance(energy, (int, float)):
            return float(energy), value
        return None, value
    energy = getattr(value, "energy", None)
    if isinstance(energy, (int, float)):
        as_dict = getattr(value, "as_dict", None)
        raw = as_dict() if callable(as_dict) else str(value)
        return float(energy), raw
    return None, str(value)


def load_band_structure(bs_dict):
    if not isinstance(bs_dict, dict):
        return None
    from pymatgen.electronic_structure.bandstructure import (
        BandStructure, BandStructureSymmLine)
    name = bs_dict.get("@class")
    cls = BandStructureSymmLine if name == "BandStructureSymmLine" else BandStructure
    try:
        return cls.from_dict(bs_dict)
    except Exception:
        try:
            return BandStructure.from_dict(bs_dict)
        except Exception:
            return None


def flatten_edge_record(raw):
    row = {
        "material_id": raw.get("material_id"),
        "band_gap_eV": raw.get("band_gap"),
        "band_gap_type": raw.get("band_gap_type"),
        "is_gap_direct": raw.get("is_gap_direct"),
        "is_metal": raw.get("is_metal"),
        "has_band_structure": raw.get("band_structure") is not None,
    }
    for name in EDGE_NAMES:
        energy, extra = split_edge_value(raw.get(name))
        row[f"{name}_eV"] = energy
        row[f"{name}_json"] = None if extra is None else json.dumps(
            extra, ensure_ascii=False, default=json_default)
    return row


def fill_edges_from_bs(row, bs_dict):
    bs = load_band_structure(bs_dict)
    if bs is None:
        return
    for name, getter in (("cbm", "get_cbm"), ("vbm", "get_vbm")):
        if row.get(f"{name}_eV") is not None:
            continue
        fn = getattr(bs, getter, None)
        if not callable(fn):
            continue
        try:
            info = fn()
        except Exception:
            continue
        if isinstance(info, dict):
            energy = info.get("energy")
            if isinstance(energy, (int, float)):
                row[f"{name}_eV"] = float(energy)
                row[f"{name}_json"] = json.dumps(info, ensure_ascii=False, default=json_default)


def recompute_gap_check(row):
    cbm = row.get("cbm_eV")
    vbm = row.get("vbm_eV")
    row["gap_from_edges_eV"] = None if (cbm is None or vbm is None) else cbm - vbm
    gap = row.get("band_gap_eV")
    if row["gap_from_edges_eV"] is not None and gap is not None:
        row["gap_check_ok"] = bool(abs(row["gap_from_edges_eV"] - gap) < 0.05)
    else:
        row["gap_check_ok"] = None
    return row


def save_table(df, base_path):
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(base_path.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    parquet_error = None
    try:
        df.to_parquet(base_path.with_suffix(".parquet"), index=False)
    except Exception as exc:
        parquet_error = str(exc)
    return parquet_error


def main():
    parser = argparse.ArgumentParser(description="MP 带边位置补充")
    parser.add_argument("--labels", default=None, help="materials.csv 或 ids.jsonl")
    parser.add_argument("--out", default=None)
    parser.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument("--with-band-structure", action="store_true",
                        help="同时导出完整能带 JSON（体积大）")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()

    if args.probe:
        with MPRester() as mpr:
            doc = probe_es(mpr)
        if doc is None:
            print("探测失败：未返回文档")
            return 1
        keys = sorted(doc_keys(doc))
        print(f"materials.electronic_structure 支持字段 {len(keys)} 个：")
        for name in keys:
            print(f"  {name}")
        return 0

    if not args.labels:
        raise SystemExit("请用 --labels 指定 materials.csv / ids.jsonl，或先运行 --probe")
    labels_path = Path(args.labels)
    if not labels_path.exists():
        raise SystemExit(f"找不到输入文件: {labels_path}")
    out_dir = Path(args.out) if args.out else labels_path.parent / "band_edges"
    cache_dir = out_dir / "cache"
    query_date = utc_now()

    ids = load_ids(labels_path)
    if args.limit:
        ids = ids[:args.limit]
    print(f"输入材料数: {len(ids)}")

    with MPRester() as mpr:
        doc = probe_es(mpr)
        supported = doc_keys(doc) if doc is not None else set()
        if not supported:
            print("警告: 字段探测失败，使用最小字段集")
            fields = ["material_id", "band_gap", "is_gap_direct", "is_metal"]
            missing = [f for f in CORE_FIELDS if f not in fields]
        else:
            fields, missing = pick_fields(supported, args.with_band_structure)
            print(f"electronic_structure 支持 {len(supported)} 个字段；本次请求 {len(fields)} 个")
            if missing:
                print(f"不可用字段（跳过）: {missing}")
        if not fields:
            print("没有可用字段，终止")
            return 1
        fetch_edges(mpr, ids, cache_dir, fields,
                    chunk_size=args.chunk_size, force=args.force)

    rows = []
    struct_dir = out_dir / "band_structures"
    for raw in iter_records(cache_dir / "es"):
        row = flatten_edge_record(raw)
        bs_dict = raw.get("band_structure")
        if args.with_band_structure and bs_dict is not None:
            struct_dir.mkdir(parents=True, exist_ok=True)
            target = struct_dir / f"{row['material_id']}.json"
            if args.force or not target.exists():
                target.write_text(
                    json.dumps(bs_dict, ensure_ascii=False, default=json_default),
                    encoding="utf-8")
        if bs_dict is not None:
            fill_edges_from_bs(row, bs_dict)
        rows.append(recompute_gap_check(row))

    df = pd.DataFrame(rows)
    if df.empty:
        print("没有采集到数据")
        return 1
    df = df.drop_duplicates("material_id").reset_index(drop=True)
    parquet_error = save_table(df, out_dir / "band_edges")

    meta = {
        "dataset": "mp_electronic_structure_edges",
        "created_utc": utc_now(),
        "query_date": query_date,
        "endpoint": "materials.electronic_structure",
        "mp_api_version": package_version("mp-api"),
        "pymatgen_version": package_version("pymatgen"),
        "source_labels": str(labels_path),
        "fields_requested": fields,
        "fields_missing_from_api": missing,
        "with_band_structure": bool(args.with_band_structure),
        "n_materials": int(len(df)),
        "n_with_band_structure": int(df["has_band_structure"].sum()),
        "parquet_written": parquet_error is None,
        "parquet_error": parquet_error,
        "notes": [
            "cbm_eV / vbm_eV / efermi_eV 均相对费米能级，不是绝对真空能级",
            "绝对带边位置需功函数/电负性法/实验 IP-EA 或杂化泛函校准",
            "gap_check_ok = |cbm-vbm - band_gap| < 0.05 eV（自洽性抽查）",
            "轨道/元素投影在 band_structures/*.json 的 projections 中（若原始数据带投影）",
        ],
    }
    (out_dir / "band_edges_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
    print(f"完成: {len(df)} 条 -> {out_dir}")
    print("  band_edges.csv / band_edges.parquet / band_edges_meta.json")
    if args.with_band_structure:
        print(f"  完整能带 JSON: {struct_dir}")
    if parquet_error:
        print(f"  Parquet 未写出: {parquet_error}")
    if bool(df["cbm_eV"].notna().any()):
        print(f"  含 CBM 数值: {int(df['cbm_eV'].notna().sum())} 条 | "
              f"gap 自洽通过: {int((df['gap_check_ok'] == True).sum())} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
