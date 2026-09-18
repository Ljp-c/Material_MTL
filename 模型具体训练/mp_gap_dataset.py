"""从 Materials Project 构建「带隙 + 热力学稳定性 + 结构」数据集。

数据源: materials.summary endpoint。
字段: band_gap(eV) / is_gap_direct / is_metal / formation_energy_per_atom(eV/atom) /
      energy_above_hull(eV/atom) / symmetry / density / volume / structure / 磁学字段。
策略: 字段自动探测 + 按 material_ids 分块批处理 + 断点续传（cache/ 目录）+
      CSV/Parquet/JSONL 导出 + 可选合并 HSE06/GW/实验带隙校准表。
默认: 全库（阶段一预训练数据）；用 --batio3 / --chemsys 可只下子集。
注意: summary 的 band_gap 为 GGA/GGA+U，仅作初筛；高保真标签需校准。

用法:
    python mp_gap_dataset.py --probe
    python mp_gap_dataset.py                      # 全库（默认，预训练用）
    python mp_gap_dataset.py --batio3             # 只下 BaTiO3 掺杂家族
    python mp_gap_dataset.py --chemsys "Ba-O-Ti,Ba-Nb-O-Ti"
    python mp_gap_dataset.py --limit 5000
    python mp_gap_dataset.py --calib-csv band_gap_calibrated.csv
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
from pymatgen.core import Composition

DOPANTS = ["La", "Nd", "Sm", "Pb", "Sr", "K", "Ca", "Nb", "Zr", "Sn", "Hf", "Fe",
           "Co", "Ni", "Cr", "Mn", "Mg", "Al", "Sc", "V", "Si", "Y", "Gd", "Dy",
           "Er", "Ho", "Tb", "Eu", "Li", "Bi"]

CORE_FIELDS = [
    "material_id", "formula_pretty", "chemsys", "elements", "nsites", "nelements",
    "density", "volume", "symmetry", "structure", "band_gap", "is_gap_direct",
    "is_metal", "formation_energy_per_atom", "energy_above_hull", "deprecated",
]

OPTIONAL_FIELDS = [
    "is_stable", "is_magnetic", "ordering", "total_magnetization",
    "num_magnetic_sites", "num_sites", "theoretical", "band_gap_type",
    "cbm", "vbm", "efermi", "origins", "last_updated", "database_IDs",
]

NUMERIC_TARGETS = [
    "band_gap_eV", "formation_energy_per_atom_eV", "energy_above_hull_eV_per_atom",
]

STABILITY_TOL = 1e-5
METASTABLE_MAX = 0.1


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


def enum_text(value):
    if value is None:
        return None
    inner = getattr(value, "value", None)
    if inner is not None and not callable(inner):
        return str(inner)
    return str(value)


def database_version(mpr):
    fn = getattr(mpr, "get_database_version", None)
    if callable(fn):
        try:
            return str(fn())
        except Exception:
            return None
    return None


def probe_summary(mpr, probe_id="mp-149"):
    docs = mpr.materials.summary.search(material_ids=[probe_id])
    if not docs:
        return None
    return docs[0]


def pick_fields(supported, full):
    if full or not supported:
        return None, []
    wanted = CORE_FIELDS + OPTIONAL_FIELDS
    chosen = [f for f in wanted if f in supported]
    missing = [f for f in wanted if f not in supported]
    return chosen, missing


def build_chemsys_list(args):
    if args.chemsys:
        return [cs.strip() for cs in args.chemsys.split(",") if cs.strip()]
    if args.batio3:
        combos = {"Ba-O-Ti"}
        for dopant in DOPANTS:
            combos.add("-".join(sorted({"Ba", "Ti", "O", dopant})))
        return sorted(combos)
    return None


def collect_ids(mpr, chemsys_list, cache_dir, force=False):
    cache_dir.mkdir(parents=True, exist_ok=True)
    ids_path = cache_dir / "ids.jsonl"
    progress_path = cache_dir / "ids_progress.json"
    if force:
        for path in (ids_path, progress_path):
            if path.exists():
                path.unlink()
    progress = {"done": {}}
    if progress_path.exists():
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
        except Exception:
            progress = {"done": {}}
    done = progress.setdefault("done", {})
    if chemsys_list is None:
        targets = [("ALL", n) for n in range(1, 10)]
    else:
        targets = [(chemsys, None) for chemsys in chemsys_list]
    with open(ids_path, "a", encoding="utf-8") as handle:
        for chemsys, nelements in tqdm(targets, desc="ids"):
            key = chemsys if nelements is None else f"ALL_nelements={nelements}"
            if key in done:
                continue
            docs = None
            for attempt in range(3):
                try:
                    if nelements is None:
                        docs = mpr.materials.summary.search(
                            chemsys=chemsys, fields=["material_id"])
                    else:
                        docs = mpr.materials.summary.search(
                            nelements=nelements, fields=["material_id"])
                    break
                except Exception as exc:
                    print(f"  {key} 第 {attempt + 1} 次失败: {exc}")
                    time.sleep(2.0 * (attempt + 1))
            if docs is None:
                print(f"  {key} 放弃，下次运行将续传")
                continue
            n = 0
            for doc in docs:
                mid = get_field(doc, "material_id")
                if mid is None:
                    continue
                handle.write(json.dumps({"chemsys": key, "material_id": str(mid)}) + "\n")
                n += 1
            handle.flush()
            done[key] = n
            progress["done"] = done
            progress_path.write_text(
                json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")
    ids = []
    seen = set()
    with open(ids_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            mid = json.loads(line)["material_id"]
            if mid not in seen:
                seen.add(mid)
                ids.append(mid)
    return sorted(ids)


def fetch_summaries(mpr, ids, cache_dir, fields, chunk_size=400, force=False, retries=3):
    chunks_dir = cache_dir / "summary"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    n_chunks = math.ceil(len(ids) / chunk_size) if ids else 0
    for i in tqdm(range(n_chunks), desc="summary"):
        path = chunks_dir / f"summary_{i:05d}.jsonl"
        tmp = path.with_suffix(".jsonl.tmp")
        if path.exists() and path.stat().st_size > 0 and not force:
            continue
        chunk = ids[i * chunk_size:(i + 1) * chunk_size]
        kwargs = {"material_ids": chunk}
        if fields is not None:
            kwargs["fields"] = fields
        docs = None
        for attempt in range(retries):
            try:
                docs = mpr.materials.summary.search(**kwargs)
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


def iter_raw_records(chunks_dir):
    for path in sorted(chunks_dir.glob("summary_*.jsonl")):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)


def flatten_record(raw):
    sym = raw.get("symmetry")
    if isinstance(sym, dict):
        sg_number = sym.get("number")
        sg_symbol = sym.get("symbol")
        crystal_system = sym.get("crystal_system")
        point_group = sym.get("point_group")
    elif sym is not None:
        sg_number = getattr(sym, "number", None)
        sg_symbol = getattr(sym, "symbol", None)
        crystal_system = getattr(sym, "crystal_system", None)
        point_group = getattr(sym, "point_group", None)
    else:
        sg_number = sg_symbol = crystal_system = point_group = None
    elements = raw.get("elements") or []
    row = {
        "material_id": raw.get("material_id"),
        "formula": raw.get("formula_pretty"),
        "chemsys": raw.get("chemsys"),
        "elements": "-".join(sorted(str(e) for e in elements)),
        "nsites": raw.get("nsites"),
        "nelements": raw.get("nelements"),
        "volume_A3": raw.get("volume"),
        "density_g_cm3": raw.get("density"),
        "spacegroup_number": sg_number,
        "spacegroup_symbol": sg_symbol,
        "crystal_system": enum_text(crystal_system),
        "point_group": enum_text(point_group),
        "band_gap_eV": raw.get("band_gap"),
        "band_gap_type": raw.get("band_gap_type"),
        "is_gap_direct": raw.get("is_gap_direct"),
        "is_metal": raw.get("is_metal"),
        "formation_energy_per_atom_eV": raw.get("formation_energy_per_atom"),
        "energy_above_hull_eV_per_atom": raw.get("energy_above_hull"),
        "is_stable": raw.get("is_stable"),
        "is_magnetic": raw.get("is_magnetic"),
        "magnetic_ordering": enum_text(raw.get("ordering")),
        "total_magnetization": raw.get("total_magnetization"),
        "num_magnetic_sites": raw.get("num_magnetic_sites"),
        "theoretical": raw.get("theoretical"),
        "deprecated": raw.get("deprecated"),
        "last_updated": raw.get("last_updated"),
        "has_structure": raw.get("structure") is not None,
    }
    return row


def build_frames(chunks_dir, limit=None):
    rows = []
    structures = []
    for raw in iter_raw_records(chunks_dir):
        if limit and len(rows) >= limit:
            break
        struct = raw.get("structure")
        rows.append(flatten_record(raw))
        if struct is not None:
            structures.append({
                "material_id": raw.get("material_id"),
                "formula": raw.get("formula_pretty"),
                "structure": struct,
            })
    return pd.DataFrame(rows), structures


def reduced_formula(formula):
    if not formula:
        return None
    try:
        return Composition(formula).reduced_formula
    except Exception:
        return None


def stability_class(eah):
    if eah is None:
        return None
    try:
        value = float(eah)
    except (TypeError, ValueError):
        return None
    if math.isnan(value):
        return None
    if value <= STABILITY_TOL:
        return "stable"
    if value <= METASTABLE_MAX:
        return "metastable"
    return "unstable"


def trust_from_method(method):
    if method is None:
        return "C_gga"
    text = str(method).lower()
    if "exp" in text or "gw" in text:
        return "A_gw_or_exp"
    if "hse" in text:
        return "B_hse"
    return "B_calibrated"


def postprocess(df, query_date):
    df = df.copy()
    df["material_id"] = df["material_id"].astype(str)
    df = df.drop_duplicates("material_id", keep="first").reset_index(drop=True)
    df["query_date"] = query_date
    reduced = df["formula"].map(reduced_formula)
    df["reduced_formula"] = reduced
    df["stability_class"] = df["energy_above_hull_eV_per_atom"].map(stability_class)
    is_metal = df["is_metal"].fillna(False).astype(bool)
    df["has_band_gap"] = df["band_gap_eV"].notna() & ~is_metal
    df["band_gap_trust"] = ["none" if pd.isna(gap) else "C_gga" for gap in df["band_gap_eV"]]
    counts = df.groupby("reduced_formula")["material_id"].transform("nunique")
    df["polymorph_count"] = counts.fillna(1).astype(int)
    df["split_group"] = df["reduced_formula"].fillna(df["formula"])
    df["preferred_stability"] = df["stability_class"].isin(["stable", "metastable"]).fillna(False)
    preferred = (
        df["preferred_stability"]
        & df["has_band_gap"].fillna(False)
        & df["has_structure"].fillna(False)
        & (~df["deprecated"].fillna(False).astype(bool))
    )
    df["preferred"] = preferred.astype(bool)
    missing_mask = df[NUMERIC_TARGETS].isna()
    df["missing_fields"] = [
        ",".join(name for name, flag in zip(NUMERIC_TARGETS, row))
        for row in missing_mask.to_numpy()
    ]
    return df


def apply_calibration(df, calib_csv, gap_col, method_col, ref_col):
    calib = pd.read_csv(calib_csv)
    if "material_id" not in calib.columns:
        raise SystemExit("校准表必须包含 material_id 列")
    if gap_col not in calib.columns:
        raise SystemExit(f"校准表必须包含带隙列 {gap_col}")
    columns = ["material_id", gap_col]
    for name in (method_col, ref_col):
        if name in calib.columns:
            columns.append(name)
    calib = calib[columns].drop_duplicates("material_id")
    rename = {gap_col: "band_gap_calibrated_eV"}
    if method_col in calib.columns:
        rename[method_col] = "band_gap_calib_method"
    if ref_col in calib.columns:
        rename[ref_col] = "band_gap_calib_ref"
    calib = calib.rename(columns=rename)
    df = df.merge(calib, on="material_id", how="left")
    if "band_gap_calib_method" in df.columns:
        methods = df["band_gap_calib_method"]
    else:
        methods = pd.Series([None] * len(df))
    trust = []
    for gap, method, base in zip(df["band_gap_calibrated_eV"], methods, df["band_gap_trust"]):
        trust.append(trust_from_method(method) if pd.notna(gap) else base)
    df["band_gap_trust"] = trust
    return df


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


def export_all(df, structures, out_dir, meta):
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_error = save_table(df, out_dir / "materials")
    with open(out_dir / "structures.jsonl", "w", encoding="utf-8") as handle:
        for item in structures:
            handle.write(json.dumps(item, ensure_ascii=False, default=json_default) + "\n")
    meta["n_materials"] = int(len(df))
    meta["n_structures"] = len(structures)
    meta["n_preferred"] = int(df["preferred"].sum())
    meta["n_stable"] = int((df["stability_class"] == "stable").sum())
    meta["n_metastable"] = int((df["stability_class"] == "metastable").sum())
    meta["n_unstable"] = int((df["stability_class"] == "unstable").sum())
    meta["n_with_band_gap"] = int(df["has_band_gap"].sum())
    meta["parquet_written"] = parquet_error is None
    meta["parquet_error"] = parquet_error
    (out_dir / "dataset_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
    return parquet_error


def main():
    parser = argparse.ArgumentParser(description="MP 带隙+稳定性数据集构建")
    here = Path(__file__).resolve().parent
    parser.add_argument("--out", default=str(here / "data" / "全库_形成能与带隙"))
    parser.add_argument("--chemsys", default=None, help="逗号分隔的化学体系，如 'Ba-O-Ti,Ba-Nb-O-Ti'")
    parser.add_argument("--batio3", action="store_true", help="只下载 BaTiO3 掺杂家族（31 个体系）")
    parser.add_argument("--all", action="store_true", help="全库模式（现为默认，保留兼容）")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--limit", type=int, default=None, help="调试：只处理前 N 条")
    parser.add_argument("--full-fields", action="store_true", help="请求 API 默认全字段")
    parser.add_argument("--calib-csv", default=None, help="HSE06/GW/实验校准表")
    parser.add_argument("--calib-gap-col", default="band_gap_eV")
    parser.add_argument("--calib-method-col", default="method")
    parser.add_argument("--calib-ref-col", default="reference")
    parser.add_argument("--force", action="store_true", help="忽略缓存重新抓取")
    parser.add_argument("--probe", action="store_true", help="只探测 API 字段")
    args = parser.parse_args()

    out_dir = Path(args.out)
    cache_dir = out_dir / "cache"
    query_date = utc_now()
    db_version = None

    with MPRester() as mpr:
        if args.probe:
            doc = probe_summary(mpr)
            if doc is None:
                print("探测失败：未返回文档")
                return 1
            keys = sorted(doc_keys(doc))
            print(f"materials.summary 支持字段 {len(keys)} 个：")
            for name in keys:
                print(f"  {name}")
            return 0

        doc = probe_summary(mpr)
        supported = doc_keys(doc) if doc is not None else set()
        fields, missing = pick_fields(supported, args.full_fields)
        requested = fields if fields is not None else "ALL(default)"
        if fields is None:
            print(f"summary 支持 {len(supported)} 个字段；本次请求: API 默认全字段")
        else:
            print(f"summary 支持 {len(supported)} 个字段；本次请求 {len(fields)} 个")
        if missing:
            print(f"API 不提供的候选字段（跳过）：{missing}")

        chemsys_list = build_chemsys_list(args)
        if chemsys_list is None:
            print("查询模式: 全库（预训练数据；首次会建立 mp-api 本地缓存，可能较慢）")
        elif args.batio3:
            print(f"查询模式: BaTiO3 掺杂家族（{len(chemsys_list)} 个化学体系）")
        else:
            print(f"查询模式: {len(chemsys_list)} 个化学体系")
        ids = collect_ids(mpr, chemsys_list, cache_dir, force=args.force)
        print(f"去重后 ID 总数: {len(ids)}")
        if args.limit:
            ids = ids[:args.limit]
            print(f"limit 生效: 仅处理前 {len(ids)} 条")
        fetch_summaries(mpr, ids, cache_dir, fields,
                        chunk_size=args.chunk_size, force=args.force)
        db_version = database_version(mpr)

    df, structures = build_frames(cache_dir / "summary", limit=args.limit)
    if df.empty:
        print("没有采集到数据；请检查参数、缓存或网络")
        return 1
    df = postprocess(df, query_date)
    if args.calib_csv:
        df = apply_calibration(df, args.calib_csv, args.calib_gap_col,
                               args.calib_method_col, args.calib_ref_col)
        print("已合并外部校准表")

    meta = {
        "dataset": "mp_summary_gap_stability",
        "created_utc": utc_now(),
        "query_date": query_date,
        "endpoint": "materials.summary",
        "mp_api_version": package_version("mp-api"),
        "pymatgen_version": package_version("pymatgen"),
        "pandas_version": package_version("pandas"),
        "mp_database_version": db_version,
        "query_scope": {
            "chemsys": chemsys_list if chemsys_list is not None else "ALL",
            "limit": args.limit,
            "chunk_size": args.chunk_size,
        },
        "fields_requested": requested,
        "fields_missing_from_api": missing,
        "calibration_csv": args.calib_csv,
        "notes": [
            "默认全库用于阶段一预训练；BaTiO3 掺杂子集（--batio3）用于阶段二微调，两者按 split_group 隔离",
            "band_gap 为 GGA/GGA+U 值（PBE 系），仅作定性初筛；高保真标签需 HSE06/GW/实验校准",
            "energy_above_hull=0 为热力学稳定；<=0.1 eV/atom 记为 metastable",
            "CBM/VBM 请用 mp_band_edges.py 补充；均为相对费米能级，非真空能级",
            "MP API 未暴露 Hubbard U 与自旋配置，未编造",
            "优先子集口径: preferred = 非废弃 + 有结构 + stable/metastable + 有带隙",
            "分组划分用 split_group (=reduced_formula)，避免同组成泄漏",
        ],
    }
    parquet_error = export_all(df, structures, out_dir, meta)
    print(f"完成: {len(df)} 条 -> {out_dir}")
    print("  materials.csv / materials.parquet / structures.jsonl / dataset_meta.json")
    if parquet_error:
        print(f"  Parquet 未写出: {parquet_error}")
    print(df["stability_class"].value_counts(dropna=False).to_string())
    print(f"preferred 子集: {int(df['preferred'].sum())} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
