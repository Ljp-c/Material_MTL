"""审计并整理 模型具体训练/data 下的已有数据集（铁电 + BaTiO3 掺杂）。

输入:
    data/铁电材料_极化/labels.csv / records.jsonl / structures.pkl / cif/
    data/BaTiO3掺杂_结构_形成能_凸包能量/*.pkl
输出（整理版并入各数据集目录）:
    data/铁电材料_极化/audit.json + {labels_unified, records_fields, structure_index}.csv/.parquet
    data/铁电材料_极化/structures.jsonl（可用 --no-structure-json 关闭）
    data/BaTiO3掺杂_结构_形成能_凸包能量/audit.json + export_summary.json
    data/BaTiO3掺杂_结构_形成能_凸包能量/<tag>.csv/.parquet + <tag>_structure_index.* + <tag>_structures.jsonl

用法:
    python organize_existing.py --audit-only
    python organize_existing.py --fast
    python organize_existing.py
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from pymatgen.core import Composition

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FERRO_DIR = DATA_DIR / "铁电材料_极化"
BTO_DIR = DATA_DIR / "BaTiO3掺杂_结构_形成能_凸包能量"

NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
SAFE_RE = re.compile(r"[^A-Za-z0-9_\-]+")

POLAR_POINT_GROUPS = {"1", "2", "m", "mm2", "4", "4mm", "3", "3m", "6", "6mm"}

SOURCE_LIB = {
    "ferroelectrics": "SciData2020 (doi:10.1038/s41597-020-0407-9)",
    "ferroelectrics_ext": "npj2023 (doi:10.1038/s41524-023-01193-3)",
}

RECORD_FIELD_MAP = {
    "polarization.norm": "polarization_norm_uC_cm2",
    "polarization": "polarization_c_axis_uC_cm2",
    "bandgap.polar": "bandgap_polar_eV",
    "bandgap.nonpolar": "bandgap_nonpolar_eV",
    "energy|diff": "energy_diff_eV_per_atom",
    "distortion.dmax": "distortion_dmax_A",
    "distortion.dmax.before": "distortion_dmax_before_A",
    "distortion.dmax.after": "distortion_dmax_after_A",
    "distortion.delta": "distortion_delta",
    "distortion.s": "distortion_s",
    "distortion.dav": "distortion_dav_A",
    "polarizations.smoothness.index": "pol_smoothness_index",
    "polarizations.smoothness.max": "pol_smoothness_max_uC_cm2",
    "polarizations.jumps.max": "pol_jumps_max_uC_cm2",
    "polarizations.jumps.index": "pol_jumps_index",
    "energies.smoothness": "energies_smoothness_eV_per_atom",
    "energies.jumps|max": "energies_jumps_max_eV_per_atom",
    "polarization.quanta.a": "pol_quanta_a_uC_cm2",
    "polarization.quanta.b": "pol_quanta_b_uC_cm2",
    "polarization.quanta.c": "pol_quanta_c_uC_cm2",
    "polarization.vector.a": "pol_vector_a_uC_cm2",
    "polarization.vector.b": "pol_vector_b_uC_cm2",
    "polarization.vector.c": "pol_vector_c_uC_cm2",
    "workflow.status": "workflow_status",
    "workflow.category": "workflow_category",
    "polar.mpid": "polar_mpid",
    "nonpolar.mpid": "nonpolar_mpid",
    "bilbao.spacegroup.polar": "bilbao_sg_polar",
    "bilbao.spacegroup.nonpolar": "bilbao_sg_nonpolar",
    "polar.spacegroup": "polar_spacegroup_record",
    "nonpolar.spacegroup": "nonpolar_spacegroup_record",
    "workflow.id|search": "workflow_id_search",
    "id|search": "id_search",
    "distance": "distance",
}

STRING_TARGETS = {"workflow_status", "workflow_category", "polar_mpid", "nonpolar_mpid"}


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_number(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = NUM_RE.search(str(value))
    return float(match.group()) if match else None


def reduced_formula_of(formula):
    if not formula:
        return None
    try:
        return Composition(formula).reduced_formula
    except Exception:
        return None


def is_missing(value):
    if value is None:
        return True
    try:
        return value != value
    except Exception:
        return False


def structure_row(structure, key, source, fast=False):
    row = {
        "key": key,
        "source": source,
        "n_sites": len(structure),
        "formula": structure.composition.formula,
        "reduced_formula": structure.composition.reduced_formula,
        "a_A": structure.lattice.a,
        "b_A": structure.lattice.b,
        "c_A": structure.lattice.c,
        "alpha_deg": structure.lattice.alpha,
        "beta_deg": structure.lattice.beta,
        "gamma_deg": structure.lattice.gamma,
        "volume_A3": structure.volume,
        "density_g_cm3": structure.density,
    }
    if fast:
        return row
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    try:
        analyzer = SpacegroupAnalyzer(structure, symprec=1e-3)
        point_group = analyzer.get_point_group_symbol()
        row["spacegroup_number"] = analyzer.get_space_group_number()
        row["spacegroup_symbol"] = analyzer.get_space_group_symbol()
        row["point_group"] = point_group
        row["is_polar_pointgroup"] = point_group in POLAR_POINT_GROUPS
        row["symmetry_error"] = None
    except Exception as exc:
        row["spacegroup_number"] = None
        row["spacegroup_symbol"] = None
        row["point_group"] = None
        row["is_polar_pointgroup"] = None
        row["symmetry_error"] = str(exc)
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


def audit_ferro():
    info = {}
    labels_path = FERRO_DIR / "labels.csv"
    if labels_path.exists():
        labels = pd.read_csv(labels_path)
        info["labels"] = {
            "path": str(labels_path),
            "rows": int(len(labels)),
            "columns": list(labels.columns),
            "per_project": {str(k): int(v) for k, v in labels["project"].value_counts().items()},
            "duplicated_keys": int(labels["key"].duplicated().sum()),
            "per_column_nonnull": {c: int(labels[c].notna().sum()) for c in labels.columns},
        }
    records_path = FERRO_DIR / "records.jsonl"
    if records_path.exists():
        field_union = set()
        per_project = {}
        n_records = 0
        with open(records_path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                n_records += 1
                project = str(record.get("key", "")).split("::")[0]
                per_project[project] = per_project.get(project, 0) + 1
                field_union |= set(record.get("data", {}).keys())
        info["records"] = {
            "path": str(records_path),
            "rows": n_records,
            "per_project": per_project,
            "n_data_fields": len(field_union),
            "data_fields": sorted(field_union),
        }
    struct_path = FERRO_DIR / "structures.pkl"
    if struct_path.exists():
        with open(struct_path, "rb") as handle:
            structures = pickle.load(handle)
        keys = list(structures.keys())
        per_project_struct = {}
        for key in keys:
            project = str(key).split("::")[0]
            per_project_struct[project] = per_project_struct.get(project, 0) + 1
        sample = structures[keys[0]] if keys else None
        info["structures"] = {
            "path": str(struct_path),
            "n": len(structures),
            "per_project": per_project_struct,
            "sample_keys": [str(k) for k in keys[:3]],
            "value_types": sorted({type(v).__name__ for v in structures.values()}),
            "sample_n_sites": len(sample) if sample is not None else None,
            "sample_formula": sample.composition.reduced_formula if sample is not None else None,
        }
    cif_dir = FERRO_DIR / "cif"
    info["cif"] = {
        "dir": str(cif_dir),
        "n_files": len(list(cif_dir.glob("*.cif"))) if cif_dir.exists() else 0,
    }
    return info


def audit_batio3():
    info = {}
    if not BTO_DIR.exists():
        return info
    for path in sorted(BTO_DIR.glob("*.pkl")):
        entry = {"path": str(path), "size_bytes": path.stat().st_size}
        try:
            with open(path, "rb") as handle:
                obj = pickle.load(handle)
        except Exception as exc:
            entry["error"] = str(exc)
            info[path.name] = entry
            continue
        if isinstance(obj, pd.DataFrame):
            entry["kind"] = "DataFrame"
            entry["rows"] = int(obj.shape[0])
            entry["columns"] = list(obj.columns)
            entry["columns_dtype"] = {c: str(obj[c].dtype) for c in obj.columns}
            if "structure" in obj.columns:
                nonnull = obj["structure"].dropna()
                entry["n_structures"] = int(len(nonnull))
                if len(nonnull):
                    entry["structure_type"] = type(nonnull.iloc[0]).__name__
        elif isinstance(obj, dict):
            entry["kind"] = "dict"
            entry["n_keys"] = len(obj)
            entry["sample_keys"] = [str(k) for k in list(obj.keys())[:5]]
        else:
            entry["kind"] = type(obj).__name__
        info[path.name] = entry
    return info


def export_ferro_labels(outdir):
    labels = pd.read_csv(FERRO_DIR / "labels.csv")
    df = pd.DataFrame()
    df["key"] = labels["key"]
    df["project"] = labels["project"]
    df["source_lib"] = labels["project"].map(SOURCE_LIB).fillna("unknown")
    df["identifier"] = labels["identifier"]
    df["formula"] = labels["formula"]
    df["reduced_formula"] = [reduced_formula_of(f) for f in labels["formula"]]
    df["split_group"] = df["reduced_formula"].fillna(df["formula"])
    df["polarization_key"] = labels["polarization_key"]
    df["polarization_uC_cm2"] = pd.to_numeric(labels["polarization_uC_cm2"], errors="coerce")
    df["polarization_raw"] = labels["polarization_raw"]
    df["polar_spacegroup"] = labels["polar_spacegroup"].fillna(
        labels["Polar_spacegroup"]).map(lambda v: None if pd.isna(v) else str(v))
    df["nonpolar_spacegroup"] = labels["nonpolar_spacegroup"].fillna(
        labels["Nonpolar_spacegroup"]).map(lambda v: None if pd.isna(v) else str(v))
    df["bandgap_polar_eV"] = pd.to_numeric(
        labels["bandgap_polar"].fillna(labels["Polar_bandgap"]), errors="coerce")
    df["bandgap_nonpolar_eV"] = pd.to_numeric(
        labels["bandgap_nonpolar"].fillna(labels["Nonpolar_bandgap"]), errors="coerce")
    df["energy_diff_eV_per_atom"] = pd.to_numeric(labels["energy_diff"], errors="coerce")
    df["energy_diff_ext_raw"] = pd.to_numeric(labels["Energy_diff"], errors="coerce")
    df["workflow_status"] = labels["workflow_status"]
    df["workflow_category"] = labels["workflow_category"]
    for column in ("CL_score", "MAD_pseudo", "MAD_relax", "F_score"):
        df[column] = pd.to_numeric(labels[column], errors="coerce")
    return df


def export_ferro_records_fields():
    rows = []
    with open(FERRO_DIR / "records.jsonl", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            data = record.get("data", {})
            key = record.get("key", "")
            row = {"key": key, "project": str(key).split("::")[0] if key else None}
            for source, target in RECORD_FIELD_MAP.items():
                raw = data.get(source)
                if target in STRING_TARGETS:
                    row[target] = None if raw is None else str(raw)
                else:
                    row[target] = parse_number(raw)
            rows.append(row)
    return pd.DataFrame(rows)


def export_ferro_structures(outdir, fast=False, export_json=True):
    with open(FERRO_DIR / "structures.pkl", "rb") as handle:
        structures = pickle.load(handle)
    index_rows = []
    json_handle = None
    if export_json:
        json_handle = open(outdir / "structures.jsonl", "w", encoding="utf-8")
    try:
        for key, structure in tqdm(structures.items(), desc="ferro structures"):
            parts = str(key).split("::", 2)
            project = parts[0] if len(parts) > 0 else None
            identifier = parts[1] if len(parts) > 1 else None
            name = parts[2] if len(parts) > 2 else None
            row = structure_row(structure, str(key), "ferroelectrics", fast=fast)
            row["project"] = project
            row["identifier"] = identifier
            row["structure_name"] = name
            index_rows.append(row)
            if json_handle is not None:
                payload = {
                    "key": str(key),
                    "project": project,
                    "identifier": identifier,
                    "structure_name": name,
                    "structure": structure.as_dict() if hasattr(structure, "as_dict") else structure,
                }
                json_handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    finally:
        if json_handle is not None:
            json_handle.close()
    return pd.DataFrame(index_rows)


def export_batio3(outdir, fast=False, export_json=True):
    outdir.mkdir(parents=True, exist_ok=True)
    results = {}
    for path in sorted(BTO_DIR.glob("*.pkl")):
        tag = SAFE_RE.sub("_", path.stem).strip("_")
        with open(path, "rb") as handle:
            obj = pickle.load(handle)
        entry = {"path": str(path), "tag": tag}
        if not isinstance(obj, pd.DataFrame):
            entry["kind"] = type(obj).__name__
            entry["exported"] = False
            results[path.name] = entry
            continue
        df = obj.copy()
        struct_series = None
        if "structure" in df.columns:
            struct_series = df["structure"]
            df = df.drop(columns=["structure"])
        save_table(df, outdir / tag)
        n_structures = 0 if struct_series is None else int(struct_series.notna().sum())
        entry.update({
            "kind": "DataFrame",
            "rows": int(obj.shape[0]),
            "columns": list(obj.columns),
            "n_structures": n_structures,
            "exported": True,
        })
        index_rows = []
        if struct_series is not None:
            json_handle = None
            if export_json:
                json_handle = open(outdir / f"{tag}_structures.jsonl", "w", encoding="utf-8")
            try:
                for idx, structure in struct_series.items():
                    if is_missing(structure):
                        continue
                    mid = df.at[idx, "material_id"] if "material_id" in df.columns else None
                    if is_missing(mid):
                        mid = None
                    row = structure_row(structure, str(mid), "batio3", fast=fast)
                    index_rows.append(row)
                    if json_handle is not None:
                        json_handle.write(json.dumps({
                            "material_id": None if mid is None else str(mid),
                            "structure": structure.as_dict() if hasattr(structure, "as_dict") else structure,
                        }, ensure_ascii=False, default=str) + "\n")
            finally:
                if json_handle is not None:
                    json_handle.close()
        if index_rows:
            save_table(pd.DataFrame(index_rows), outdir / f"{tag}_structure_index")
        results[path.name] = entry
    return results


def main():
    parser = argparse.ArgumentParser(description="整理 data/ 下已有数据集")
    parser.add_argument("--audit-only", action="store_true", help="只审计不导出")
    parser.add_argument("--fast", action="store_true", help="跳过空间群/点群分析")
    parser.add_argument("--no-structure-json", action="store_true", help="不导出结构 JSONL")
    args = parser.parse_args()

    report = {
        "generated_utc": utc_now(),
        "operator_note": "本地整理产物；原始数据未改动",
        "ferroelectrics": audit_ferro(),
        "batio3": audit_batio3(),
    }
    common = {
        "generated_utc": report["generated_utc"],
        "operator_note": report["operator_note"],
    }
    (FERRO_DIR / "audit.json").write_text(
        json.dumps({**common, "ferroelectrics": report["ferroelectrics"]},
                   ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (BTO_DIR / "audit.json").write_text(
        json.dumps({**common, "batio3": report["batio3"]},
                   ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("=" * 70)
    print("审计摘要")
    print("=" * 70)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    if args.audit_only:
        print(f"\n审计完成 -> {FERRO_DIR / 'audit.json'} 与 {BTO_DIR / 'audit.json'}")
        return 0

    labels = export_ferro_labels(FERRO_DIR)
    parquet_error = save_table(labels, FERRO_DIR / "labels_unified")
    if parquet_error:
        print(f"labels_unified.parquet 未写出: {parquet_error}")
    records = export_ferro_records_fields()
    parquet_error = save_table(records, FERRO_DIR / "records_fields")
    if parquet_error:
        print(f"records_fields.parquet 未写出: {parquet_error}")
    index = export_ferro_structures(FERRO_DIR, fast=args.fast, export_json=not args.no_structure_json)
    parquet_error = save_table(index, FERRO_DIR / "structure_index")
    if parquet_error:
        print(f"structure_index.parquet 未写出: {parquet_error}")
    print(f"ferroelectrics: labels {labels.shape} | records {records.shape} | structures {index.shape}")

    results = export_batio3(BTO_DIR, fast=args.fast, export_json=not args.no_structure_json)
    (BTO_DIR / "export_summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"完成 -> {FERRO_DIR.name} 与 {BTO_DIR.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
