r"""只读数据审计：标签覆盖、金属占比、空位位点标签、Ba 子集构成、batio3 口径、家族重叠、全局特征覆盖。

用法（工作目录 E:\Material_MTL）:
    python Models\audit_data.py
    python Models\audit_data.py --vacancy-shard-sample 1 --json Models\audit_report.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Composition

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import (A_SITE_ELEMENTS, B_SITE_ELEMENTS, GLOBAL_FEATURE_COLUMNS, Source,
                  formula_elements, load_family_groups, normalize_group)

SPLIT_CFG = {"val_frac": 0.0, "test_frac": 0.0, "seed": 0}


def quantiles(series, points=(0.0, 0.5, 0.9, 0.99, 1.0)):
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {}
    return {f"p{int(point * 100)}": round(float(values.quantile(point)), 4) for point in points}


def coverage(table, column):
    values = pd.to_numeric(table[column], errors="coerce")
    return {"finite": int(values.notna().sum()), "ratio": round(float(values.notna().mean()), 4)}


def audit_formation(source: Source, report: dict) -> None:
    table = source.table
    mids = table["material_id"].astype(str)
    groups = table["group"].astype(str)
    report["rows"] = int(len(table))
    report["unique_material_id"] = int(mids.nunique())
    report["duplicate_rows"] = int(len(table) - mids.nunique())
    report["duplicate_examples"] = sorted(set(mids[mids.duplicated(keep=False)]))[:5]
    duplicate_ids = mids[mids.duplicated(keep=False)]
    if len(duplicate_ids):
        subset = table[table["material_id"].astype(str).isin(set(duplicate_ids))]
        numeric = subset.set_index("material_id").select_dtypes(include=[np.number])
        per_mid = numeric.groupby(level=0).nunique()
        report["duplicate_mids_with_differing_values"] = int((per_mid > 1).any(axis=1).sum())
    else:
        report["duplicate_mids_with_differing_values"] = 0
    report["unique_groups"] = int(groups.nunique())
    report["max_group_size"] = int(groups.value_counts().iloc[0]) if len(table) else 0
    for column in ("formation_energy_per_atom_eV", "band_gap_eV", "cbm_eV", "vbm_eV", "gap_from_edges_eV"):
        report[column] = coverage(table, column)
    gap = pd.to_numeric(table["band_gap_eV"], errors="coerce").dropna()
    positive = gap[gap > 0]
    report["gap"] = {
        "metal_rows": int((gap == 0).sum()),
        "metal_ratio": round(float((gap == 0).mean()), 4),
        "positive_quantiles": {f"p{int(q * 100)}": round(float(positive.quantile(q)), 3)
                               for q in (0.0, 0.5, 0.9, 0.99, 1.0)},
    }
    cbm = pd.to_numeric(table["cbm_eV"], errors="coerce")
    vbm = pd.to_numeric(table["vbm_eV"], errors="coerce")
    gap_edges = pd.to_numeric(table["gap_from_edges_eV"], errors="coerce")
    band_gap = pd.to_numeric(table["band_gap_eV"], errors="coerce")
    report["cbm_finite_gap_zero_rows"] = int((cbm.notna() & band_gap.notna() & (band_gap == 0)).sum())
    mask = cbm.notna() & vbm.notna() & gap_edges.notna()
    if mask.any():
        residual = (gap_edges[mask] - (cbm[mask] - vbm[mask])).abs()
        report["gap_from_edges_vs_cbm_minus_vbm"] = {
            "n": int(mask.sum()),
            "median_abs_diff": round(float(residual.median()), 4),
            "ratio_le_0.05": round(float((residual <= 0.05).mean()), 4),
        }
    mask = cbm.notna() & vbm.notna() & band_gap.notna()
    if mask.any():
        residual = (band_gap[mask] - (cbm[mask] - vbm[mask])).abs()
        report["band_gap_vs_cbm_minus_vbm"] = {
            "n": int(mask.sum()),
            "median_abs_diff": round(float(residual.median()), 4),
            "ratio_le_0.05": round(float((residual <= 0.05).mean()), 4),
        }
    ehull = pd.to_numeric(table["energy_above_hull_eV_per_atom"], errors="coerce")
    report["ehull_negative_rows"] = int((ehull < -1e-6).sum())


def audit_vacancy(source: Source, report: dict, shard_sample: int) -> None:
    table = source.table
    report["rows"] = int(len(table))
    atoms = pd.to_numeric(table["n_atoms"], errors="coerce")
    report["n_atoms"] = {
        "min": float(atoms.min()), "p25": float(atoms.quantile(0.25)), "p50": float(atoms.quantile(0.5)),
        "p75": float(atoms.quantile(0.75)), "p95": float(atoms.quantile(0.95)), "max": float(atoms.max()),
        "le_1": int((atoms <= 1).sum()), "le_4": int((atoms <= 4).sum()),
    }
    site_mean = pd.to_numeric(table["site_vacancy_mean"], errors="coerce")
    valid = site_mean.notna().sum()
    report["site_label_coverage"] = round(float(site_mean.notna().mean()), 4)
    report["site_mean"] = {"negative_ratio": round(float((site_mean < 0).sum() / max(valid, 1)), 4),
                           **quantiles(site_mean)}
    formation = pd.to_numeric(table["formation_energy_per_atom"], errors="coerce")
    ehull = pd.to_numeric(table["ehull"], errors="coerce")
    both = formation.notna() & ehull.notna()
    identical = (formation - ehull).abs() <= 1e-6
    report["formation_vs_ehull"] = {
        "n_both": int(both.sum()),
        "identical_ratio": round(float(identical[both].mean()), 4) if both.any() else None,
    }
    report["formation_range"] = quantiles(formation)
    emace = pd.to_numeric(table["E_mace"], errors="coerce")
    energy = pd.to_numeric(table["energy_per_atom"], errors="coerce")
    if formation.notna().sum() > 10:
        report["formation_spearman"] = {
            "vs_ehull": round(float(formation.corr(ehull, method="spearman")), 4),
            "vs_E_mace": round(float(formation.corr(emace, method="spearman")), 4),
            "vs_energy_per_atom": round(float(formation.corr(energy, method="spearman")), 4),
        }
    stable = pd.to_numeric(table["stable"], errors="coerce")
    report["stable_ratio"] = round(float((stable > 0).mean()), 4)
    text = table["formula"].fillna(table["group"]).astype(str)
    ba_mask = text.str.contains("Ba", na=False)
    ba_text = text[ba_mask]
    elemental = 0
    oxo = 0
    chemsys: dict[str, int] = {}
    for value in ba_text:
        elements = formula_elements(value)
        if len(elements) == 1:
            elemental += 1
        if "O" in elements:
            oxo += 1
        key = "-".join(sorted(elements))
        chemsys[key] = chemsys.get(key, 0) + 1
    report["ba_subset"] = {
        "n": int(ba_mask.sum()),
        "single_element_ratio": round(elemental / max(len(ba_text), 1), 4),
        "contains_oxygen_ratio": round(oxo / max(len(ba_text), 1), 4),
        "top_chemsys": sorted(chemsys.items(), key=lambda item: -item[1])[:8],
    }
    if shard_sample > 0:
        site_values = []
        missing = 0
        for path in sorted(source.dir.glob("crystal_graph_part*.pkl"))[:shard_sample]:
            with open(path, "rb") as fh:
                crystal = pickle.load(fh)
            for sample in crystal.values():
                value = getattr(sample, "vacancy", None)
                if value is None:
                    missing += 1
                    continue
                tensor = value.reshape(-1)
                site_values.append(tensor[torch.isfinite(tensor)].numpy())
        if site_values:
            stacked = np.concatenate(site_values)
            report["shard_site_values"] = {
                "shards": shard_sample,
                "sites": int(stacked.size),
                "quantiles": {f"p{int(q * 100)}": round(float(np.quantile(stacked, q)), 3)
                              for q in (0.0, 0.5, 0.9, 0.99, 1.0)},
                "negative_ratio": round(float((stacked < 0).mean()), 4),
                "abs_gt_10_ratio": round(float((np.abs(stacked) > 10).mean()), 5),
                "samples_without_site_labels": missing,
            }


def audit_batio3(source: Source, report: dict) -> None:
    table = source.table
    report["rows"] = int(len(table))
    abo3 = 0
    strict = 0
    examples = []
    for record in table.to_dict("records"):
        text = record.get("group")
        if not (isinstance(text, str) and text.strip()):
            text = record.get("formula")
        if not (isinstance(text, str) and text.strip()):
            continue
        value = text
        try:
            amounts = Composition(value).get_el_amt_dict()
        except Exception:
            continue
        n_o = amounts.get("O", 0.0)
        a = sum(amounts.get(element, 0.0) for element in A_SITE_ELEMENTS)
        b = sum(amounts.get(element, 0.0) for element in B_SITE_ELEMENTS)
        ba = amounts.get("Ba", 0.0)
        ti = amounts.get("Ti", 0.0)
        is_abo3 = abs(n_o - 3.0 * a) < 1e-6 and abs(a - b) < 1e-6 and a > 0
        if is_abo3:
            abo3 += 1
        if is_abo3 and ba / max(a, 1e-9) >= 0.5 and ti / max(b, 1e-9) >= 0.5:
            strict += 1
        elif len(examples) < 8:
            examples.append({"formula": value, "O": n_o, "A": a, "B": b})
    report["abo3_rows"] = abo3
    report["strict_batio3_rows"] = strict
    report["non_abo3_examples"] = examples
    for column in ("formation_energy_per_atom_eV", "band_gap_eV", "energy_above_hull_eV_per_atom",
                   "e_total", "e_ionic", "e_electronic", "debye_temperature", "num_magnetic_sites"):
        if column in table.columns:
            report[f"{column}_coverage"] = int(pd.to_numeric(table[column], errors="coerce").notna().sum())


def audit_family(sources: dict, report: dict) -> None:
    family = load_family_groups()
    cache: dict[str, str | None] = {}

    def normalized(text):
        if text not in cache:
            cache[text] = normalize_group(text)
        return cache[text]

    report["family_groups_total"] = len(family)
    for name, source in sources.items():
        count = 0
        examples = []
        for record in source.records.values():
            if normalized(record["group"]) in family:
                count += 1
                if len(examples) < 5:
                    examples.append(record["group"])
        report[name] = {"samples_in_family": count, "examples": examples}


def audit_global_feat(sources: dict, report: dict) -> None:
    for name, source in sources.items():
        path = source.dir / "global_feat.csv"
        entry = {"exists": bool(path.exists())}
        if path.exists():
            frame = pd.read_csv(path)
            entry["rows"] = int(len(frame))
            entry["missing_columns"] = [c for c in GLOBAL_FEATURE_COLUMNS if c not in frame.columns]
            have = set(frame["material_id"].astype(str))
            missing = [mid for mid in source.records if mid not in have]
            entry["missing_in_index"] = len(missing)
            entry["missing_examples"] = missing[:5]
            numeric = frame[list(GLOBAL_FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
            entry["nan_rows"] = int((~np.isfinite(numeric)).any(axis=1).sum())
        report[name] = entry


def main() -> int:
    parser = argparse.ArgumentParser(description="只读数据审计（不修改任何数据）")
    parser.add_argument("--vacancy-shard-sample", type=int, default=0, help="额外抽 N 个 vacancy 分片统计位点级标签")
    parser.add_argument("--json", default=None, help="把审计报告写入 JSON 文件")
    args = parser.parse_args()

    sources = {
        "formation_energy_band_gap": Source("formation_energy_band_gap", SPLIT_CFG),
        "vacancy_screening": Source("vacancy_screening", SPLIT_CFG),
        "batio3": Source("batio3", SPLIT_CFG),
        "batio3_doped": Source("batio3_doped", SPLIT_CFG),
    }
    print("[1/5] formation_energy_band_gap", flush=True)
    report = {"formation_energy_band_gap": {}}
    audit_formation(sources["formation_energy_band_gap"], report["formation_energy_band_gap"])
    print("[2/5] vacancy_screening", flush=True)
    report["vacancy_screening"] = {}
    audit_vacancy(sources["vacancy_screening"], report["vacancy_screening"], args.vacancy_shard_sample)
    print("[3/5] batio3", flush=True)
    report["batio3"] = {}
    audit_batio3(sources["batio3"], report["batio3"])
    print("[4/5] family overlap", flush=True)
    report["family"] = {}
    audit_family(sources, report["family"])
    print("[5/5] global_feat coverage", flush=True)
    report["global_feat"] = {}
    audit_global_feat(sources, report["global_feat"])

    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.json:
        out = Path(args.json)
        if not out.is_absolute():
            out = Path(__file__).resolve().parent / out
        out.write_text(text, encoding="utf-8")
        print(f"[out] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
