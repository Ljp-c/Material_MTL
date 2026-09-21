"""下载 MPContribs 铁电数据库（带自发极化的材料 + 结构）。

数据来源:
    ferroelectrics       Sci Data 2020,  doi:10.1038/s41597-020-0407-9   (255 条有极化)
    ferroelectrics_ext   npj Comput Mater 2023, doi:10.1038/s41524-023-01193-3  (386 条有极化)

每个材料包含极化/非极化结构对，以及 DFT 极化、带隙、空间群等数值。
注意: 极化值单位 µC/cm²，字段里带单位字符串，脚本会自动解析成数字。

用法:
    python download_ferroelectrics_db.py                # 全量
    python download_ferroelectrics_db.py --limit 3      # 冒烟测试
    python download_ferroelectrics_db.py --with-attachments
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import time
from pathlib import Path

from mp_api.client import MPRester
from tqdm import tqdm

PROJECTS = ["ferroelectrics", "ferroelectrics_ext"]
NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_number(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = NUM_RE.search(str(value))
    return float(match.group()) if match else None


def find_polarization(data: dict):
    preferred = ["polarization.norm", "Polarization", "polarization"]
    for name in preferred:
        for key, value in data.items():
            if key.lower() == name.lower():
                number = parse_number(value)
                if number is not None:
                    return key, number, value
    skip = ("vector", "quant", "smooth", "jump", "index")
    for key, value in data.items():
        low = key.lower()
        if "polarization" in low and not any(s in low for s in skip):
            number = parse_number(value)
            if number is not None:
                return key, number, value
    return None, None, None


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.\-]+", "_", str(text))


def plain(value):
    if value is None or isinstance(value, (int, float, str, bool)):
        return value
    for attr in ("value", "magnitude"):
        if hasattr(value, attr):
            try:
                return float(getattr(value, attr))
            except Exception:
                break
    return str(value)


def load_done(labels_csv: Path) -> set:
    if not labels_csv.exists():
        return set()
    done = set()
    try:
        import pandas as pd
        df = pd.read_csv(labels_csv)
        done = set(df["key"].astype(str))
    except Exception:
        pass
    return done


def download_project(contribs, project: str, outdir: Path, limit=None,
                     force=False, with_attachments=False, logger=print):
    cif_dir = outdir / "cif"
    cif_dir.mkdir(parents=True, exist_ok=True)
    labels_csv = outdir / "labels.csv"
    struct_pkl = outdir / "structures.pkl"
    records_jsonl = outdir / "records.jsonl"
    done = set() if force else load_done(labels_csv)

    logger(f"[{project}] 查询贡献条目 ...")
    result = contribs.query_contributions(
        query={"project": project},
        fields=["identifier", "formula", "data", "structures", "attachments"],
        paginate=True,
    )
    recs = result["data"]
    logger(f"[{project}] 共 {len(recs)} 条，筛出带极化值的 ...")

    targets = []
    for rec in recs:
        key_name, value, raw = find_polarization(rec.get("data", {}))
        if value is None:
            continue
        targets.append((rec, key_name, value, raw))
    if limit:
        targets = targets[: int(limit)]
    logger(f"[{project}] 带极化值: {len(targets)} 条")

    if struct_pkl.exists() and not force:
        with open(struct_pkl, "rb") as f:
            structures = pickle.load(f)
    else:
        structures = {}

    rows = []
    records_fh = open(records_jsonl, "a", encoding="utf-8")
    t0 = time.time()

    for idx, (rec, key_name, value, raw) in enumerate(tqdm(targets, desc=project)):
        key = f"{project}::{rec['identifier']}"
        struct_keys = [f"{project}::{rec['identifier']}::{s['name']}"
                       for s in rec.get("structures", [])]
        if key in done and all(k in structures for k in struct_keys):
            continue
        data = rec.get("data", {})
        row = {
            "key": key,
            "project": project,
            "identifier": rec["identifier"],
            "formula": rec["formula"],
            "polarization_key": key_name,
            "polarization_uC_cm2": value,
            "polarization_raw": str(raw),
        }
        for field in ["polar.spacegroup", "nonpolar.spacegroup", "Polar.spacegroup",
                      "Nonpolar.spacegroup", "bandgap.polar", "bandgap.nonpolar",
                      "Polar.bandgap", "Nonpolar.bandgap", "energy|diff", "Energy|diff",
                      "workflow.status", "workflow.category", "F|score", "CL|score",
                      "MAD.pseudo", "MAD.relax"]:
            if field in data:
                row[field.replace("|", "_").replace(".", "_")] = plain(data[field])

        for s in rec.get("structures", []):
            ssid = s["id"]
            sname = s["name"]
            try:
                st = contribs.get_structure(ssid)
            except Exception as exc:
                logger(f"   结构获取失败 {rec['identifier']}/{sname}: {exc}")
                continue
            structures[f"{project}::{rec['identifier']}::{sname}"] = st
            cif_path = cif_dir / f"{safe_name(project)}__{safe_name(rec['identifier'])}__{safe_name(sname)}.cif"
            try:
                st.to(filename=str(cif_path))
            except Exception as exc:
                logger(f"   CIF 写出失败 {cif_path.name}: {exc}")

        records_fh.write(json.dumps({"key": key, "data": data,
                                     "structures": [s["name"] for s in rec.get("structures", [])]},
                                    ensure_ascii=False, default=str) + "\n")
        records_fh.flush()
        rows.append(row)

        if with_attachments:
            att_dir = outdir / "attachments"
            att_dir.mkdir(parents=True, exist_ok=True)
            for a in rec.get("attachments", []):
                try:
                    att = contribs.get_attachment(a["id"])
                    att.write(att_dir)
                except Exception as exc:
                    logger(f"   附件获取失败 {rec['identifier']}/{a.get('name')}: {exc}")

        if idx % 25 == 0:
            with open(struct_pkl, "wb") as f:
                pickle.dump(structures, f)

    records_fh.close()
    with open(struct_pkl, "wb") as f:
        pickle.dump(structures, f)

    import pandas as pd
    new_df = pd.DataFrame(rows)
    if labels_csv.exists() and not force:
        old = pd.read_csv(labels_csv)
        new_df = pd.concat([old, new_df], ignore_index=True).drop_duplicates("key")
    new_df.to_csv(labels_csv, index=False, encoding="utf-8-sig")

    logger(f"[{project}] 完成: 新增 {len(rows)} 条 | 结构总数 {len(structures)} | "
           f"耗时 {time.time() - t0:.1f}s")
    logger(f"[{project}] 产物: {labels_csv} | {struct_pkl} | {cif_dir}")
    return new_df


def main() -> int:
    here = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="下载 MPContribs 铁电数据库")
    parser.add_argument("--out", default=str(here / "data" / "铁电材料_极化"))
    parser.add_argument("--projects", nargs="*", default=PROJECTS)
    parser.add_argument("--limit", type=int, default=None, help="每个项目限制条数（冒烟测试）")
    parser.add_argument("--force", action="store_true", help="忽略已有进度，重新下载")
    parser.add_argument("--with-attachments", action="store_true",
                        help="同时下载附件(distortion/workflow json.gz，较慢)")
    args = parser.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    with MPRester() as mpr:
        contribs = mpr.contribs
        for project in args.projects:
            download_project(contribs, project, outdir, limit=args.limit,
                             force=args.force, with_attachments=args.with_attachments)

    try:
        import pandas as pd
        df = pd.read_csv(outdir / "labels.csv")
        print("\n===== 汇总 =====")
        print(df.groupby("project")["polarization_uC_cm2"].describe().to_string())
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
