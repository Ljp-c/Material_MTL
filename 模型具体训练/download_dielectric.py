"""下载 Materials Project 全量 DFPT 介电数据集 (dielectric endpoint) 及其晶体结构。

数据规模：MP 全库 dielectric 约 7332 条，字段 e_total / e_ionic / e_electronic / n。
dielectric 端点只支持按 material_ids 或 e_* 数值范围过滤，
所以结构需要再用 summary.search(material_ids=...) 分块拉取。

用法：
    python download_dielectric.py                 # 全量下载
    python download_dielectric.py --limit 100     # 冒烟测试
    python download_dielectric.py --labels-only   # 只下标签
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from pymatgen.core import Structure

DIELECTRIC_FIELDS = [
    "material_id", "formula_pretty", "nsites", "elements", "nelements", "chemsys",
    "density", "symmetry", "e_total", "e_ionic", "e_electronic", "n", "deprecated",
]
STRUCTURE_FIELDS = ["material_id", "structure"]


def get_field(obj, key, default=None):
    """兼容 mp-api 返回的 pydantic 对象与 dict。"""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def fetch_labels(limit: int | None = None) -> pd.DataFrame:
    from mp_api.client import MPRester

    print("[1/3] 拉取 dielectric 标签 ...")
    t0 = time.time()
    with MPRester() as mpr:
        docs = mpr.materials.dielectric.search(fields=DIELECTRIC_FIELDS)

    rows = []
    for d in docs:
        if get_field(d, "deprecated"):
            continue
        symmetry = get_field(d, "symmetry")
        elements = get_field(d, "elements") or []
        rows.append({
            "material_id": str(get_field(d, "material_id")),
            "formula": get_field(d, "formula_pretty"),
            "nsites": get_field(d, "nsites"),
            "nelements": get_field(d, "nelements"),
            "chemsys": get_field(d, "chemsys"),
            "elements": "+".join(sorted(str(getattr(e, "symbol", e)) for e in elements)),
            "density": get_field(d, "density"),
            "spacegroup_number": getattr(symmetry, "number", None),
            "spacegroup_symbol": getattr(symmetry, "symbol", None),
            "e_total": get_field(d, "e_total"),
            "e_ionic": get_field(d, "e_ionic"),
            "e_electronic": get_field(d, "e_electronic"),
            "n_refractive": get_field(d, "n"),
        })

    df = pd.DataFrame(rows).drop_duplicates("material_id").reset_index(drop=True)
    df = df[df["e_total"].notna()].reset_index(drop=True)
    if limit:
        df = df.head(int(limit)).reset_index(drop=True)
    print(f"      标签: {len(df)} 条 (耗时 {time.time() - t0:.1f}s)")
    return df


def fetch_structures(ids, out_pkl: Path, chunk_size: int = 400,
                     retries: int = 3) -> dict:
    from mp_api.client import MPRester

    cache: dict = {}
    if out_pkl.exists():
        with open(out_pkl, "rb") as f:
            cache = pickle.load(f)
        print(f"      已有结构缓存: {len(cache)} 条")

    todo = [i for i in ids if i not in cache]
    if not todo:
        return cache

    print(f"[2/3] 拉取结构: 需要 {len(todo)} 条 (chunk={chunk_size}) ...")
    t0 = time.time()
    with MPRester() as mpr:
        for start in tqdm(range(0, len(todo), chunk_size), desc="structures"):
            chunk = todo[start:start + chunk_size]
            docs = None
            for attempt in range(retries):
                try:
                    docs = mpr.materials.summary.search(
                        material_ids=chunk, fields=STRUCTURE_FIELDS)
                    break
                except Exception as exc:
                    print(f"      chunk {start} 第 {attempt + 1} 次失败: {exc}")
                    time.sleep(2 * (attempt + 1))
            if docs is None:
                continue
            for d in docs:
                mid = str(get_field(d, "material_id"))
                st = get_field(d, "structure")
                if st is None:
                    continue
                if isinstance(st, dict):
                    st = Structure.from_dict(st)
                cache[mid] = st
            with open(out_pkl, "wb") as f:
                pickle.dump(cache, f)
            time.sleep(0.1)

    print(f"      结构: {len(cache)} 条 (耗时 {time.time() - t0:.1f}s)")
    return cache


def main() -> int:
    parser = argparse.ArgumentParser(description="下载 MP DFPT 介电数据集 + 结构")
    default_out = Path(__file__).resolve().parent / "data" / "全库_介电"
    parser.add_argument("--out", default=str(default_out), help="输出目录")
    parser.add_argument("--limit", type=int, default=None, help="限制条数（冒烟测试用）")
    parser.add_argument("--chunk-size", type=int, default=400, help="结构拉取分块大小")
    parser.add_argument("--labels-only", action="store_true", help="只下载介电标签")
    parser.add_argument("--force", action="store_true", help="忽略已有标签缓存")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    labels_csv = out / "dielectric_labels.csv"

    if labels_csv.exists() and not args.force:
        df = pd.read_csv(labels_csv)
        if args.limit:
            df = df.head(int(args.limit)).reset_index(drop=True)
        print(f"[1/3] 复用已有标签: {len(df)} 条 -> {labels_csv}")
    else:
        df = fetch_labels(limit=args.limit)
        df.to_csv(labels_csv, index=False, encoding="utf-8-sig")
        print(f"      已保存: {labels_csv}")

    if args.labels_only:
        print("[3/3] 仅标签模式，结束。")
        return 0

    struct_pkl = out / "structures.pkl"
    structures = fetch_structures(df["material_id"].tolist(), struct_pkl,
                                  chunk_size=args.chunk_size)

    missing = [m for m in df["material_id"] if m not in structures]
    if missing:
        print(f"      警告: {len(missing)} 条标签没有对应结构，已丢弃")
        df = df[df["material_id"].isin(structures)].reset_index(drop=True)

    dataset_pkl = out / "dielectric_dataset.pkl"
    with open(dataset_pkl, "wb") as f:
        pickle.dump({"labels": df, "structures": {m: structures[m] for m in df["material_id"]}}, f)

    print("[3/3] 完成")
    print(f"      标签表 : {labels_csv}")
    print(f"      结构库 : {struct_pkl}")
    print(f"      合并集 : {dataset_pkl}")
    print(f"      样本数 : {len(df)}")
    print("      e_total 统计:")
    print(df["e_total"].describe().to_string())
    if df["e_ionic"].notna().any():
        print(f"      含 e_ionic 的样本: {int(df['e_ionic'].notna().sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
