r"""分析 predict.py 导出的逐样本 CSV：定位误差集中区间、最差样本与化学体系。

用法（工作目录 E:\Material_MTL）:
    python Models\analyze_preds.py --csv Models\artifacts\pretrain\preds_test.csv
    python Models\analyze_preds.py --csv ... --target gap --plot Models\artifacts\gap_scatter.png
    python Models\analyze_preds.py --csv ... --target vacancy --top 20

判读：某区间 MAE 高且"平方误差占比"大，才是真正的误差源；MAE 高但样本极少的桶影响有限。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import SOURCE_REGISTRY, formula_elements

GAP_EDGES = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0, float("inf")]
VACANCY_EDGES = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, float("inf")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="逐样本误差分析（区间分桶/最差样本/化学体系）")
    parser.add_argument("--csv", required=True, help="predict.py 输出的逐样本 CSV")
    parser.add_argument("--target", default="gap", choices=["formation", "gap", "cbm", "vbm", "vacancy"])
    parser.add_argument("--top", type=int, default=15, help="最差样本展示条数")
    parser.add_argument("--plot", default=None, help="保存 pred-vs-true 散点图的路径（可选）")
    return parser.parse_args()


def build_formula_map() -> dict:
    mapping: dict = {}
    for spec in SOURCE_REGISTRY.values():
        path = spec["dir"] / "index.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, usecols=lambda c: c in {"material_id", "formula", "group"})
        for record in frame.to_dict("records"):
            mid = str(record.get("material_id"))
            if mid in mapping:
                continue
            formula = record.get("formula")
            group = record.get("group")
            if isinstance(formula, str) and formula.strip():
                mapping[mid] = formula
            elif isinstance(group, str) and group.strip():
                mapping[mid] = group
    return mapping


def bucket_table(sub: pd.DataFrame, basis: pd.Series, edges: list) -> None:
    sse_total = float(((sub["pred"] - sub["true"]) ** 2).sum())
    n_total = len(sub)
    print(f"  {'区间':>14s}  {'n':>7s}  {'样本占比':>8s}  {'MAE':>8s}  {'RMSE':>8s}  {'平方误差占比':>12s}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (basis >= lo) & (basis < hi)
        count = int(mask.sum())
        if count == 0:
            continue
        err = sub.loc[mask, "pred"] - sub.loc[mask, "true"]
        sse = float((err ** 2).sum())
        label = f"[{lo:g},{hi:g})"
        print(f"  {label:>14s}  {count:>7d}  {count / n_total * 100:>7.1f}%  "
              f"{err.abs().mean():>8.4f}  {np.sqrt((err ** 2).mean()):>8.4f}  {sse / sse_total * 100:>11.1f}%")


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"找不到 CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {"source", "material_id", "target", "true", "pred"}
    if not required <= set(df.columns):
        raise SystemExit(f"CSV 缺列，需要 {sorted(required)}")

    print("=== 按来源×目标 ===")
    for (source_name, target), sub in df.groupby(["source", "target"]):
        err = sub["pred"] - sub["true"]
        ss_tot = float(((sub["true"] - sub["true"].mean()) ** 2).sum())
        r2 = 1.0 - float((err ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")
        print(f"  {source_name:26s} {target:10s} n={len(sub):7d} MAE={err.abs().mean():.4f} "
              f"RMSE={np.sqrt((err ** 2).mean()):.4f} R2={r2:.3f}")

    target = args.target
    sub = df[df["target"] == target].copy()
    if sub.empty:
        print(f"\nCSV 中没有 target={target} 的行")
        return 0

    print(f"\n=== {target} 区间分桶（按真实值{'绝对值' if target == 'vacancy' else ''}） ===")
    if target == "gap":
        bucket_table(sub, sub["true"], GAP_EDGES)
    elif target == "vacancy":
        bucket_table(sub, sub["true"].abs(), VACANCY_EDGES)
    else:
        edges = list(np.unique(np.quantile(sub["true"], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])))
        bucket_table(sub, sub["true"], edges)

    if target == "gap":
        low_true = sub[sub["true"] < 0.5]
        low_pred = sub[sub["pred"] < 0.5]
        slope = float(np.polyfit(sub["true"], sub["pred"], 1)[0]) if len(sub) > 2 else float("nan")
        print("\n=== gap 压缩诊断 ===")
        print(f"  真实<0.5 eV 的样本 n={len(low_true)}，其平均预测={low_true['pred'].mean():.3f}（越接近 0.5 越像'向均值回归'）")
        print(f"  预测<0.5 eV 的样本 n={len(low_pred)}，其平均真实={low_pred['true'].mean():.3f}")
        print(f"  pred ~ true 线性斜率 = {slope:.3f}（<1 表示预测被压缩）")

    print(f"\n=== {target} 最差 {args.top} 条 ===")
    worst = sub.sort_values("abs_error", ascending=False).head(args.top)
    show = worst[["source", "material_id", "true", "pred", "residual"]].copy()
    print(show.to_string(index=False))

    formula_map = build_formula_map()
    elements = sub["material_id"].astype(str).map(
        lambda mid: "-".join(sorted(formula_elements(formula_map.get(mid, "")))) if formula_map.get(mid) else "unknown")
    sub = sub.assign(elements=elements)
    sse_total = float(((sub["pred"] - sub["true"]) ** 2).sum())
    grouped = []
    for key, part in sub.groupby("elements"):
        sse = float(((part["pred"] - part["true"]) ** 2).sum())
        grouped.append((key, len(part), sse))
    grouped.sort(key=lambda item: -item[2])
    print(f"\n=== {target} 化学体系（按平方误差贡献 Top 10） ===")
    for key, count, sse in grouped[:10]:
        print(f"  {key:24s} n={count:7d} 平方误差占比={sse / sse_total * 100:5.1f}%")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=150)
        ax.scatter(sub["true"], sub["pred"], s=4, alpha=0.25, edgecolors="none")
        low = float(sub["true"].min())
        high = float(sub["true"].max())
        ax.plot([low, high], [low, high], color="tab:red", lw=1.0, ls="--")
        basis = sub["true"].abs() if target == "vacancy" else sub["true"]
        edges = VACANCY_EDGES if target == "vacancy" else GAP_EDGES
        centers, means = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (basis >= lo) & (basis < hi)
            if int(mask.sum()) >= 5:
                centers.append(lo + 0.25 if not np.isfinite(hi) else (lo + hi) / 2)
                means.append(float(sub.loc[mask, "pred"].mean()))
        ax.plot(centers, means, marker="o", color="tab:blue", lw=1.2)
        err = sub["pred"] - sub["true"]
        ax.set_xlabel("true (eV)")
        ax.set_ylabel("pred (eV)")
        ax.set_title(f"{target}: pred vs true  n={len(sub)}  MAE={err.abs().mean():.3f} eV")
        out = Path(args.plot)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out)
        print(f"\n[plot] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
