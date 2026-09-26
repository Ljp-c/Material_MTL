r"""从 run 目录读取 metrics.csv / losses.csv 画 epoch 曲线（支持多 run 对比叠加）。

用法（工作目录 E:\Material_MTL）:
    python Models\plot_run.py --run pretrain=Models\artifacts\pretrain
    python Models\plot_run.py --run metal=Models\artifacts\pretrain_metal --run ctrl=Models\artifacts\pretrain_metal_ctrl --out Models\artifacts\pictures\metal_test
    （--run 可重复；不写 --out 时默认存到 Models\artifacts\pictures\<第一个 run 名>）

输出：验证点曲线 val_mae.png / val_rmse.png / val_r2.png（按目标 2×3 子图）、val_loss_terms.png；
      逐 step 训练曲线 train_loss.png / train_terms.png / train_speed.png（原始细线 + 滑动平均）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

TARGETS = ("formation", "gap", "cbm", "vbm", "vacancy")
TERMS = ("formation", "gap", "cbm", "vbm", "vacancy", "consistency")
UNITS = {"formation": "eV/atom", "gap": "eV", "cbm": "eV", "vbm": "eV", "vacancy": "eV"}


def read_metrics(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"{run_dir} 缺 metrics.csv")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"{path} 为空")
    return frame


def read_losses(run_dir: Path) -> pd.DataFrame | None:
    path = run_dir / "losses.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    return None if frame.empty else frame


def read_step_log(run_dir: Path) -> pd.DataFrame | None:
    path = run_dir / "step_log.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    return None if frame.empty else frame


def plot_runs(runs, out_dir, dpi: int = 150, smooth: int = 50) -> list:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {label: read_metrics(Path(path)) for label, path in runs}
    losses = {label: read_losses(Path(path)) for label, path in runs}
    steps = {label: read_step_log(Path(path)) for label, path in runs}
    saved: list = []

    def val_rows(frame: pd.DataFrame) -> pd.DataFrame:
        return frame[(frame["split"] == "val") & (frame["source"] == "__overall__")]

    for key, filename, ylabel in (("mae", "val_mae.png", "MAE"),
                                  ("rmse", "val_rmse.png", "RMSE"),
                                  ("r2", "val_r2.png", "R2")):
        fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), dpi=dpi)
        axes = axes.ravel()
        for ax in axes[len(TARGETS):]:
            ax.axis("off")
        for index, target in enumerate(TARGETS):
            ax = axes[index]
            for label, frame in metrics.items():
                rows = val_rows(frame)
                rows = rows[rows["target"] == target].sort_values("epoch")
                if rows.empty:
                    continue
                x = rows["step"] if "step" in rows.columns else rows["epoch"]
                ax.plot(x, rows[key], marker="o", ms=3, lw=1.2, label=label)
            ax.set_title(f"{target} ({UNITS[target]})")
            ax.set_xlabel("global step")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
            if index == 0:
                ax.legend(fontsize=8)
        fig.suptitle(f"validation {ylabel}")
        fig.tight_layout()
        path = out_dir / filename
        fig.savefig(path)
        plt.close(fig)
        saved.append(path)

    if any(frame is not None for frame in losses.values()):
        fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), dpi=dpi)
        axes = axes.ravel()
        for index, term in enumerate(TERMS):
            ax = axes[index]
            for label, frame in losses.items():
                if frame is None:
                    continue
                rows = frame[(frame["split"] == "val") & (frame["term"] == term)].sort_values("epoch")
                if rows.empty:
                    continue
                x = rows["step"] if "step" in rows.columns else rows["epoch"]
                ax.plot(x, rows["value"], marker="o", ms=3, lw=1.2, label=label)
            ax.set_title(term)
            ax.set_xlabel("global step")
            ax.set_ylabel("weighted loss (z)")
            ax.grid(alpha=0.3)
            if index == 0:
                ax.legend(fontsize=8)
        fig.suptitle("validation loss terms")
        fig.tight_layout()
        path = out_dir / "val_loss_terms.png"
        fig.savefig(path)
        plt.close(fig)
        saved.append(path)

    if any(frame is not None for frame in steps.values()):
        fig, ax = plt.subplots(figsize=(10, 4.5), dpi=dpi)
        for label, frame in steps.items():
            if frame is None or "loss" not in frame.columns:
                continue
            rows = frame.sort_values("step")
            values = pd.to_numeric(rows["loss"], errors="coerce")
            ax.plot(rows["step"], values, lw=0.5, alpha=0.25)
            ax.plot(rows["step"], values.rolling(smooth, min_periods=1).mean(), lw=1.4, label=label)
        ax.set_xlabel("global step")
        ax.set_ylabel("train loss (z, weighted)")
        ax.set_title(f"per-step training loss (smooth={smooth})")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = out_dir / "train_loss.png"
        fig.savefig(path)
        plt.close(fig)
        saved.append(path)

        fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), dpi=dpi)
        axes = axes.ravel()
        for index, term in enumerate(TERMS):
            ax = axes[index]
            for label, frame in steps.items():
                if frame is None or term not in frame.columns:
                    continue
                rows = frame.sort_values("step")
                values = pd.to_numeric(rows[term], errors="coerce")
                ax.plot(rows["step"], values, lw=0.5, alpha=0.25)
                ax.plot(rows["step"], values.rolling(smooth, min_periods=1).mean(), lw=1.4, label=label)
            ax.set_title(term)
            ax.set_xlabel("global step")
            ax.set_ylabel("weighted loss (z)")
            ax.grid(alpha=0.3)
            if index == 0:
                ax.legend(fontsize=8)
        fig.suptitle(f"per-step loss terms (smooth={smooth})")
        fig.tight_layout()
        path = out_dir / "train_terms.png"
        fig.savefig(path)
        plt.close(fig)
        saved.append(path)

        fig, ax = plt.subplots(figsize=(10, 4.5), dpi=dpi)
        for label, frame in steps.items():
            if frame is None or "step_per_sec" not in frame.columns:
                continue
            rows = frame.sort_values("step")
            values = pd.to_numeric(rows["step_per_sec"], errors="coerce")
            ax.plot(rows["step"], values.rolling(smooth, min_periods=1).mean(), lw=1.4, label=label)
        ax.set_xlabel("global step")
        ax.set_ylabel("steps/s")
        ax.set_title(f"per-step speed (smooth={smooth})")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = out_dir / "train_speed.png"
        fig.savefig(path)
        plt.close(fig)
        saved.append(path)
    return saved


def plot_run(run_dir):
    run_dir = Path(run_dir)
    out_dir = Path(__file__).resolve().parent / "artifacts" / "pictures" / run_dir.name
    return plot_runs([(run_dir.name, run_dir)], out_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="run 曲线绘图（支持多 run 对比叠加）")
    parser.add_argument("--run", action="append", required=True,
                        help="label=run目录（可重复）；也可只给目录（label 取目录名）")
    parser.add_argument("--out", default=None, help="输出目录（默认 pictures/<第一个 label>）")
    parser.add_argument("--smooth", type=int, default=50, help="step 级曲线的滑动平均窗口（步数）")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runs = []
    for item in args.run:
        if "=" in item and not Path(item).exists():
            label, path = item.split("=", 1)
        else:
            label, path = Path(item).name, item
        runs.append((label, path))
    if args.out:
        out_dir = Path(args.out)
        if not out_dir.is_absolute():
            out_dir = Path(__file__).resolve().parents[1] / out_dir
    else:
        out_dir = Path(__file__).resolve().parent / "artifacts" / "pictures" / runs[0][0]
    saved = plot_runs(runs, out_dir, smooth=args.smooth)
    for path in saved:
        print(f"[plot] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
