r"""两阶段训练/评估入口（预训练与微调共用同一脚本）。

用法（工作目录 E:\Material_MTL；先跑 label_stats.py）:
    python Models\train.py --config Models\configs\pretrain.yaml
    python Models\train.py --config Models\configs\finetune.yaml
    python Models\train.py --config Models\configs\pretrain.yaml --limit-shards 1 --max-steps 50
    python Models\train.py --config Models\configs\finetune.yaml --seed 7 --tag seed7
    python Models\train.py --config Models\configs\finetune.yaml --dropout 0.0 --tag fit_test
    python Models\train.py --config Models\configs\finetune.yaml --eval-only --split test --ckpt Models\artifacts\finetune\best.pt
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import file_digest, load_config, resolve_path, seed_everything
from data import GRAPH_TARGETS, MultiSourceBatcher, SampleFilter, Source, collate, load_family_groups
from losses import MultiTaskLoss
from metrics import format_metric, regression_metrics
from model import MultiTaskModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BaTiO3 多任务 GNN（形成能/带隙/带边/空位形成能）")
    parser.add_argument("--config", required=True, help="YAML 配置（支持 base 继承）")
    parser.add_argument("--limit-shards", type=int, default=None, help="每个数据源只读前 N 个分片（冒烟）")
    parser.add_argument("--max-steps", type=int, default=None, help="训练步数上限（冒烟）")
    parser.add_argument("--device", default=None, help="覆盖配置中的 device")
    parser.add_argument("--out-dir", default=None, help="覆盖输出目录")
    parser.add_argument("--ckpt", default=None, help="eval-only 或微调初始化使用的权重")
    parser.add_argument("--eval-only", action="store_true", help="只评估，不训练")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--tag", default="", help="输出目录后缀（区分多次实验）")
    parser.add_argument("--seed", type=int, default=None, help="覆盖配置中的随机种子（多种子集成）")
    parser.add_argument("--dropout", type=float, default=None, help="覆盖模型 dropout（拟合测试 / 正则扫描）")
    return parser.parse_args()


def build_sources(cfg: dict, limit_shards: int | None, label_stats: dict | None = None) -> list[Source]:
    family = load_family_groups() if cfg.get("exclude_family") else set()
    if family:
        print(f"[family] 家族组成 {len(family)} 个（batio3 / batio3_doped / 含 Ba 的 vacancy 组成）")
    global_stats = (label_stats or {}).get("global_feat")
    require_global = int(((cfg.get("model") or {}).get("global_dim", 0)) or 0) > 0
    if require_global and not global_stats:
        raise SystemExit("配置 global_dim>0，但 label_stats 缺 global_feat 统计；请重新运行 label_stats.py")
    common_filters = cfg.get("filters") or {}
    sources = []
    for entry in cfg.get("sources") or []:
        name = entry["name"]
        filters_cfg = dict(common_filters)
        filters_cfg.update(entry.get("filters") or {})
        filters = SampleFilter(
            min_atoms=int(filters_cfg.get("min_atoms") or 0),
            max_atoms=filters_cfg.get("max_atoms"),
            contains_elements=tuple(filters_cfg.get("contains_elements") or ()),
            perovskite_batio3=bool(filters_cfg.get("perovskite_batio3", False)),
            exclude_formulas=tuple(filters_cfg.get("exclude_formulas") or ()),
        )
        source = Source(name, cfg["split"], filters=filters, exclude_groups=family, limit_shards=limit_shards,
                        global_stats=global_stats, require_global=require_global)
        sources.append(source)
        print(f"[source] {name:26s} train={source.counts['train']:7d} val={source.counts['val']:6d} "
              f"test={source.counts['test']:6d} excluded={source.counts['excluded']:6d} "
              f"(家族 {source.n_family} / 过滤 {source.n_filtered} / 全局缺失 {source.n_global_missing}) "
              f"shards={source.n_shards}")
    return sources


def load_label_stats(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"缺少标签统计 {path}；先运行 python Models\\label_stats.py --config <配置>")
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if "targets" not in raw:
        raise SystemExit(f"{path} 内容不完整（缺 targets）")
    return raw


def build_model(cfg: dict, stats: dict, checkpoint: dict | None):
    model_cfg = dict((checkpoint or {}).get("model_config") or {})
    model_cfg.update(cfg.get("model") or {})
    model = MultiTaskModel(**model_cfg)
    if checkpoint is not None:
        missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
        if missing or unexpected:
            print(f"[load] missing={list(missing)} unexpected={list(unexpected)}")
    else:
        model.init_head_biases(stats["targets"])
    return model, model_cfg


def set_backbone_trainable(model: MultiTaskModel, unfrozen_blocks) -> None:
    for parameter in model.backbone_parameters():
        parameter.requires_grad = unfrozen_blocks is None
    if unfrozen_blocks is None:
        return
    blocks = list(model.backbone.blocks)
    for index, block in enumerate(blocks):
        if index >= len(blocks) - int(unfrozen_blocks):
            for parameter in block.parameters():
                parameter.requires_grad = True


def make_optimizer(model: MultiTaskModel, train_cfg: dict):
    head_lr = float(train_cfg.get("head_lr", train_cfg.get("lr", 3e-4)))
    backbone_lr = float(train_cfg.get("backbone_lr", train_cfg.get("lr", 3e-4)))
    weight_decay = float(train_cfg.get("weight_decay", 1e-5))
    groups = []
    head_params = [p for p in model.head_parameters() if p.requires_grad]
    backbone_params = [p for p in model.backbone_parameters() if p.requires_grad]
    if head_params:
        groups.append({"params": head_params, "lr": head_lr, "name": "head"})
    if backbone_params:
        groups.append({"params": backbone_params, "lr": backbone_lr, "name": "backbone"})
    if not groups:
        raise SystemExit("没有可训练参数（检查 freeze 配置）")
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def make_scheduler(optimizer, total_steps: int, warmup_frac: float, min_lr_ratio: float):
    warmup = max(1, int(total_steps * warmup_frac))

    def schedule(step: int) -> float:
        if step < warmup:
            return float(step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        progress = min(1.0, max(0.0, progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def save_checkpoint(path: Path, model, model_cfg, stage, epoch, global_step, stats_path, stats_hash, val) -> None:
    payload = {
        "model_state": model.state_dict(),
        "model_config": model_cfg,
        "stage": stage,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "stats_path": str(stats_path),
        "stats_hash": stats_hash,
        "val": val,
    }
    torch.save(payload, path)


def print_eval(header: str, result: dict) -> None:
    print(f"--- {header} ---")
    loss = result["loss"]
    if loss:
        total = sum(loss.values())
        shares = " ".join(f"{name}={value / total * 100:.0f}%" for name, value in sorted(loss.items()))
        print(f"  loss total={total:.4f} | {shares}")
    for target, metric in sorted(result["overall"].items()):
        print(f"  {target:10s} {format_metric(metric)}")
    for key, metric in sorted(result["pairs"].items()):
        print(f"    {key:42s} {format_metric(metric)}")


def _metric_cell(value):
    return "" if value is None else f"{value:.6g}"


def write_eval_rows(metric_writer, loss_writer, epoch, step, split, result) -> None:
    for key, metric in sorted(result["pairs"].items()):
        source_name, target = key.split("/", 1)
        metric_writer.writerow([epoch, step, split, source_name, target, metric.get("n", 0),
                                _metric_cell(metric.get("mae")), _metric_cell(metric.get("rmse")),
                                _metric_cell(metric.get("r2"))])
    for target, metric in sorted(result["overall"].items()):
        metric_writer.writerow([epoch, step, split, "__overall__", target, metric.get("n", 0),
                                _metric_cell(metric.get("mae")), _metric_cell(metric.get("rmse")),
                                _metric_cell(metric.get("r2"))])
    loss = result["loss"]
    total = sum(loss.values()) if loss else 0.0
    for name, value in sorted(loss.items()):
        share = value / total if total else 0.0
        loss_writer.writerow([epoch, step, split, name, f"{value:.6g}", f"{share:.4f}"])


@torch.no_grad()
def evaluate(model, criterion, batcher, device, split: str, max_batches=None) -> dict:
    model.eval()
    store = defaultdict(lambda: {"pred": [], "true": []})
    loss_sums = defaultdict(float)
    loss_counts = defaultdict(int)
    for source_name, items in batcher.eval_batches(split, max_batches=max_batches):
        batch = collate(items).to(device)
        preds = model(batch)
        _, info = criterion(preds, batch.labels, batch.vacancy, 0, 0, force_ramp=1.0)
        for name, value in info["weighted"].items():
            loss_sums[name] += float(value)
            loss_counts[name] += 1
        for slot, target in enumerate(GRAPH_TARGETS):
            raw = batch.labels[:, slot]
            mask = torch.isfinite(raw)
            if target == "gap":
                mask = mask & (raw > 0)
            if not bool(mask.any()):
                continue
            pred_phys = preds[target] * criterion.std[slot] + criterion.mean[slot]
            store[(source_name, target)]["pred"].append(pred_phys[mask].cpu())
            store[(source_name, target)]["true"].append(raw[mask].cpu())
        vacancy = batch.vacancy.reshape(-1)
        mask = torch.isfinite(vacancy)
        if bool(mask.any()):
            pred_phys = preds["vacancy"] * criterion.vac_std + criterion.vac_mean
            store[(source_name, "vacancy")]["pred"].append(pred_phys[mask].cpu())
            store[(source_name, "vacancy")]["true"].append(vacancy[mask].cpu())
    model.train()
    pairs = {}
    overall = defaultdict(lambda: {"pred": [], "true": []})
    for (source_name, target), values in store.items():
        pred = torch.cat(values["pred"])
        true = torch.cat(values["true"])
        pairs[f"{source_name}/{target}"] = regression_metrics(pred, true)
        overall[target]["pred"].extend(values["pred"])
        overall[target]["true"].extend(values["true"])
    overall_metrics = {
        target: regression_metrics(torch.cat(values["pred"]), torch.cat(values["true"]))
        for target, values in overall.items()
    }
    averaged_loss = {name: loss_sums[name] / loss_counts[name] for name in loss_sums}
    total_loss = sum(averaged_loss.values()) if averaged_loss else None
    return {"pairs": pairs, "overall": overall_metrics, "loss": averaged_loss, "loss_total": total_loss}


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.dropout is not None:
        cfg.setdefault("model", {})["dropout"] = float(args.dropout)
    stage = str(cfg.get("stage", "pretrain"))
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)

    device_name = str(args.device or cfg.get("device", "cuda"))
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("[device] CUDA 不可用，回退 CPU")
        device_name = "cpu"
    device = torch.device(device_name)

    paths = cfg.get("paths") or {}
    out_dir = resolve_path(args.out_dir or paths.get("out_dir") or "Models/artifacts/run")
    if args.tag:
        out_dir = out_dir.parent / f"{out_dir.name}_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    stats_path = resolve_path(paths.get("stats") or "Models/stats/label_stats.json")
    stats = load_label_stats(stats_path)
    stats_hash = file_digest(stats_path)
    print(f"[stats] {stats_path} (md5 {stats_hash[:8]})")

    print(f"[config] {args.config} stage={stage} device={device} out={out_dir}")
    sources = build_sources(cfg, args.limit_shards, stats)
    train_cfg = dict(cfg.get("train") or {})
    batcher = MultiSourceBatcher(
        sources,
        batch_size=int(train_cfg.get("batch_size", 64)),
        weights=(cfg.get("sampling_weights") or {}),
        sampling=str(cfg.get("sampling", "sqrt_inv")),
        seed=seed,
    )

    init_value = args.ckpt or cfg.get("init_from")
    init_path = resolve_path(init_value)
    checkpoint = None
    if init_path is not None:
        if not init_path.exists():
            raise SystemExit(f"找不到权重文件 {init_path}")
        checkpoint = torch.load(init_path, map_location="cpu", weights_only=True)
        if stage == "finetune" and checkpoint.get("stats_hash") and checkpoint["stats_hash"] != stats_hash \
                and not cfg.get("allow_stats_change"):
            raise SystemExit("微调必须与预训练共用同一份 label_stats（当前统计与 checkpoint 记录不一致）")
        print(f"[load] {init_path}")

    model, model_cfg = build_model(cfg, stats, checkpoint)
    model = model.to(device)
    counts = model.parameter_counts()
    print(f"[model] backbone={counts['backbone']} heads={counts['heads']} total={counts['total']}")

    loss_cfg = cfg.get("loss") or {}
    criterion = MultiTaskLoss(
        weights=loss_cfg.get("weights"),
        betas=loss_cfg.get("betas"),
        consistency_weight=float(loss_cfg.get("consistency_weight", 0.1)),
        consistency_ramp_frac=float(loss_cfg.get("consistency_ramp_frac", 0.05)),
        use_metal=bool(model_cfg.get("use_metal", False)),
        label_stats=stats,
    ).to(device)

    if args.eval_only:
        if checkpoint is None:
            raise SystemExit("--eval-only 需要 --ckpt 或配置中的 init_from 指向已训练权重")
        result = evaluate(model, criterion, batcher, device, args.split, max_batches=None)
        print_eval(f"{args.split}（eval-only）", result)
        dump = {"split": args.split, "pairs": result["pairs"], "overall": result["overall"], "loss": result["loss"]}
        with open(out_dir / f"{args.split}_metrics.json", "w", encoding="utf-8") as fh:
            json.dump(dump, fh, indent=2, ensure_ascii=False)
        print(f"[out] {out_dir / (args.split + '_metrics.json')}")
        return 0

    steps_per_epoch = batcher.steps_per_epoch
    if steps_per_epoch <= 0:
        raise SystemExit("训练集为空：检查 split / filters / --limit-shards")
    max_epochs = int(train_cfg.get("max_epochs", 20))
    total_steps = max(1, steps_per_epoch * max_epochs)
    warmup_frac = float(train_cfg.get("warmup_frac", 0.05))
    min_lr_ratio = float(train_cfg.get("min_lr_ratio", 0.05))
    grad_clip = float(train_cfg.get("grad_clip", 5.0) or 0.0)
    log_every = int(train_cfg.get("log_every_steps", 50))
    patience = int(train_cfg.get("early_stop_patience", 50))
    val_max_batches = train_cfg.get("val_max_batches")
    print(f"[plan] steps/epoch={steps_per_epoch} max_epochs={max_epochs} total_steps={total_steps}")

    freeze_cfg = dict(train_cfg.get("freeze") or {})
    freeze_epochs = int(freeze_cfg.get("freeze_epochs", 0)) if stage == "finetune" else 0
    unfreeze_blocks = int(freeze_cfg.get("unfreeze_last_blocks", 0)) if stage == "finetune" else 0

    def phase_of(epoch: int):
        if stage == "finetune" and freeze_epochs > 0:
            return 0 if epoch < freeze_epochs else unfreeze_blocks
        return None

    def phase_steps(epoch: int) -> int:
        if stage == "finetune" and freeze_epochs > 0 and epoch < freeze_epochs:
            remaining = min(freeze_epochs, max_epochs) - epoch
        else:
            remaining = max_epochs - epoch
        return max(1, remaining * steps_per_epoch)

    phase = phase_of(0)
    set_backbone_trainable(model, phase)
    optimizer = make_optimizer(model, train_cfg)
    scheduler = make_scheduler(optimizer, phase_steps(0), warmup_frac, min_lr_ratio)
    lr_text = " | ".join(f"{group['name']} lr={group['lr']}" for group in optimizer.param_groups)
    print(f"[train] backbone 可训练块={phase if phase is not None else 'all'} | {lr_text}")

    metric_file = open(out_dir / "metrics.csv", "w", newline="", encoding="utf-8")
    metric_writer = csv.writer(metric_file)
    metric_writer.writerow(["epoch", "step", "split", "source", "target", "n", "mae", "rmse", "r2"])
    loss_file = open(out_dir / "losses.csv", "w", newline="", encoding="utf-8")
    loss_writer = csv.writer(loss_file)
    loss_writer.writerow(["epoch", "step", "split", "term", "value", "share"])

    best_value = float("inf")
    bad_evals = 0
    global_step = 0

    for epoch in range(max_epochs):
        new_phase = phase_of(epoch)
        if new_phase != phase:
            phase = new_phase
            set_backbone_trainable(model, phase)
            optimizer = make_optimizer(model, train_cfg)
            scheduler = make_scheduler(optimizer, phase_steps(epoch), warmup_frac, min_lr_ratio)
            lr_text = " | ".join(f"{group['name']} lr={group['lr']}" for group in optimizer.param_groups)
            print(f"[phase] epoch {epoch}: backbone 可训练块={phase} | {lr_text}")

        model.train()
        run_loss = 0.0
        run_terms = defaultdict(float)
        run_term_counts = defaultdict(int)
        steps_done = 0
        reached_limit = False
        last_log_time = time.time()
        for source_name, items in batcher.epoch_batches(epoch):
            batch = collate(items).to(device)
            total, info = criterion(model(batch), batch.labels, batch.vacancy, global_step, total_steps)
            if total is None:
                continue
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], grad_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1
            steps_done += 1
            run_loss += float(total.detach())
            for name, value in info["weighted"].items():
                run_terms[name] += float(value.detach())
                run_term_counts[name] += 1
            if log_every and global_step % log_every == 0:
                now = time.time()
                speed = log_every / max(now - last_log_time, 1e-6)
                last_log_time = now
                mean_terms = " ".join(
                    f"{name}={run_terms[name] / run_term_counts[name]:.4f}" for name in sorted(run_terms))
                print(f"[{stage}] epoch {epoch:3d} step {global_step:7d} "
                      f"loss {run_loss / max(1, steps_done):.4f} | {speed:.1f} step/s | {mean_terms}")
            if args.max_steps and global_step >= args.max_steps:
                reached_limit = True
                break
        if steps_done == 0:
            print("[warn] 本 epoch 没有可用批次")

        result = evaluate(model, criterion, batcher, device, "val", max_batches=val_max_batches)
        print_eval(f"val epoch {epoch}（step {global_step}）", result)
        write_eval_rows(metric_writer, loss_writer, epoch, global_step, "val", result)
        metric_file.flush()
        loss_file.flush()

        val_total = result["loss_total"]
        if val_total is not None and val_total < best_value - 1e-4:
            best_value = val_total
            bad_evals = 0
            save_checkpoint(out_dir / "best.pt", model, model_cfg, stage, epoch, global_step,
                            stats_path, stats_hash, result["overall"])
        else:
            bad_evals += 1
        save_checkpoint(out_dir / "last.pt", model, model_cfg, stage, epoch, global_step,
                        stats_path, stats_hash, result["overall"])

        if reached_limit:
            print(f"[stop] 达到 --max-steps ({global_step})")
            break
        if bad_evals >= patience:
            print(f"[stop] 早停：连续 {bad_evals} 次验证未提升（best={best_value:.4f}）")
            break

    metric_file.close()
    loss_file.close()

    if any(source.count("test") > 0 for source in sources):
        best_path = out_dir / "best.pt"
        if best_path.exists():
            payload = torch.load(best_path, map_location="cpu", weights_only=True)
            model.load_state_dict(payload["model_state"], strict=False)
            model = model.to(device)
        result = evaluate(model, criterion, batcher, device, "test", max_batches=None)
        print_eval("test（训练结束）", result)
        with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as fh:
            json.dump({"pairs": result["pairs"], "overall": result["overall"], "loss": result["loss"]},
                      fh, indent=2, ensure_ascii=False)
    print(f"[out] {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
