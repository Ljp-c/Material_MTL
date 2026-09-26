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

from common import file_digest, load_config, load_torch, resolve_path, seed_everything
from data import (GRAPH_TARGETS, MultiSourceBatcher, SampleFilter, Source, collate,
                  load_pretrain_exclude_groups)
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
    parser.add_argument("--resume", action="store_true",
                        help="从 out_dir/last.pt（或 --ckpt）续训：恢复权重/优化器/调度器/epoch")
    parser.add_argument("--prefetch-depth", type=int, default=None,
                        help="分片预读深度（0=关闭；默认取配置 train.prefetch_depth）")
    parser.add_argument("--val-every-steps", type=int, default=None,
                        help="每 N 步做一次验证并写入 metrics.csv（0=关闭；默认取配置 train.val_every_steps）")
    return parser.parse_args()


def build_sources(cfg: dict, limit_shards: int | None, label_stats: dict | None = None,
                  stats_hash: str | None = None) -> list[Source]:
    exclude_groups = load_pretrain_exclude_groups() if cfg.get("exclude_family") else set()
    if exclude_groups:
        print(f"[pretrain 排除] 微调留出组成 {len(exclude_groups)} 个"
              "（仅微调 val/test；其余家族数据回流预训练）")
    global_stats = (label_stats or {}).get("global_feat")
    require_global = int(((cfg.get("model") or {}).get("global_dim", 0)) or 0) > 0
    if require_global and not global_stats:
        raise SystemExit("配置 global_dim>0，但 label_stats 缺 global_feat 统计；请重新运行 label_stats.py")
    common_filters = cfg.get("filters") or {}
    prefetch_depth = int((cfg.get("train") or {}).get("prefetch_depth", 0) or 0)
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
        subset_cache = bool(entry.get("subset_cache", False))
        source = Source(name, cfg["split"], filters=filters, exclude_groups=exclude_groups,
                        limit_shards=limit_shards, global_stats=global_stats, require_global=require_global,
                        subset_cache=subset_cache, stats_hash=stats_hash, prefetch_depth=prefetch_depth)
        sources.append(source)
        print(f"[source] {name:26s} train={source.counts['train']:7d} val={source.counts['val']:6d} "
              f"test={source.counts['test']:6d} excluded={source.counts['excluded']:6d} "
              f"(预训练排除 {source.n_family} / 过滤 {source.n_filtered} / 全局缺失 {source.n_global_missing} / "
              f"子集缓存 {'开' if source.subset_cache else '关'}) shards={source.n_shards}")
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


def save_checkpoint(path: Path, model, model_cfg, stage, epoch, global_step, stats_path, stats_hash, val,
                    optimizer=None, scheduler=None, run_state=None) -> None:
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
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    if run_state is not None:
        payload["run_state"] = run_state
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
    if args.prefetch_depth is not None:
        cfg.setdefault("train", {})["prefetch_depth"] = int(args.prefetch_depth)
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
    sources = build_sources(cfg, args.limit_shards, stats, stats_hash=stats_hash)
    train_cfg = dict(cfg.get("train") or {})
    batcher = MultiSourceBatcher(
        sources,
        batch_size=int(train_cfg.get("batch_size", 64)),
        weights=(cfg.get("sampling_weights") or {}),
        sampling=str(cfg.get("sampling", "sqrt_inv")),
        seed=seed,
        max_atoms_per_batch=train_cfg.get("max_atoms_per_batch"),
    )

    resume_state = None
    start_epoch = 0
    checkpoint = None
    if args.resume:
        resume_path = resolve_path(args.ckpt) if args.ckpt else out_dir / "last.pt"
        if resume_path is None or not resume_path.exists():
            raise SystemExit(f"--resume 找不到 checkpoint：{resume_path}")
        checkpoint = load_torch(resume_path)
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        resume_state = checkpoint.get("run_state") or {}
        print(f"[resume] {resume_path} | 已完成 epoch {checkpoint.get('epoch')} / "
              f"global_step {checkpoint.get('global_step')} | 从 epoch {start_epoch} 继续")
        if args.limit_shards:
            print("[resume] 警告: --limit-shards 会改变数据流，续训与原始 run 不完全可比")
    else:
        init_value = args.ckpt or cfg.get("init_from")
        init_path = resolve_path(init_value)
        if init_path is not None:
            if not init_path.exists():
                raise SystemExit(f"找不到权重文件 {init_path}")
            checkpoint = load_torch(init_path)
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
    if args.resume and start_epoch >= max_epochs:
        print(f"[resume] 已完成 {start_epoch}/{max_epochs} 个 epoch，无需续训")
        return 0
    total_steps = max(1, steps_per_epoch * max_epochs)
    warmup_frac = float(train_cfg.get("warmup_frac", 0.05))
    min_lr_ratio = float(train_cfg.get("min_lr_ratio", 0.05))
    grad_clip = float(train_cfg.get("grad_clip", 5.0) or 0.0)
    log_every = int(train_cfg.get("log_every_steps", 50))
    patience = int(train_cfg.get("early_stop_patience", 50))
    val_max_batches = train_cfg.get("val_max_batches")
    atoms_cap = train_cfg.get("max_atoms_per_batch")
    prefetch_depth = int(train_cfg.get("prefetch_depth", 0) or 0)
    val_every_steps = int(args.val_every_steps) if args.val_every_steps is not None \
        else int(train_cfg.get("val_every_steps", 0) or 0)
    print(f"[plan] start_epoch={start_epoch} steps/epoch={steps_per_epoch} max_epochs={max_epochs} "
          f"total_steps={total_steps} batch≤{int(train_cfg.get('batch_size', 64))} 样本 / "
          f"≤{int(atoms_cap) if atoms_cap else '∞'} 原子 prefetch={prefetch_depth} val_every={val_every_steps}")

    freeze_cfg = dict(train_cfg.get("freeze") or {})
    freeze_epochs = int(freeze_cfg.get("freeze_epochs", 0)) if stage == "finetune" else 0
    unfreeze_blocks = int(freeze_cfg.get("unfreeze_last_blocks", 0)) if stage == "finetune" else 0

    def phase_of(epoch: int):
        if stage == "finetune" and freeze_epochs > 0:
            return 0 if epoch < freeze_epochs else unfreeze_blocks
        return None

    def phase_total_steps(epoch: int) -> int:
        if stage == "finetune" and freeze_epochs > 0:
            if epoch < min(freeze_epochs, max_epochs):
                span = min(freeze_epochs, max_epochs)
            else:
                span = max_epochs - min(freeze_epochs, max_epochs)
        else:
            span = max_epochs
        return max(1, span * steps_per_epoch)

    phase = phase_of(start_epoch)
    set_backbone_trainable(model, phase)
    optimizer = make_optimizer(model, train_cfg)
    scheduler = make_scheduler(optimizer, phase_total_steps(start_epoch), warmup_frac, min_lr_ratio)
    if resume_state and checkpoint.get("optimizer_state") is not None:
        saved_groups = len(checkpoint["optimizer_state"].get("param_groups", []))
        if saved_groups == len(optimizer.param_groups):
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            if checkpoint.get("scheduler_state") is not None:
                scheduler.load_state_dict(checkpoint["scheduler_state"])
            print("[resume] 优化器/调度器状态已恢复")
        else:
            print(f"[resume] 优化器分组数变化（保存 {saved_groups} vs 当前 {len(optimizer.param_groups)}），"
                  "本阶段优化器从头开始（与正常阶段切换一致）")
    elif resume_state:
        print("[resume] checkpoint 不含优化器状态（旧格式），已恢复权重与轮数")
    lr_text = " | ".join(f"{group['name']} lr={group['lr']}" for group in optimizer.param_groups)
    print(f"[train] backbone 可训练块={phase if phase is not None else 'all'} | {lr_text}")

    metric_file = open(out_dir / "metrics.csv", "w", newline="", encoding="utf-8")
    metric_writer = csv.writer(metric_file)
    metric_writer.writerow(["epoch", "step", "split", "source", "target", "n", "mae", "rmse", "r2"])
    loss_file = open(out_dir / "losses.csv", "w", newline="", encoding="utf-8")
    loss_writer = csv.writer(loss_file)
    loss_writer.writerow(["epoch", "step", "split", "term", "value", "share"])
    step_file = open(out_dir / "step_log.csv", "w", newline="", encoding="utf-8")
    step_writer = csv.writer(step_file)
    step_writer.writerow(["timestamp", "stage", "epoch", "step", "step_per_sec", "loss",
                          "formation", "gap", "cbm", "vbm", "vacancy", "consistency"])
    epoch_file = open(out_dir / "epoch_log.csv", "w", newline="", encoding="utf-8")
    epoch_writer = csv.writer(epoch_file)
    epoch_writer.writerow(["timestamp", "stage", "epoch", "step", "epoch_sec", "step_per_sec", "loss",
                           "formation", "gap", "cbm", "vbm", "vacancy", "consistency"])

    best_value = float("inf")
    bad_evals = 0
    global_step = 0
    if args.resume:
        if resume_state:
            best_value = float(resume_state.get("best_value", float("inf")))
            bad_evals = int(resume_state.get("bad_evals", 0))
        global_step = int(checkpoint.get("global_step", 0))

    for epoch in range(start_epoch, max_epochs):
        new_phase = phase_of(epoch)
        if new_phase != phase:
            phase = new_phase
            set_backbone_trainable(model, phase)
            optimizer = make_optimizer(model, train_cfg)
            scheduler = make_scheduler(optimizer, phase_total_steps(epoch), warmup_frac, min_lr_ratio)
            lr_text = " | ".join(f"{group['name']} lr={group['lr']}" for group in optimizer.param_groups)
            print(f"[phase] epoch {epoch}: backbone 可训练块={phase} | {lr_text}")

        model.train()
        run_loss = 0.0
        run_terms = defaultdict(float)
        run_term_counts = defaultdict(int)
        steps_done = 0
        reached_limit = False
        last_log_time = time.time()
        epoch_start = time.time()
        last_step_time = time.time()
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
            now = time.time()
            step_speed = 1.0 / max(now - last_step_time, 1e-6)
            last_step_time = now
            step_loss = float(total.detach())
            step_terms = {name: float(value.detach()) for name, value in info["weighted"].items()}
            run_loss += step_loss
            for name, value in step_terms.items():
                run_terms[name] += value
                run_term_counts[name] += 1
            step_writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), stage, epoch, global_step,
                                  f"{step_speed:.4f}", f"{step_loss:.6f}",
                                  *[f"{step_terms[name]:.6f}" if name in step_terms else ""
                                    for name in ("formation", "gap", "cbm", "vbm", "vacancy", "consistency")]])
            step_file.flush()
            if log_every and global_step % log_every == 0:
                speed = log_every / max(now - last_log_time, 1e-6)
                last_log_time = now
                mean_values = {name: run_terms[name] / run_term_counts[name] for name in run_terms}
                mean_loss = run_loss / max(1, steps_done)
                mean_terms = " ".join(f"{name}={mean_values[name]:.4f}" for name in sorted(mean_values))
                print(f"[{stage}] epoch {epoch:3d} step {global_step:7d} "
                      f"loss {mean_loss:.4f} | {speed:.1f} step/s | {mean_terms}")
            if args.max_steps and global_step >= args.max_steps:
                reached_limit = True
                break
            if val_every_steps > 0 and global_step % val_every_steps == 0:
                mid = evaluate(model, criterion, batcher, device, "val", max_batches=val_max_batches)
                print_eval(f"val step {global_step}", mid)
                write_eval_rows(metric_writer, loss_writer, epoch, global_step, "val", mid)
                metric_file.flush()
                loss_file.flush()
        if steps_done == 0:
            print("[warn] 本 epoch 没有可用批次")

        epoch_seconds = time.time() - epoch_start
        epoch_speed = steps_done / max(epoch_seconds, 1e-6)
        epoch_loss = run_loss / max(1, steps_done)
        epoch_values = {name: run_terms[name] / run_term_counts[name] for name in run_terms}
        epoch_writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), stage, epoch, global_step,
                               f"{epoch_seconds:.1f}", f"{epoch_speed:.4f}", f"{epoch_loss:.6f}",
                               *[f"{epoch_values[name]:.6f}" if name in epoch_values else ""
                                 for name in ("formation", "gap", "cbm", "vbm", "vacancy", "consistency")]])
        epoch_file.flush()
        print(f"[{stage}] epoch {epoch:3d} 汇总: loss {epoch_loss:.4f} | {epoch_speed:.1f} step/s "
              f"| {epoch_seconds / 60:.1f} min")

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
                        stats_path, stats_hash, result["overall"],
                        optimizer=optimizer, scheduler=scheduler,
                        run_state={"best_value": best_value, "bad_evals": bad_evals})

        if reached_limit:
            print(f"[stop] 达到 --max-steps ({global_step})")
            break
        if bad_evals >= patience:
            print(f"[stop] 早停：连续 {bad_evals} 次验证未提升（best={best_value:.4f}）")
            break

    metric_file.close()
    loss_file.close()
    step_file.close()
    epoch_file.close()

    if any(source.count("test") > 0 for source in sources):
        best_path = out_dir / "best.pt"
        if best_path.exists():
            payload = load_torch(best_path)
            model.load_state_dict(payload["model_state"], strict=False)
            model = model.to(device)
        result = evaluate(model, criterion, batcher, device, "test", max_batches=None)
        print_eval("test（训练结束）", result)
        with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as fh:
            json.dump({"pairs": result["pairs"], "overall": result["overall"], "loss": result["loss"]},
                      fh, indent=2, ensure_ascii=False)
    try:
        from plot_run import plot_run
        for path in plot_run(out_dir):
            print(f"[plot] {path}")
    except Exception as exc:
        print(f"[plot] 跳过绘图: {type(exc).__name__}: {exc}")
    try:
        import plot_run
        picture_dir = Path(__file__).resolve().parent / "artifacts" / "pictures" / out_dir.name
        saved = plot_run.plot_runs([(out_dir.name, out_dir)], picture_dir)
        if saved:
            print(f"[plots] {picture_dir}")
    except Exception as exc:
        print(f"[plots] 跳过: {type(exc).__name__}: {exc}")
    print(f"[out] {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
