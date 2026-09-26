r"""数据-模型契约检查：全量扫描图分片，验证与模型代码的接口假设。

用法（工作目录 E:\Material_MTL）:
    python Models\check_data.py
    python Models\check_data.py --only formation_energy_band_gap vacancy_screening
    python Models\check_data.py --limit-shards 2 --skip-prepare
    python Models\check_data.py --max-examples 10

检查项：
  分片配对/键一致；x/z/edge/线图张量的形状、dtype、索引范围、有限性；
  每样本标签与 index.csv 对齐；prepare_sample 全量试跑（与训练同路径）；
  global_feat.csv 覆盖与 NaN；子集缓存完整性；data/model 常量一致性。
换数据或重建图分片后应先跑本脚本，再开始训练。
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import file_digest, resolve_path
from data import EDGE_DIM, GLOBAL_FEATURE_COLUMNS, NODE_DIM, SOURCE_REGISTRY, Source, prepare_sample
from model import EDGE_DIM as MODEL_EDGE_DIM
from model import NODE_DIM as MODEL_NODE_DIM

SPLIT_CFG = {"val_frac": 0.0, "test_frac": 0.0, "seed": 0}
EXPECTED_KEYS = {"x", "z", "edge_index", "edge_attr", "line_edge_index", "line_edge_attr",
                 "labels", "vacancy", "global_feat", "name", "split_group", "n_atoms"}


def _is_nan(value) -> bool:
    if value is None:
        return True
    try:
        return bool(value != value)
    except Exception:
        return True


def _attr_value(data, column: str):
    value = getattr(data, column, None)
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        return float(value.reshape(-1)[0])
    return float(value)


def _same_value(expected, actual) -> bool:
    expected_nan = _is_nan(expected)
    if actual is None:
        return expected_nan
    actual_nan = not math.isfinite(actual)
    if expected_nan:
        return actual_nan
    return (not actual_nan) and abs(actual - float(expected)) < 1e-6


class DatasetCheck:
    def __init__(self, name: str, max_examples: int):
        self.name = name
        self.max_examples = max_examples
        self.hard = 0
        self.examples: list[str] = []
        self.total = 0
        self.unique = 0
        self.duplicates = 0
        self.prepared_ok = 0
        self.prepare_failed = 0
        self.label_mismatch = 0
        self.atoms_mismatch = 0
        self.empty_edges = 0
        self.empty_line = 0
        self.vacancy_missing = 0
        self.vacancy_samples = 0
        self.max_atoms = 0
        self.max_edges = 0
        self.max_lines = 0

    def fail(self, mid: str, message: str) -> None:
        self.hard += 1
        if len(self.examples) < self.max_examples:
            self.examples.append(f"{mid}: {message}")


def raw_checks(check: DatasetCheck, mid: str, data, line_obj, source: Source,
               label_maps: dict, atoms_map: dict) -> bool:
    state = {"ok": True}

    def fail(message: str) -> None:
        state["ok"] = False
        check.fail(mid, message)

    x = getattr(data, "x", None)
    if not torch.is_tensor(x) or x.dim() != 2 or x.size(1) != NODE_DIM:
        fail(f"x 形状异常 {None if x is None else tuple(x.shape)}")
        n_atoms = 0
    else:
        n_atoms = int(x.size(0))
        if x.dtype != torch.float32:
            fail(f"x dtype={x.dtype}")
        if n_atoms == 0:
            fail("N=0")
        elif not torch.isfinite(x).all():
            fail("x 含 NaN/Inf")

    z = getattr(data, "z", None)
    if not torch.is_tensor(z) or z.dim() != 1 or z.dtype != torch.int64:
        fail("z 缺失或形状/dtype 异常")
    elif n_atoms and z.size(0) != n_atoms:
        fail("z 长度 != N")
    elif z.numel() and (int(z.min()) < 1 or int(z.max()) > 118):
        fail(f"z 越界 [{int(z.min())},{int(z.max())}]")

    if n_atoms and int(getattr(data, "n_atoms", -1)) != n_atoms:
        fail(f"n_atoms={getattr(data, 'n_atoms', None)} != N={n_atoms}")

    edge_index = getattr(data, "edge_index", None)
    if (not torch.is_tensor(edge_index) or edge_index.dim() != 2
            or edge_index.size(0) != 2 or edge_index.dtype != torch.int64):
        fail("edge_index 形状/dtype 异常")
        e_count = 0
    else:
        e_count = int(edge_index.size(1))
        if e_count and n_atoms and (int(edge_index.min()) < 0 or int(edge_index.max()) >= n_atoms):
            fail("edge_index 越界")
        if int(getattr(data, "n_edges", -1)) != e_count:
            fail(f"n_edges={getattr(data, 'n_edges', None)} != E={e_count}")

    edge_attr = getattr(data, "edge_attr", None)
    if not torch.is_tensor(edge_attr) or edge_attr.dim() != 2 or edge_attr.size(1) != EDGE_DIM:
        fail("edge_attr 形状异常")
    else:
        if edge_attr.size(0) != e_count:
            fail("edge_attr 行数 != E")
        if edge_attr.dtype != torch.float32:
            fail(f"edge_attr dtype={edge_attr.dtype}")
        if edge_attr.numel() and not torch.isfinite(edge_attr).all():
            fail("edge_attr 含 NaN/Inf")
    if e_count == 0:
        check.empty_edges += 1
    check.max_atoms = max(check.max_atoms, n_atoms)
    check.max_edges = max(check.max_edges, e_count)

    line_index = line_obj.get("line_edge_index")
    if (not torch.is_tensor(line_index) or line_index.dim() != 2
            or line_index.size(0) != 2 or line_index.dtype != torch.int64):
        fail("line_edge_index 形状/dtype 异常")
        l_count = 0
    else:
        l_count = int(line_index.size(1))
        if l_count and e_count and (int(line_index.min()) < 0 or int(line_index.max()) >= e_count):
            fail("line_edge_index 越界")

    line_attr = line_obj.get("line_edge_attr")
    if not torch.is_tensor(line_attr) or line_attr.dim() != 2:
        fail("line_edge_attr 形状异常")
    else:
        if line_attr.size(0) != l_count:
            fail("line_edge_attr 行数 != L")
        if l_count and line_attr.size(1) != 1:
            fail(f"非空 line_edge_attr 宽度 {line_attr.size(1)} != 1")
        if line_attr.dtype != torch.float32:
            fail(f"line_edge_attr dtype={line_attr.dtype}")
        if line_attr.numel() and not torch.isfinite(line_attr).all():
            fail("line_edge_attr 含 NaN/Inf")
        if l_count == 0:
            check.empty_line += 1
        check.max_lines = max(check.max_lines, l_count)
    if int(line_obj.get("n_edges", -1)) != e_count:
        fail(f"线图 n_edges={line_obj.get('n_edges')} != E={e_count}")

    for slot, column in source.labels.items():
        mapping = label_maps.get(column)
        if mapping is None:
            continue
        if mid not in mapping:
            fail(f"mid 不在 index（列 {column}）")
            continue
        actual = _attr_value(data, column)
        if not _same_value(mapping[mid], actual):
            check.label_mismatch += 1
            fail(f"标签 {slot} 不一致 attr={actual} index={mapping[mid]}")

    if n_atoms and mid in atoms_map:
        index_atoms = atoms_map[mid]
        if not _is_nan(index_atoms) and int(index_atoms) != n_atoms:
            check.atoms_mismatch += 1
            fail(f"n_atoms 与 index 不一致 graph={n_atoms} index={int(index_atoms)}")

    if source.has_vacancy:
        vacancy = getattr(data, "vacancy", None)
        if vacancy is None:
            check.vacancy_missing += 1
        else:
            check.vacancy_samples += 1
            if (not torch.is_tensor(vacancy) or vacancy.dim() != 2 or vacancy.size(1) != 1
                    or vacancy.size(0) != n_atoms):
                fail(f"vacancy 形状异常 {None if vacancy is None else tuple(vacancy.shape)}")
    return state["ok"]


def prepared_checks(check: DatasetCheck, mid: str, sample) -> None:
    keys = set(sample.keys())
    if keys != EXPECTED_KEYS:
        check.fail(mid, f"拼批键集不一致 多={sorted(keys - EXPECTED_KEYS)} 少={sorted(EXPECTED_KEYS - keys)}")
    labels = getattr(sample, "labels", None)
    if not torch.is_tensor(labels) or tuple(labels.shape) != (1, 4):
        check.fail(mid, f"labels 形状 {None if labels is None else tuple(labels.shape)}")
    vacancy = getattr(sample, "vacancy", None)
    if not torch.is_tensor(vacancy) or vacancy.dim() != 2 or vacancy.size(1) != 1:
        check.fail(mid, "vacancy 形状异常")
    feature = getattr(sample, "global_feat", None)
    if not torch.is_tensor(feature) or tuple(feature.shape) != (1, len(GLOBAL_FEATURE_COLUMNS)):
        check.fail(mid, f"global_feat 形状 {None if feature is None else tuple(feature.shape)}")
    line_attr = getattr(sample, "line_edge_attr", None)
    if not torch.is_tensor(line_attr) or line_attr.dim() != 2 or line_attr.size(1) != 1:
        check.fail(mid, "prepare 后 line_edge_attr 宽度 != 1")


def check_dataset(name: str, args, global_stats) -> tuple[DatasetCheck, dict]:
    check = DatasetCheck(name, args.max_examples)
    source = Source(name, SPLIT_CFG, global_stats=global_stats, require_global=False)
    crystal_paths = sorted(source.dir.glob("crystal_graph_part*.pkl"))
    line_paths = sorted(source.dir.glob("line_graph_part*.pkl"))
    if len(crystal_paths) != len(line_paths):
        check.fail("-", f"分片数不一致 crystal={len(crystal_paths)} line={len(line_paths)}")

    table = source.table[~source.table.index.duplicated(keep="first")]
    label_maps: dict = {}
    for column in sorted(set(source.labels.values())):
        if column in table.columns:
            label_maps[column] = table[column].to_dict()
        else:
            check.fail("-", f"index.csv 缺列 {column}")
    atoms_map = table["n_atoms"].to_dict() if "n_atoms" in table.columns else {}

    limit = min(args.limit_shards or len(crystal_paths), len(crystal_paths), len(line_paths))
    seen: dict[str, int] = {}
    for index in range(limit):
        with open(crystal_paths[index], "rb") as fh:
            crystal = pickle.load(fh)
        with open(line_paths[index], "rb") as fh:
            line = pickle.load(fh)
        if set(crystal.keys()) != set(line.keys()):
            check.fail("-", f"第 {index} 片 crystal/line 键集不一致")
        for mid, data in crystal.items():
            check.total += 1
            if mid in seen:
                check.duplicates += 1
                if check.duplicates <= check.max_examples:
                    check.examples.append(f"[info] {mid}: 跨分片重复（分片 {seen[mid]} 与 {index}）")
                continue
            seen[mid] = index
            line_obj = line.get(mid)
            if line_obj is None:
                check.fail(mid, "线图缺样本")
                continue
            ok = raw_checks(check, mid, data, line_obj, source, label_maps, atoms_map)
            if ok and not args.skip_prepare:
                try:
                    sample = prepare_sample(data, line_obj, source, source.records.get(mid), mid)
                except Exception as exc:
                    check.prepare_failed += 1
                    check.fail(mid, f"prepare_sample: {type(exc).__name__}: {exc}")
                else:
                    check.prepared_ok += 1
                    prepared_checks(check, mid, sample)
    check.unique = len(seen)

    feature_path = source.dir / "global_feat.csv"
    missing_global = [mid for mid in seen if mid not in source.global_feat]
    nan_global = sum(1 for value in source.global_feat.values() if not np.isfinite(value).all())
    if not feature_path.exists():
        check.fail("-", "global_feat.csv 缺失")
    elif missing_global:
        check.fail("-", f"graph 样本缺全局特征 {len(missing_global)} 条（示例 {missing_global[:3]}）")
    if nan_global:
        check.fail("-", f"global_feat 含 NaN 行 {nan_global}")
    extra = {
        "rows": len(source.global_feat),
        "missing": len(missing_global),
        "nan": nan_global,
        "skipped_prepare": args.skip_prepare,
    }
    return check, extra


def print_summary(check: DatasetCheck, extra: dict) -> None:
    print(f"== {check.name} ==")
    print(f"  样本={check.total}（唯一 {check.unique}，跨分片重复 {check.duplicates}）")
    if extra["skipped_prepare"]:
        print("  prepare_sample: 已跳过（--skip-prepare）")
    else:
        print(f"  prepare_sample: 成功 {check.prepared_ok} / 失败 {check.prepare_failed}")
    print(f"  标签不一致={check.label_mismatch} / n_atoms 不一致={check.atoms_mismatch}")
    print(f"  空边图={check.empty_edges} / 空线图={check.empty_line}")
    print(f"  最大单图: N={check.max_atoms} E={check.max_edges} L={check.max_lines}")
    if check.vacancy_samples or check.vacancy_missing:
        print(f"  vacancy 标签: 有 {check.vacancy_samples} / 无 {check.vacancy_missing}")
    print(f"  global_feat: 表 {extra['rows']} 行 / 缺失 {extra['missing']} / NaN {extra['nan']}")
    print(f"  硬性违规 = {check.hard}")
    for example in check.examples:
        print(f"    {example}")


def check_subset_caches() -> list[tuple[str, str, int]]:
    results: list[tuple[str, str, int]] = []
    stats_path = resolve_path("Models/stats/label_stats.json")
    stats_hash = file_digest(stats_path) if stats_path.exists() else None
    for spec in SOURCE_REGISTRY.values():
        cache_dir = Path(spec["dir"]) / "subsets"
        manifest_path = cache_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for filename, meta in manifest.items():
            path = cache_dir / filename
            if not path.exists():
                results.append((filename, "文件不存在", 1))
                continue
            with open(path, "rb") as fh:
                payload = pickle.load(fh)
            samples = payload.get("samples", {})
            hard = 0
            notes: list[str] = []
            expected_key = filename[len("subset_"):-len(".pkl")]
            if payload.get("meta", {}).get("key") != expected_key:
                hard += 1
                notes.append("meta.key 与文件名不符")
            if len(samples) != int(meta.get("n", -1)):
                hard += 1
                notes.append(f"样本数 {len(samples)} != manifest {meta.get('n')}")
            if stats_hash and meta.get("stats_hash") != stats_hash:
                notes.append("[warn] stats md5 与当前 label_stats 不一致（训练会自动回退全量扫描）")
            bad_keys = 0
            bad_feature = 0
            for sample in samples.values():
                if set(sample.keys()) != EXPECTED_KEYS:
                    bad_keys += 1
                feature = getattr(sample, "global_feat", None)
                if not torch.is_tensor(feature) or tuple(feature.shape) != (1, len(GLOBAL_FEATURE_COLUMNS)):
                    bad_feature += 1
            if bad_keys:
                hard += 1
                notes.append(f"键集异常 {bad_keys}")
            if bad_feature:
                hard += 1
                notes.append(f"global_feat 形状异常 {bad_feature}")
            results.append((filename, "; ".join(notes) if notes else "OK", hard))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="数据-模型契约检查")
    parser.add_argument("--only", nargs="*", default=None, help="只检查这些数据源")
    parser.add_argument("--limit-shards", type=int, default=None, help="每个数据源只查前 N 个分片（冒烟）")
    parser.add_argument("--skip-prepare", action="store_true", help="不做 prepare_sample 全量试跑（更快）")
    parser.add_argument("--max-examples", type=int, default=8, help="每类问题最多打印几个例子")
    args = parser.parse_args()

    total_hard = 0
    if (NODE_DIM, EDGE_DIM) != (MODEL_NODE_DIM, MODEL_EDGE_DIM):
        print(f"[预检] FAIL: data 常量 ({NODE_DIM},{EDGE_DIM}) != model 常量 ({MODEL_NODE_DIM},{MODEL_EDGE_DIM})")
        total_hard += 1
    else:
        print(f"[预检] data/model 常量一致: NODE_DIM={NODE_DIM} EDGE_DIM={EDGE_DIM}")

    stats_path = resolve_path("Models/stats/label_stats.json")
    global_stats = None
    if stats_path.exists():
        with open(stats_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        global_stats = payload.get("global_feat")
        if global_stats and list(global_stats.get("columns", [])) != list(GLOBAL_FEATURE_COLUMNS):
            print("[预检] FAIL: label_stats.global_feat 列与 data.GLOBAL_FEATURE_COLUMNS 不一致")
            total_hard += 1
        else:
            print("[预检] label_stats.global_feat 列一致")
    else:
        print("[预检] 警告: 缺少 label_stats.json（按无全局特征统计检查）")

    names = args.only or ["formation_energy_band_gap", "vacancy_screening", "batio3", "batio3_doped"]
    for name in names:
        if name not in SOURCE_REGISTRY:
            print(f"未知数据源: {name}")
            return 2
        print(f"[检查] {name} ...", flush=True)
        check, extra = check_dataset(name, args, global_stats)
        print_summary(check, extra)
        total_hard += check.hard

    cache_results = check_subset_caches()
    if cache_results:
        print("== 子集缓存 ==")
        for filename, note, hard in cache_results:
            print(f"  {filename}: {note}")
            total_hard += hard

    print(f"\n[结论] 硬性违规总数 = {total_hard}")
    return 1 if total_hard else 0


if __name__ == "__main__":
    raise SystemExit(main())
