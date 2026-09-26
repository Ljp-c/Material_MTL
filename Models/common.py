"""公共工具：配置加载（支持 base 继承）、路径解析、随机种子、文件摘要。"""
from __future__ import annotations

import hashlib
import random
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path) -> dict:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    base = cfg.pop("base", None)
    if base:
        base_path = Path(base)
        if not base_path.is_absolute():
            base_path = REPO_ROOT / base_path
        cfg = deep_merge(load_config(base_path), cfg)
    return cfg


def resolve_path(value) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def file_digest(path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_torch(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)
