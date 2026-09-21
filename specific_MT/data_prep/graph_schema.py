"""图数据的 PyG Data 子类：修正键角线图索引在多图 batch 时的偏移量。

    line_edge_index 指向的是"原图边"的编号，拼接时应加 num_edges（不是 num_nodes）。
    PyG 默认把 *_index 类键的偏移量取为 num_nodes，直接用会指错边。

下游训练脚本加载 graphs.pkl 时需要能导入本模块（例如把 data_prep 目录加入 sys.path）。
"""
from __future__ import annotations

from typing import Any

from torch_geometric.data import Data


class CrystalData(Data):
    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key == "line_edge_index":
            return self.edge_index.size(1)
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key == "line_edge_index":
            return -1
        return super().__cat_dim__(key, value, *args, **kwargs)
