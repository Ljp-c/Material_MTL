# pkl 文件转图文件（build_graphs.py）

> 把 `dielectric_dataset.pkl` 里的晶体结构，转成图神经网络（GNN）能直接训练的图数据。
> 对应脚本：`E:\Material_MTL\模型具体训练\build_graphs.py`

---

## 1. 输入：pkl 里有什么

`dielectric_dataset.pkl` 是一个**字典**，只有两个键：

```
{
    "labels": <pandas 表格>,          # 7327 行 × 13 列，介电数值等
    "structures": {                   # 又一个字典：material_id -> 晶体结构
        "mp-14":   <Structure 对象>,   # Se
        "mp-5986": <Structure 对象>,   # BaTiO3（四方 P4mm）
        ... 共 7327 条
    }
}
```

- `labels` 的列：`material_id, formula, nsites, nelements, chemsys, elements, density, spacegroup_number, spacegroup_symbol, e_total, e_ionic, e_electronic, n_refractive`
- `structures` 的值是 pymatgen 的 `Structure` 对象，每个对象包含三部分：
  - **晶格**：3×3 矩阵（三条边的向量）
  - **原子表**：每个原子的元素 + 分数坐标
  - **pbc**：三个方向是否周期性（都是 True）

注意：结构在这个文件里**已经是展开好的完整晶胞**（如 BaTiO3 是 5 个原子），
不需要再用空间群去生成；空间群只是 `labels` 里的两个标签列。

---

## 2. 输出：什么是"图"

把晶体看成一张"社交网络"：

| 晶体里的东西 | 图里的东西 | 代码字段 |
|---|---|---|
| 原子 | 人（节点） | `x` |
| 相邻的两个原子 | 朋友关系（边） | `edge_index` |
| 两原子的距离 | 关系有多近 | `edge_attr` |
| 介电常数 | 这张图的"答案" | `y` |

**为什么用图？** 因为 GNN 能处理任意大小、任意形状的结构，
且"谁和谁相邻"这种局部信息对预测性质最重要。

---

## 3. 转换流程（脚本的 6 步）

### 第 1 步：读取 pkl（`build_graphs.py:114-117`）

```python
with open(args.data, "rb") as f:
    dataset = pickle.load(f)
labels = dataset["labels"]
structures = dataset["structures"]
```

### 第 2 步：可选过滤（`:119-127`）

- `--limit N`：只取前 N 条，用于冒烟测试；
- `--max-atoms 200`：跳过超大的晶胞（默认跳了 1 个）。

### 第 3 步：准备一把"距离尺子"（`:34-39`）

```python
self.centers = torch.linspace(0.0, self.cutoff, self.rbf_bins)  # 0~5 Å 撒 32 个刻度
self.width = self.cutoff / max(self.rbf_bins - 1, 1)            # 刻度间距
```

### 第 4 步：逐条把结构变成图（`:45-90` 核心）

对每个晶体做 5 件事：

1. **节点特征**（`:53`）：每个原子的原子序数（Ba=56, Ti=22, O=8）
2. **找邻居**（`:57`）：`structure.get_all_neighbors(5.0, include_index=True)`
   —— 自动考虑周期性，跨晶胞边界的原子也算
3. **筛边**（`:59-61`）：每个原子的邻居按距离排序，只留最近的 12 个；
   记录"邻居编号 → 中心原子编号"和距离
4. **边特征**（`:68-70`）：把 1 个距离数展开成 32 维的"距离指纹"（RBF）
5. **挂标签**（`:88-93`）：`e_total` 存成标准的 `y`，附带 `e_ionic`、`e_electronic`

### 第 5 步：保存（`build_graphs.py:144-160`）

- `graphs.pkl`：`{"graphs": {material_id: Data}, "meta": {...}}`
- `graph_index.csv`：每个图的编号、化学式、原子数、边数、e_total

### 第 6 步：自检（`:162-170`）

用 `DataLoader` 拼一个批次，打印各张量的形状，确认模型能直接吃。

---

## 4. 关键概念（大白话）

### cutoff（截断半径，默认 5.0 Å）

只把 5 Å 以内的原子算作邻居。太远的原子相互作用很弱，不连。
5 Å 能覆盖 Ba-O（约 2.83 Å）和 Ti-O（约 2.0 Å），更远的第二壳层就不连了。

### max_neighbors（每个原子最多连几个，默认 12）

防止个别原子邻居太多导致图过大。

### edge_index（谁连谁）

形状 `[2, 边数]`：
- 第 0 行 = 起点（邻居）
- 第 1 行 = 终点（中心原子）

方向是"**邻居 → 中心**"，意思是"中心原子从邻居那里收集信息"（CGCNN 的做法）。

### RBF 展开（把 1 个数变成 32 个数）

距离不能直接当输入，要展开成一个 32 维向量：

```python
def rbf_expand(distances, centers, width):
    d = distances.view(-1, 1)
    return torch.exp(-((d - centers.view(1, -1)) ** 2) / (2.0 * width ** 2))
```

大白话：拿距离去和 32 个刻度逐个比。"距离 ≈ 这个刻度"就输出接近 1，
"离这个刻度很远"就输出接近 0。于是距离 2.0 Å 会变成一个"2.0 附近亮起来"的 32 维向量。

### 周期性 vs 空间群（最容易搞混）

| | 是什么 | 这个脚本用了吗 |
|---|---|---|
| **周期性** | 晶胞向上下左右无限重复 | 用了：`:57` 找邻居时会算跨边界的邻居 |
| **空间群** | 哪些旋转/镜面/平移让结构看起来不变 | 没用：结构已经是完整的，不需要再生成 |

**结论**：`build_graphs.py` 用到的是"格子会重复"，不是"结构有对称性"。

---

## 5. 每个"图"里有什么

| 字段 | 含义 | 形状 |
|---|---|---|
| `x` | 每个原子的原子序数 | 原子数 |
| `edge_index` | 谁连谁 | 2 × 边数 |
| `edge_attr` | 每条边的距离 RBF 指纹 | 边数 × 32 |
| `y` | 要预测的目标（e_total） | 1 |
| `e_ionic` / `e_electronic` | 附带的其他介电数值 | 1 |
| `n_atoms` / `n_edges` | 这张图的原子数、边数 | 标量 |

`graphs.pkl` 的 `meta` 里记录了构造参数：`cutoff=5.0`、`max_neighbors=12`、
`rbf_bins=32`、`edge_dim=32`、`target=e_total`。

---

## 6. 实测产物

| 文件 | 大小 | 内容 |
|---|---|---|
| `data/dielectric/graphs.pkl` | 214.7 MB | 7326 个图 |
| `data/dielectric/graph_index.csv` | 小 | 每个图的索引 |

- 平均原子数 16.7，平均边数 200，边特征维度 32
- 只跳过了 1 个超 200 原子的超大晶胞，转换失败 0 个
- 自检输出：`batch.x (6,), batch.edge_index (2, 72), batch.edge_attr (72, 32), batch.y (2,)`

---

## 7. 怎么用（训练时）

```python
import pickle
from torch_geometric.loader import DataLoader

with open("data/dielectric/graphs.pkl", "rb") as f:
    blob = pickle.load(f)
graphs = list(blob["graphs"].values())      # 7326 个 Data

loader = DataLoader(graphs, batch_size=32, shuffle=True)
for batch in loader:
    batch = batch.to("cuda")
    pred = model(batch)
    loss = (pred - batch.y.reshape(-1)).pow(2).mean()
    loss.backward()
```

---

## 8. 可调参数

```powershell
python build_graphs.py --cutoff 6 --max-neighbors 12 --rbf-bins 64
```

| 参数 | 默认 | 影响 |
|---|---|---|
| `--cutoff` | 5.0 Å | 越大图越胖、越慢；越小可能丢相互作用 |
| `--max-neighbors` | 12 | 控制每个原子的连接数上限 |
| `--rbf-bins` | 32 | 决定边特征维度；**改了模型输入维度必须跟着改** |
| `--target` | e_total | 换成 e_ionic / e_electronic |

---

## 9. 常见坑

1. **标签必须设成 `y`**：如果只把目标存成列名（如 `e_total`），
   `DataLoader` 拼批后 `batch.y` 是 `None`，训练时取不到标签。
   脚本里同时写了两份（`data[name]` 和 `data.y`）。
2. **维度要匹配**：模型端输入的边特征维度必须等于 `rbf_bins`，
   RBF 中心数变了模型也要重建。
3. **pkl 是字典不是表格**：要先用 `d["labels"]`、`d["structures"]` 取出内容。
4. **结构已是完整晶胞**：不要试图用空间群再展开一次。
5. **图文件较大**：214.7 MB，加载时内存要留够（全量读进来约几百 MB）。
