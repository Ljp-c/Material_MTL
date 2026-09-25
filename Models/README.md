# Models —— BaTiO3 类掺杂晶体多任务 GNN（带隙 / 带边 / 缺陷形成能 / 形成能）

对齐 `工作思路.md` §4.1.1（数据 Schema）与 §4.2（图模型构建思路）。两阶段迁移：全库预训练 → BaTiO3 族微调。当前训练目标锁定四个：**带隙、带边位置（CBM/VBM）、缺陷（空位）形成能、形成能**。

## 1. 目录结构

```
Models/
  README.md              本文件
  common.py              配置加载（base 继承）、路径解析、随机种子、文件 md5
  data.py                分片流式读取、统一标签层、8 维全局特征、按组成分组划分、多源批采样
  model.py               双图骨干（晶体图 + 键角线图）+ 多任务头（池化 136）
  losses.py              掩码多任务损失（z 空间 SmoothL1 + 一致性项 + 可选 metal BCE）
  metrics.py             MAE / RMSE / R²
  label_stats.py         标签与全局特征 z-score 统计（两阶段共用）
  train.py               训练 / 微调 / 评估入口（两阶段共用）
  smoke_test.py          最小自检（不读数据）
  audit_data.py          只读数据审计（标签覆盖 / 组成口径 / 家族重叠 / 特征覆盖）
  configs/
    pretrain.yaml        阶段一：全库预训练（global_dim: 8）
    finetune.yaml        阶段二：BaTiO3 族微调（严格 BaTiO3 子集 + doped + 含 Ba 空位；冻结 30 epoch → 解冻 2 块）
    finetune_probe.yaml  对照：线性探针（骨干全程冻结，只训头）
    finetune_scratch.yaml 对照：家族数据从零训练
  stats/                 运行 label_stats.py 后生成 label_stats.json（含 global_feat 统计）
  audit_report.json      运行 audit_data.py 生成的审计报告
  artifacts/             运行后生成，每个 run 一个子目录
```

## 2. 环境与运行方式

- 解释器：`E:\Material_MTL\crystal_env\Scripts\python.exe`
- 工作目录：`E:\Material_MTL`（配置中的相对路径都相对仓库根解析）
- 运行格式：`python Models\<脚本>.py ...`（脚本会把自身目录与 `specific_MT/data_prep` 注入 `sys.path`）

数据侧（本目录只读，不修改）：
- 图数据：`specific_MT/graph/<数据集>/{crystal_graph_partNNNNN.pkl, line_graph_partNNNNN.pkl, index.csv, meta.json}`
- 全局特征旁挂表：`specific_MT/graph/<数据集>/global_feat.csv`（a, b, c, α, β, γ, 体积/原子, 密度），
  由 `specific_MT/data_prep/build_global_feat.py` 生成（不重建图分片）；标准化统计统一写入 label_stats.json
- 标量特征统计：`specific_MT/graph/feature_stats.json`（建图时已标准化，本目录不再处理）

## 3. 快速开始（按顺序）

```powershell
# 0) 代码自检（秒级，不读数据文件）
E:\Material_MTL\crystal_env\Scripts\python.exe Models\smoke_test.py

# 1) 一次性生成 8 维全局特征旁挂表（formation ~1 min / vacancy ~16 s；已存在会跳过）
E:\Material_MTL\crystal_env\Scripts\python.exe specific_MT\data_prep\build_global_feat.py --jobs 8

# 2) 生成标签 + 全局特征统计（默认读 16 个 vacancy 分片，约 1 分钟）
E:\Material_MTL\crystal_env\Scripts\python.exe Models\label_stats.py --config Models\configs\pretrain.yaml

# 3) 数据管线冒烟（1 个分片、200 步；日志含真实步频 step/s）
E:\Material_MTL\crystal_env\Scripts\python.exe Models\train.py --config Models\configs\pretrain.yaml --limit-shards 1 --max-steps 200

# 4) 正式预训练（20 epoch、约 29 万图；想先看曲线可在配置里把 max_epochs 调成 5）
E:\Material_MTL\crystal_env\Scripts\python.exe Models\train.py --config Models\configs\pretrain.yaml

# 5) 微调与两个必修对照（微调依赖预训练的 best.pt）
E:\Material_MTL\crystal_env\Scripts\python.exe Models\train.py --config Models\configs\finetune.yaml
E:\Material_MTL\crystal_env\Scripts\python.exe Models\train.py --config Models\configs\finetune_probe.yaml
E:\Material_MTL\crystal_env\Scripts\python.exe Models\train.py --config Models\configs\finetune_scratch.yaml

# 6) 测试集评估（训练结束会自动跑一次；也可单独跑）
E:\Material_MTL\crystal_env\Scripts\python.exe Models\train.py --config Models\configs\finetune.yaml --eval-only --split test --ckpt Models\artifacts\finetune\best.pt
```

`train.py` 常用覆盖参数：

| 参数 | 作用 |
|---|---|
| `--limit-shards N` | 每个数据源只读前 N 个分片（冒烟 / 小子集拟合测试） |
| `--max-steps N` | 训练步数上限（冒烟） |
| `--device cpu|cuda` | 覆盖配置中的 device（CUDA 不可用会自动回退） |
| `--seed N` | 覆盖随机种子（5 种子集成用） |
| `--dropout p` | 覆盖模型 dropout（拟合测试 0.0 / 正则扫描 0.1、0.3） |
| `--tag 名称` | 输出目录加后缀，避免多次实验互相覆盖 |
| `--out-dir 路径` | 覆盖输出目录 |
| `--ckpt 路径` | `--eval-only` 或微调初始化使用 |
| `--eval-only --split val|test` | 只评估，不训练 |

## 4. 数据语义（当前锁定的四个目标）

| 目标 | 数据源 | 标签列 | 层级 | 量纲 | 掩码规则 |
|---|---|---|---|---|---|
| formation | formation_energy_band_gap / batio3 | `formation_energy_per_atom_eV` | 图级 | eV/atom | 有限值 |
| formation | batio3_doped | `formation_energy_per_atom` | 图级 | eV/atom | 有限值 |
| gap | formation_energy_band_gap / batio3 | `band_gap_eV` | 图级 | eV | 有限值且 `> 0`（金属排除） |
| cbm / vbm | formation_energy_band_gap | `cbm_eV` / `vbm_eV`（相对费米能级） | 图级 | eV | 有限值且 `gap > 0`（与 gap 同口径；覆盖约 55%） |
| vacancy | vacancy_screening | 位点属性 `vacancy` `[N,1]` | **位点级** | eV | 位点有限值（样本位点覆盖 85%） |

说明：
- 统一标签层把四个图级目标物化为 `data.labels [1,4]`（顺序 formation/gap/cbm/vbm，缺失 NaN），空位为 `data.vacancy [N,1]`（缺失 NaN）；拼批后为 `[B,4]` 与 `[ΣN,1]`，由掩码跳过缺失项。
- **gap / cbm / vbm 统一 `gap > 0` 口径**：金属行既不进回归也不进 z-score 统计（`loss_mean_z = 0`）。
- `vacancy_screening` 的 `formation_energy_per_atom` 经审计（与 ehull 仅 2.4% 相同、与 MACE 能量 Spearman 0.51、极值 -11.9 eV/atom 异常）判定为**方法来源不一致**，未接入 formation 头；如需训练请另挂独立头，不可与 MP GGA 形成能混训。
- 介电 / 铁电数据集不在当前训练目标内，未加载。

## 5. 划分与防泄漏

- 一律**按组成分组划分**（`index.csv` 的 `group`/`formula` 列），禁止随机按样本划分；同一组成的多态/多构型只会落在同一个 split。
- **家族由 `finetune.yaml` 自动推导**：把微调各数据源按其 `filters` 过滤后实际用到的组成（归一化 reduced formula）收集为家族集合，预训练（`exclude_family: true`）据此剔除——配置改，家族随之改，不会失同步。当前家族来源：
  - `batio3`：仅保留严格 BaTiO₃ 基（`perovskite_batio3: true`，§1 口径：ABO₃ 计量 + Ba≥50% A + Ti≥50% B；153 → 18）；
  - `batio3_doped`：全部保留（La/Nd/Sr 等掺杂有序相）；
  - `vacancy_screening`：含 Ba 子集（`contains_elements: [Ba]`）；
  - 顶层 `filters.exclude_formulas` 把非钙钛矿系列（BaMg6TiO8、BaMg14TiO16、BaMg30TiO32、Ba2Ti3AlO8、BaMgTi4O8）**移出微调、留在预训练**。
- 预训练（`pretrain.yaml`）：val 2% / test 2%（按组）；剔除数量在启动日志中打印（当前 formation 8,207 / vacancy 8,269）。
- 微调（`finetune.yaml`）：`exclude_family: false`，家族内部按组成再做 val 15% / test 15% 留出。
- 两阶段必须共用 `Models/stats/label_stats.json`：checkpoint 记录其 md5，微调时不一致会直接报错（除非显式 `allow_stats_change: true`，不建议）。

## 6. 模型与损失（对照 §4.2）

- 输入编码：节点 `Linear(149→64)+SiLU+LN`；键 `MLP(48→64→64)`；键角 `Fourier(K=8)→MLP(17→32)`。
- 消息传递 4 × A 块（晶体图卷积 → 线图卷积）：门控消息 `σ(Wf)·softplus(Ws)`，节点/键均残差 + LayerNorm；线图按共享中心原子的键对做均值聚合。
- 池化：`mean ‖ max`（128）+ 8 维标准化全局特征 → **136 维**（`global_dim: 8`，对应 §4.2.4 头结构）。
- 头：`formation`(1)、`gap`(1)、`edges`(cbm/vbm 双输出)、`vacancy`(节点级)；`metal` 分类头可选（`use_metal`，默认关闭）。
- 损失：z 空间 SmoothL1（β：formation 0.05，其余 0.1）；权重 formation 1.0 / gap 1.0 / cbm 0.5 / vbm 0.5 / vacancy 1.0 / metal 0.2；一致性项 `gap ≈ cbm − vbm` 权重 0.1，按总步数前 5% ramp-up；空位项用 **mean** 聚合。
- 采样：`sampling: sqrt_inv`（批构成比例 `∝ 1/sqrt(N_t)`，最大余数法分配）；可切 `proportional`。
- 两阶段冻结：默认 `freeze_epochs: 30` 全冻骨干只训头 → 解冻最后 `unfreeze_last_blocks: 2` 个 A 块（backbone lr 1e-5）；探针 = 永久冻结 + 只训头；scratch = 不加载权重、全参 3e-4。

## 7. 输出产物

每个 run 输出到 `Models/artifacts/<run>/`：

| 文件 | 内容 |
|---|---|
| `best.pt` / `last.pt` | 权重 + 模型配置 + 训练元信息（stage、epoch、stats md5、val 指标） |
| `metrics.csv` | `epoch, step, split, source, target, n, mae, rmse, r2`（`source=__overall__` 为按目标汇总） |
| `losses.csv` | `epoch, step, split, term, value, share`（逐项损失与占比，用于查单头主导） |
| `<split>_metrics.json` | `--eval-only` 或训练结束的测试集指标 |

指标一律**物理单位**（还原 z-score）：formation 为 eV/atom，gap/cbm/vbm/vacancy 为 eV；`best.pt` 按验证集总损失（加权）选择，早停同指标。

统计文件 `Models/stats/label_stats.json` 另含：四目标 + vacancy 位点 + `global_feat` 的 mean/std、`loss_mean_z`（输出头 bias 初始化）与 population 元信息（train 划分、家族剔除、分片数）。

## 8. 诊断与实验清单（§4.2.9 / §4.2.7）

| 实验 | 命令 / 配置 |
|---|---|
| 100–200 条小子集拟合测试 | `train.py --config ...\finetune.yaml --limit-shards 1 --dropout 0.0 --tag fit_test`（对 batio3 源） |
| 头容量扫描 | 配置里改 `model.head_hidden1 / head_hidden2`（默认 128/64；瘦身 64/32；线性头 `head_hidden1: null`） |
| 正则扫描 | 同配置加 `--dropout 0.1 / 0.3` |
| 5 种子集成 | `--seed 7 --tag seed7` 等 |
| 线性探针 / 从零基线 | `finetune_probe.yaml` / `finetune_scratch.yaml` |
| 描述符基线（GPR/RF） | 未包含在本目录，后续单独做 |

判读口径：逐头看训练/验证曲线；`losses.csv` 单项占比长期 > 70% 时先调采样/权重；多种子一致地差优先查数据/掩码/划分，而不是优化器。

## 9. 已知差异与决策记录

1. **136 已按文档落地**：8 维全局特征（a, b, c, α, β, γ, 体积/原子, 密度）以旁挂表 `global_feat.csv` 实现（不重建 62 GB 图分片），`data.py` 在线标准化后拼在池化向量后，头输入 = 128 + 8 = 136；两阶段共用同一套统计（label_stats.json）。
2. **家族口径**：不再是硬编码名单，而是**从 `finetune.yaml` 推导**（见 §5）；改过滤即改家族。
3. **`use_metal` 默认关闭**：金属识别暂不做；gap 头由 `gap > 0` 掩码学习非金属子集。
4. **cbm/vbm 与 gap 同口径**：审计发现 6,267 条"gap=0 但 cbm/vbm 有限"的行，已统一排除（约 -7% 样本）。
5. **line_edge_index 批偏移**：由 `graph_schema.CrystalData.__inc__` 处理（按 `num_edges` 偏移），模型侧**不要**重复偏移；`smoke_test.py` 有专门断言。
6. **分片流式**：每 epoch 顺序读取全部分片（formation 78 + vacancy 77，约 61 GB）；实测步频 5–8 step/s（1 分片冒烟），全量 20 epoch 约 3.5–7 小时；未实现后台预取与断点续训。
7. **`batio3` 的 `group` 列为空**：loader 已用 `formula` 回退（组成分组/家族匹配因此恢复正确）。

## 10. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `缺少标签统计 ...；先运行 python Models\label_stats.py` | 未生成 `Models/stats/label_stats.json` |
| `微调必须与预训练共用同一份 label_stats` | 统计文件与 checkpoint 记录的 md5 不一致；重新用预训练配置生成，或删除 `allow_stats_change` 思路（不建议开启） |
| `--eval-only 需要 --ckpt 或配置中的 init_from` | 评估必须指定权重 |
| `line_edge_attr 形状 ... 期望 [L,1]` | 图数据不是 `--line-mode angle` 建的；用 `build_all_graphs.py --line-mode angle` 重建 |
| `CUDA 不可用，回退 CPU` | 正常回退；检查驱动或改用 `--device cpu` |
| `缺少 ...\global_feat.csv；先运行 python specific_MT\data_prep\build_global_feat.py` | 未生成全局特征旁挂表（或 `global_dim>0` 时缺表） |
| `配置 global_dim>0，但 label_stats 缺 global_feat 统计` | 生成旁挂表后重新运行 label_stats.py |
| 单个原子/零键角样本 | `vacancy_screening` 存在 39 条 1 原子原胞（如 O84），按你的要求**默认保留** |
| 空位标签部分缺失（如 mp-1182332） | loader 自动补 NaN 并掩码，不影响训练 |

## 11. 对接课程节点

- 项目 0（属性预测）：本目录四目标多任务 GNN 与其对照实验
- 项目 2.5（遗传搜索）：用 `best.pt` 做筛选/适应度时需要**新增推理脚本**（当前未提供），输入掺杂结构 → 预测四目标
- 项目 4（主动学习闭环）：集成方差（多种子）可作为不确定度来源，当前未实现

## 12. 数据审计要点（`audit_data.py`，报告见 `audit_report.json`）

```powershell
E:\Material_MTL\crystal_env\Scripts\python.exe Models\audit_data.py --vacancy-shard-sample 1 --json audit_report.json
```

- **formation_energy_band_gap**：154,030 行 / 153,877 唯一 mid（153 条完全重复行，数值一致，无害）；金属占比 46.9%；`gap_from_edges` 与 `cbm−vbm`、`band_gap` 与 `cbm−vbm` 一致率 100%（|Δ|≤0.05 eV）；6,267 条金属行带 cbm/vbm，已按 `gap>0` 口径排除。
- **vacancy_screening**：位点标签覆盖 85.0%；位点能量重尾（|v|>10 占 2.4%，负值 5.3%）；`formation_energy_per_atom` 与 ehull 不等价（2.4%），未接入。
- **batio3**：153 行中仅 25 行满足 ABO₃ 计量、18 行满足严格 BaTiO₃ 基（微调只取这 18 行）；`e_total` 仅 5 行（介电缺口）。
- **家族重叠**（当前）：formation 剔除 8,207 / vacancy 剔除 8,269 / batio3_doped 剔除 252（`batio3` 的微调子集 18 行）。
- **global_feat 覆盖**：四个数据集 100% 覆盖、无 NaN。
