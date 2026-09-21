# data/ 数据集说明（数据字典 · 来源 · 校准 · ML 使用建议）

> 工程位置：`E:\Material_MTL\模型具体训练`
> 解释器：`E:\Material_MTL\crystal_env\Scripts\python.exe`
> 更新日期：2026-09-19（全部数据下载完成并已按数据类别重命名）

> **最新数据集清单（2026-09-19）**：
>
> | 文件夹 | 规模 | 关键文件 |
> |---|---|---|
> | `全库_形成能与带隙` | 153,877 条 | materials.csv/.parquet、structures.jsonl、dataset_meta.json |
> | `全库_带边` | 153,573 条（87,926 含 CBM） | band_edges.csv/.parquet |
> | `全库_介电` | 7,332 条 | dielectric_labels.csv、structures.pkl、dielectric_dataset.pkl |
> | `全库_空位形成能` | 86,259 种材料 | Vacancies.json（378 MB）、Vacancies.db（2.35 GB）、2D_layers.json/.db、README.txt |
> | `氧空位形成能_ABO3` | 5,329 条（4,914 含 OV） | Emery 2017 CSV（列：Formation energy / Stability / Band gap / Vacancy energy [eV/O atom] 等 21 列） |
> | `BaTiO3_结构_形成能_带隙_稳定性` | 153 条 | materials.csv/.parquet、structures.jsonl/.pkl |
> | `BaTiO3_介电` | 5 条 | dielectric.csv/.parquet |
> | `BaTiO3_弹性` | 15 条 | elasticity.csv/.parquet |
> | `BaTiO3_磁性` | 153 条 | magnetism.csv/.parquet |
> | `BaTiO3_热力学` | 325 条 | thermo.csv/.parquet |
> | `BaTiO3掺杂_结构_形成能_凸包能量` | 252 + 8 条 | 2 个 pkl（含 structure） |
> | `铁电材料_极化` | 641 条 + 2,408 结构 | labels.csv、records.jsonl、structures.pkl |
> | `全库_形成能与带隙/原始快照_summary_2025-09-25/` | 210,579 条 | AWS 全字段快照（原 mp_full，已并入） |
>
> 整理版文件（labels_unified、records_fields、structure_index 等）已并入对应数据集目录。
>
> 各数据集字段与来源见下文各节；关键数字以 `data/数据总览.json` 和各自 `*_meta.json` 为准。

---

## 0. 目录总览

| 路径 | 内容 | 状态 |
|---|---|---|
| `data/ferroelectrics/` | MPContribs 铁电数据库（Sci Data 2020 + npj 2023 两个子库） | 已下载：641 条记录 / 2408 个 CIF / structures.pkl |
| `data/BaTiO3掺杂_结构_形成能_凸包能量/` | BaTiO3 类掺杂材料的 MP summary 快照（2 个 pkl + 整理版 CSV/Parquet） | 已下载；字段以同级 `audit.json` 实测为准 |
| `data/gap_dataset/` | 带隙 + 热力学稳定性数据集（`mp_gap_dataset.py` 产出） | 待运行生成 |
| `data/gap_dataset/band_edges/` | 带边位置 CBM/VBM/Efermi（`mp_band_edges.py` 产出） | 待运行生成 |
| 整理版文件 | 由 `organize_existing.py` 生成，已并入 `铁电材料_极化/` 与 `BaTiO3掺杂_结构_形成能_凸包能量/` | 已生成 |
| `data/README.md` | 本文件 | — |

处理脚本（位于 `模型具体训练/`）：

| 脚本 | 作用 |
|---|---|
| `download_mp_s3.py` | AWS 直下全库 collection（summary/dielectric/...，JSONL.gz），预训练数据一次性下载 |
| `mp_gap_dataset.py` | 带隙 + 稳定性 + 结构数据集（**默认全库**；`--batio3` 为掺杂子集），支持断点续传 |
| `mp_band_edges.py` | 补充带边位置；可选导出完整能带 JSON |
| `organize_existing.py` | 审计并整理 `data/ferroelectrics` 与 `data/BaTiO3` |

---

## 1. 数据集 A：MPContribs 铁电数据库（ferroelectrics）

### 1.1 来源

- 子库 `ferroelectrics`：Sci Data 2020，doi:10.1038/s41597-020-0407-9（255 条含极化值记录）
- 子库 `ferroelectrics_ext`：npj Comput Mater 2023，doi:10.1038/s41524-023-01193-3（386 条含极化值记录）
- 通过 `mp-api` 的 MPContribs 接口下载（`download_ferroelectrics_db.py`），查询日期即本地文件生成时间 2026-09-18。

### 1.2 文件清单

| 文件 | 说明 |
|---|---|
| `labels.csv` | 已抽取的标签表（641 行数据），但两个子库列名不同（小写/大写各一套） |
| `records.jsonl` | 原始完整记录，`data` 字段含 30+ 个键值（值为“数字 + 单位”字符串） |
| `structures.pkl` | `dict[key → pymatgen.Structure]`，key 形如 `{project}::{identifier}::{structure_name}` |
| `cif/` | 2408 个 CIF；每个材料的 5 种结构变体：`orig_polar_structure`、`orig_nonpolar_structure`、`distortion.high_symm`、`distortion.low_symm`、`distortion.high_low_setting` |

### 1.3 字段字典

**`labels.csv` 原始列**（两套命名按 project 分行填充，另一套为空）：

| 列 | 含义 | 单位 | 来源 |
|---|---|---|---|
| `polarization_uC_cm2` | 自发极化模长 | µC/cm² | 两个子库 |
| `polar_spacegroup` / `Polar_spacegroup` | 极性相空间群 | 编号或 HM 符号 | 两子库分别 |
| `nonpolar_spacegroup` / `Nonpolar_spacegroup` | 非极性相空间群 | 编号或 HM 符号 | 两子库分别 |
| `bandgap_polar` / `Polar_bandgap` | 极性结构带隙 | eV | 两子库分别 |
| `bandgap_nonpolar` / `Nonpolar_bandgap` | 非极性结构带隙 | eV | 两子库分别 |
| `energy_diff` | 极化/非极化能量差 | eV/atom（负值 = 极化相更稳） | `energy\|diff` |
| `Energy_diff` | 同上（ext 子库） | **原始值，单位未在数据中标注，须核对原文** | ext |
| `workflow_status` / `workflow_category` | 工作流状态/类别 | — | 子库 1 |
| `CL_score` / `MAD_pseudo` / `MAD_relax` / `F_score` | 高质量筛选指标（定义见 npj 论文） | — | 子库 2 |

**`铁电材料_极化/labels_unified.csv`（由 `organize_existing.py` 生成）**：
把上述两套列合并为统一列（`polar_spacegroup`、`bandgap_polar_eV`、`energy_diff_eV_per_atom`、`energy_diff_ext_raw` 等），
并增加 `source_lib`、`reduced_formula`、`split_group`（= reduced formula，用于分组划分）。

**`铁电材料_极化/records_fields.csv`（展开自 records.jsonl）**，主要列：

| 列 | 含义 | 单位 |
|---|---|---|
| `polarization_norm_uC_cm2` | 极化模长 | µC/cm² |
| `polarization_c_axis_uC_cm2` | 极化 c 轴分量 | µC/cm² |
| `bandgap_polar_eV` / `bandgap_nonpolar_eV` | 带隙 | eV |
| `energy_diff_eV_per_atom` | 极化-非极化能量差 | eV/atom |
| `distortion_dmax_A` / `_before_A` / `_after_A` | 最大原子位移（畸变幅度） | Å |
| `distortion_delta` / `distortion_s` / `distortion_dav_A` | 畸变统计量 | 无量纲 / Å |
| `pol_smoothness_index` / `pol_smoothness_max_uC_cm2` | 极化曲线平滑度 | — / µC/cm² |
| `pol_jumps_max_uC_cm2` / `pol_jumps_index` | 极化跳变检查 | µC/cm² / — |
| `energies_smoothness_eV_per_atom` / `energies_jumps_max_eV_per_atom` | 能量曲线检查 | eV/atom |
| `pol_quanta_a/b/c_uC_cm2`、`pol_vector_a/b/c_uC_cm2` | 极化量子/矢量分量 | µC/cm² |
| `workflow_status` / `workflow_category` | 工作流状态 | — |
| `polar_mpid` / `nonpolar_mpid` | 对应 MP material_id（**可用于与 MP 库关联**） | — |
| `bilbao_sg_polar` / `bilbao_sg_nonpolar` | Bilbao 空间群编号 | — |

**`铁电材料_极化/structure_index.csv`**：每个结构一行——
`key, source, project, identifier, structure_name, n_sites, formula, reduced_formula, a_A..., volume_A3, density_g_cm3, spacegroup_number, spacegroup_symbol, point_group, is_polar_pointgroup, symmetry_error`。

### 1.4 已知限制（重要）

1. 极化值为 **DFPT/Berry 相位计算的电子极化**，不是实验 ε_r；两者语义不同，禁止混入同一回归目标。
2. 两个子库预加工程度不同：`ferroelectrics` 有 workflow 状态与 `energy|diff`（eV/atom），`ferroelectrics_ext` 的 `Energy_diff` 单位未标注（候选为 meV/atom，**未经核实不得当 eV/atom 使用**）。
3. 同一化学式有多条记录（如 SiO2、WO3、PH3O4），随机划分会泄漏——必须按 `split_group` 分组。
4. MPContribs 数据的泛函为 PBE 系，带隙系统性低估，仅作定性参考。
5. pkl 与 pymatgen 版本绑定；跨环境加载可能失败（本工程环境 pymatgen 2026.5.4）。

---

## 2. 数据集 B：BaTiO3 掺杂 MP 快照（data/BaTiO3/）

### 2.1 来源与查询口径

由 `dataDownload.py` 生成：对 30 个掺杂元素 × 5 个目标空间群（`Pm-3m, P4mm, Amm2, R3m, P6_3/mmc`），
用 `summary.search(chemsys="Ba-O-Ti-X", spacegroup_symbol=...)` 拉取 `material_id / formula_pretty / structure / formation_energy_per_atom / energy_above_hull`，
按 `material_id` 去重。

| 文件 | 状态 |
|---|---|
| `BaTiO3_doped_multi.pkl` | 主产物；时间 2026-09-18 00:51 |
| `BaTiO3_doped_multi(rough).pkl` | 中间版本；时间 2026-09-18 01:53；推测为带结构与不带结构两版之一，**具体差异以 `processed/audit.json` 实测为准** |

### 2.2 已知限制

1. **空间群白名单过滤**（只保留 5 个对称性）会漏掉低对称掺杂相，做全空间群补充查询更稳。
2. 只查了 30 个指定掺杂元素；未覆盖 Hf 之外的潜在元素。
3. 不含带隙、介电、磁性字段；缺失值以 `None` 表示，未编造。
4. `structure` 列是否存在于两个 pkl 中、行数多少，请以 `organize_existing.py --audit-only` 输出为准。

---

## 3. 数据集 C：MP 带隙 + 稳定性数据集（脚本生成）

> 采集范围：**默认全库**（阶段一预训练用）；`--batio3` 只取 BaTiO3 掺杂家族（阶段二微调域）。
> 全库下载另有更快的 AWS 直连脚本 `download_mp_s3.py`（见第 4 节）。

### 3.1 查询方案（字段映射）

| 目标 | endpoint | 字段 | 单位/说明 | 可信度 |
|---|---|---|---|---|
| 带隙 | `materials.summary` | `band_gap` | eV | C（GGA/GGA+U） |
| | | `is_metal`, `is_gap_direct`, `band_gap_type`(若有) | — / boolean | 分类可靠 |
| 稳定性 | `materials.summary` | `formation_energy_per_atom` | eV/atom | B |
| | | `energy_above_hull` | eV/atom；=0 稳定 | B |
| | | `is_stable` (若有) | boolean | B |
| 结构 | `materials.summary` | `structure` | pymatgen → `structures.jsonl` | — |
| | | `symmetry`, `volume`, `density`, `nsites` | — | — |
| 磁性 | `materials.summary` | `is_magnetic`, `ordering`, `total_magnetization`, `num_magnetic_sites` | µB 等 | C |
| 带边 | `materials.electronic_structure` | `cbm` / `vbm` / `efermi` | eV，**相对费米能级** | C |
| | | `band_structure`（可选） | 完整能带 + 投影 | C |
| 泛函/U 值 | — | **MP API 未暴露**，列置 None | — | — |

> 脚本在启动时自动探测 API 实际支持的字段，请求缺失字段会记录到 `dataset_meta.json` 的 `fields_missing_from_api`。

### 3.2 `materials.csv` 字段字典（核心列）

| 列 | 含义 | 单位 |
|---|---|---|
| `material_id` / `formula` / `chemsys` / `elements` | MP 标识与化学信息 | — |
| `nsites`, `nelements` | 原子数 / 元素数 | — |
| `volume_A3`, `density_g_cm3` | 晶胞体积、密度 | Å³, g/cm³ |
| `spacegroup_number`, `spacegroup_symbol`, `crystal_system`, `point_group` | 对称性信息 | — |
| `band_gap_eV` | GGA/GGA+U 带隙 | eV |
| `is_gap_direct`, `is_metal` | 直接带隙 / 金属 | boolean |
| `formation_energy_per_atom_eV` | 形成能 | eV/atom |
| `energy_above_hull_eV_per_atom` | 凸包之上能量 | eV/atom |
| `stability_class` | `stable`(≤1e-5) / `metastable`(≤0.1) / `unstable` | — |
| `is_stable`, `theoretical`, `deprecated`, `last_updated` | MP 元信息 | — |
| `is_magnetic`, `magnetic_ordering`, `total_magnetization`, `num_magnetic_sites` | 磁学信息 | — |
| `has_structure` | 是否带结构 | boolean |
| `has_band_gap` | 非金属且有带隙值 | boolean |
| `band_gap_trust` | `none` / `C_gga` / `B_hse` / `B_calibrated` / `A_gw_or_exp` | — |
| `preferred_stability` | stable/metastable | boolean |
| `preferred` | 非废弃 + 有结构 + stable/metastable + 有带隙 | boolean |
| `polymorph_count` | 同 reduced_formula 的多态条目数 | — |
| `split_group` | 分组划分键（= reduced_formula） | — |
| `missing_fields` | 缺失的数值标签清单 | — |
| `query_date` | 查询日期（ISO，UTC） | — |

校准表合并后新增：`band_gap_calibrated_eV`、`band_gap_calib_method`、`band_gap_calib_ref`。
校准表要求列：`material_id, band_gap_eV, method, reference`（method 含 `HSE`→B 级、`GW`/`exp`→A 级）。

### 3.3 `band_edges.csv` 字段（带边）

| 列 | 含义 | 单位 |
|---|---|---|
| `cbm_eV` / `vbm_eV` / `efermi_eV` | 导带底 / 价带顶 / 费米能级 | eV（相对费米能级） |
| `cbm_json` / `vbm_json` / `efermi_json` | 原始对象（含 k 点等，若有） | — |
| `has_band_structure` | 该材料是否导出/含完整能带 | boolean |
| `gap_from_edges_eV` | cbm − vbm | eV |
| `gap_check_ok` | 与 `band_gap_eV` 自洽（<0.05 eV） | boolean |

### 3.4 版本、日期与校准

- MP 数据库版本、`mp-api`/`pymatgen` 版本、查询日期、字段缺失清单：见 `data/gap_dataset/dataset_meta.json` 与 `band_edges_meta.json`。
- **校准方法**：`band_gap_eV` 默认标 `C_gga`（GGA/GGA+U 仅定性）；用 `--calib-csv` 合并 HSE06/GW/实验值后升级为 `B_hse` / `A_gw_or_exp`。
- 未获取的字段（Hubbard U、自旋配置）一律为 `None`，原因是 MP API 未暴露计算输入参数。

---

## 4. 运行顺序（供复现）

```powershell
cd E:\Material_MTL\模型具体训练
& E:\Material_MTL\crystal_env\Scripts\python.exe organize_existing.py --audit-only   # 第一步：审计现有数据
& E:\Material_MTL\crystal_env\Scripts\python.exe organize_existing.py                # 整理（并入各数据集目录）

# 第二步：全库数据（预训练）——以下两条路线二选一：
& E:\Material_MTL\crystal_env\Scripts\python.exe download_mp_s3.py --collection summary --list-only
& E:\Material_MTL\crystal_env\Scripts\python.exe download_mp_s3.py --collection summary
& E:\Material_MTL\crystal_env\Scripts\python.exe mp_gap_dataset.py                    # 或走 API 全库（默认）

# 第三步：BaTiO3 掺杂子集（微调域）与带边补充
& E:\Material_MTL\crystal_env\Scripts\python.exe mp_gap_dataset.py --batio3
& E:\Material_MTL\crystal_env\Scripts\python.exe mp_band_edges.py --labels data\gap_dataset\materials.csv
```

所有脚本支持缓存续传：中断后重跑会跳过已完成的分块；`--force` 才会重下。

---

## 5. 面向机器学习的建议

### 5.1 推荐标签

| 数据集 | 主标签 | 辅助标签 | 注意 |
|---|---|---|---|
| ferroelectrics | `polarization_uC_cm2`（回归） | `bandgap_polar_eV`、`energy_diff_eV_per_atom` | 极化长尾（0.002–90 µC/cm²）→ `log1p` 变换或分桶 |
| BaTiO3 快照 | `formation_energy_per_atom_eV` | `energy_above_hull_eV_per_atom` | 小样本，适合 GPR/RF 基线 |
| gap_dataset | `band_gap_eV`（C 级） | `stability_class`（分类） | 建议只用 `preferred` 子集做筛选实验 |

### 5.2 推荐特征

- 成分：元素分数 + 元素属性统计（电负性、离子半径、原子质量、族/周期）。
- 结构：`a/b/c`、体积、密度、空间群、最近邻距离、配位数、c/a 四方性、Ti 偏心位移、氧八面体畸变。
- 掺杂元数据（BaTiO3 家族）：x_A/x_B、电荷不平衡、空位浓度。
- 图表示：CGCNN 风格（原子序数嵌入 + 距离 RBF 边特征），已有 `build_graphs.py`。

### 5.3 划分策略（硬性）

按组成分组，禁止随机划分。所有整理产物都带 `split_group` 列，可直接用：

```python
from sklearn.model_selection import GroupShuffleSplit
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
train_idx, test_idx = next(gss.split(df, groups=df["split_group"]))
```

- 单独报告：全测试集、BaTiO3 家族留出集、成分外推集。
- 长尾指标：MAE / RMSE / R² + log 空间指标；金属/非金属分类另报准确率。
- **泄漏红线**：预训练集（全库/T1）若含与微调/测试同组成的 BaTiO3 或掺杂结构，必须按 `split_group` 隔离或剔除。

### 5.4 基线模型（由弱到强）

1. `DummyRegressor(strategy="mean")` —— 底线。
2. Ridge / 随机森林 / `HistGradientBoostingRegressor` —— 成分+结构描述符。
3. CGCNN 风格 GNN —— 结构敏感任务（`build_graphs.py` 管线）。
4. 两阶段迁移：大规模晶体预训练（T1/全库）→ BaTiO3 掺杂微调（T2/T3），分层学习率、先冻结后解冻；对照线性探针与从零训练基线。

### 5.5 标签可信度分级

| 级别 | 定义 | 典型来源 |
|---|---|---|
| A | GW / 实验值 | 外部校准表 `method=GW/exp` |
| B | HSE06 或 DFPT 全收敛 | `--calib-csv` 合并；MP DFPT 介电 |
| C | GGA/GGA+U / PBE 定性 | MP `summary.band_gap`、带边（相对费米能级） |
| none | 缺失 | 字段为 `None`（未编造） |

---

## 6. 待确认项（跑完审计脚本后回填）

1. `data/BaTiO3/*.pkl` 的实际行数、列、是否含 `structure`（见 `processed/audit.json`）。
2. `records.jsonl` 与 `labels.csv` 行数是否一一对应，以及每列非空计数。
3. `ferroelectrics_ext` 的 `Energy_diff` 单位（核对 npj Comput Mater 2023 原文后更新本文件）。
