# data/ 数据说明

> 全部数据集已按【数据类别】命名，共 12 个文件夹、约 6.2 GB。性质若为独立文件则独立文件夹存放；整理版/快照并入来源目录。
> 解释器：`E:\Material_MTL\crystal_env\Scripts\python.exe`
> 详细字段字典 → `README_详细版.md`；机器可读总览 → `数据总览.json`

---

## 一、数据集清单（全部已下载完成）

| 文件夹 | 数据类别 | 来源 | 规模 |
|---|---|---|---|
| `全库_形成能与带隙` | 形成能 / 带隙 / 稳定性 | MP 全库（整理版 + 原始快照） | 整理版 **153,877 条**（stable 32,643 / metastable 68,261 / unstable 52,969）；原始快照 210,579 条 |
| `全库_带边` | 带边位置 CBM/VBM | MP 电子结构库 | **153,573 条**（87,926 条含 CBM/VBM） |
| `全库_介电` | 介电常数（DFPT） | MP dielectric | **7,332 条** |
| `全库_空位形成能` | 空位形成能 | 2025 通用 MLIP 筛选（Zenodo） | **86,259 种材料**（Vacancies.json 378 MB + Vacancies.db 2.35 GB） |
| `氧空位形成能_ABO3` | 氧空位形成能 | Scientific Data 2017（Emery & Wolverton） | **5,329 条**（4,914 条含 OV 能） |
| `BaTiO3_结构_形成能_带隙_稳定性` | BaTiO3 基础表 | MP summary | **153 条**（含结构） |
| `BaTiO3_介电` | BaTiO3 介电（e_total/e_ionic/e_electronic/n） | MP dielectric | 5 条 |
| `BaTiO3_弹性` | BaTiO3 弹性（杨氏模量等） | MP elasticity | 15 条 |
| `BaTiO3_磁性` | BaTiO3 磁性（磁矩/磁序） | MP magnetism | 153 条 |
| `BaTiO3_热力学` | BaTiO3 热力学（形成能/分解产物等） | MP thermo | 325 条 |
| `BaTiO3掺杂_结构_形成能_凸包能量` | BaTiO3 掺杂快照 | MP summary | 2 个 pkl（252 条 + 8 条）+ 整理版 CSV/Parquet |
| `铁电材料_极化` | 铁电极化 | MPContribs 铁电数据库 | **641 条** + 2,408 个结构（含整理版 CSV/Parquet） |

## 二、按用途分

**预训练（大规模）**
- `全库_形成能与带隙`：含结构 + 形成能 + 带隙 + 稳定性
- `全库_带边`：CBM/VBM/Efermi（相对费米能级）
- `全库_介电`：DFPT 介电常数（e_total/e_ionic/e_electronic/n）
- `全库_空位形成能`：86,259 材料的空位形成能（MLIP 计算，含结构坐标）
- 全库原始快照：`全库_形成能与带隙/原始快照_summary_2025-09-25/`（210,579 条全字段 JSONL，含介电/弹性/功函数等 67 列，备份与补字段用）

**微调（BaTiO3 / 铁电 / 氧空位）**
- `BaTiO3_结构_形成能_带隙_稳定性`、`BaTiO3_介电`、`BaTiO3_弹性`、`BaTiO3_磁性`、`BaTiO3_热力学`、`BaTiO3掺杂_结构_形成能_凸包能量`
- `铁电材料_极化`（极化值、带隙、能量差）
- `氧空位形成能_ABO3`（ABO3 钙钛矿 OV 形成能）

**派生**
- 铁电整理版（labels_unified、records_fields、structure_index）→ `铁电材料_极化/`
- 掺杂快照整理版（CSV/Parquet/结构索引）→ `BaTiO3掺杂_结构_形成能_凸包能量/`

## 三、注意事项

1. 形成能/带隙为 GGA/GGA+U 计算值，仅作定性初筛；带边位置相对**费米能级**，非真空能级。
2. `全库_空位形成能` 的 `Vacancies.json`（JSON）与 `Vacancies.db`（SQL，可用 ASE 浏览）内容相同、格式不同。
3. 空位形成能来自通用机器学习势计算，非 DFT；与 Emery 数据集的 DFT 值混用时注意区分方法。
4. 预训练与微调数据若同组成，必须按 `split_group`（reduced_formula）分组隔离，防止泄漏。
5. 每个数据集目录内有 `*_meta.json` 记录查询日期、API 版本与字段缺失；总数字以 `数据总览.json` 为准。

## 四、常用命令

```powershell
cd E:\Material_MTL\模型具体训练

# 数据总览（全部数据集规模统计）
& E:\Material_MTL\crystal_env\Scripts\python.exe audit_all_data.py

# 查看 / 更新以下数据时：
& E:\Material_MTL\crystal_env\Scripts\python.exe mp_gap_dataset.py            # 全库形成能+带隙（续传）
& E:\Material_MTL\crystal_env\Scripts\python.exe mp_band_edges.py --labels "data\全库_形成能与带隙\materials.csv" --out "data\全库_带边"
& E:\Material_MTL\crystal_env\Scripts\python.exe download_batio3_all.py       # BaTiO3 全性质
& E:\Material_MTL\crystal_env\Scripts\python.exe download_dielectric.py       # 全库介电
& E:\Material_MTL\crystal_env\Scripts\python.exe download_zenodo_vacancies.py # 空位形成能（Zenodo，可续传）
& E:\Material_MTL\crystal_env\Scripts\python.exe organize_existing.py         # 整理（并入各数据集目录）
```
