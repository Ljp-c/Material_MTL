from mp_api.client import MPRester
import pandas as pd
dopants = ["La","Nd","Sm","Pb","Sr","K","Ca","Nb","Zr","Sn","Hf","Fe","Co","Ni","Cr","Mn","Mg","Al","Sc","V","Si","Y","Gd","Dy","Er","Ho","Tb","Eu","Li","Bi"]

target_spacegroups= ["Pm-3m","P4mm","Amm2","R3m","P6_3/mmc"]
all_data = []

with MPRester() as mpr:
    # 2. 双重循环：遍历每一种组合
    for dopant in dopants:
        # 动态构建化学体系字符串，例如 "Ba-Sr-Ti-O"
        # 注意：元素必须按字母顺序排列
        elements = sorted(["Ba", "Ti", "O", dopant])
        chemsys = "-".join(elements)
        
        for sg in target_spacegroups:
            try:
                docs = mpr.materials.summary.search(
                    chemsys=chemsys,
                    spacegroup_symbol=sg,
                    fields=[
                        "material_id", "formula_pretty", "structure",
                        "formation_energy_per_atom", "energy_above_hull"
                    ]
                )
                if docs:
                    print(f"找到 {len(docs)} 条: {chemsys} + {sg}")
                    for doc in docs:
                        all_data.append({
                            "material_id": doc.material_id,
                            "formula": doc.formula_pretty,
                            "structure": doc.structure,
                            "formation_energy_per_atom": doc.formation_energy_per_atom,
                            "energy_above_hull": doc.energy_above_hull,
                            "dopant": dopant,
                            "queried_spacegroup": sg
                        })
            except Exception as e:
                print(f"查询失败 {chemsys} + {sg}: {e}")


df = pd.DataFrame(all_data)


if not df.empty:
    df = df.drop_duplicates(subset=["material_id"], keep="first")
    print(f"\n总计获得 {len(df)} 条唯一材料数据")
    df.to_pickle("BaTiO3_doped_multi.pkl")
else:
    print("未找到任何数据，请检查元素组合或空间群设置")