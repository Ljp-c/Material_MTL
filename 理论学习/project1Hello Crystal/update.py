from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import numpy as np

# 用我们之前的 Si 结构，或者读取你自己的 CIF
from pymatgen.core import Lattice
si = Structure(
    lattice=Lattice.cubic(5.43),
    species=["Si", "Si"],
    coords=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]
)

#=================获取空间位置信息===============
sga = SpacegroupAnalyzer(si)
print("="*60)
print("获取空间群信息")
print("="*60)
print(f"空间群国际符号：{sga.get_space_group_symbol()}")
print(f"空间群编号：{sga.get_space_group_number()}")
print(f"晶系：{sga.get_crystal_system()}")
print(f"点群：{sga.get_point_group_symbol()}")

#=================获取wyckoff位置信息===============

print("\n"+("="*60))
print("获取Wyckoff位置信息")
print("="*60)

sym_data=sga.get_symmetry_dataset()
wyckoff=sym_data['wyckoffs']
equivalent_atoms=sym_data['equivalent_atoms']

print(f"{'原子索引':10} {'元素':<8} {'分数坐标':<10} {'Wyckoff':<10} {'等价组':<10}")
print("="*60)

for i,site in enumerate(si):
    elem=site.specie.symbol
    frac=site.frac_coords
    wyck=wyckoff[i]
    eq_group=equivalent_atoms[i]
    print(f"{i:<10} {elem:<8} {frac[0]:.4f},{frac[1]:.4f},{frac[2]:.4f} {wyck:<10} {eq_group:<10}")

#=================wyckoff位置信息多重性================
print("\n"+("="*60))
print("wyckoff位置信息多重性")
print("="*60)

from collections import Counter
wyckoff_counts = Counter(wyckoff)
for wyck,count in wyckoff_counts.items():
    print(f"Wyckoff位置'{wyck}':出现{count}次")

print(f"\n总原子数:{len(si)}")
print(f"独立wyckoff位置数:{len(wyckoff_counts)}")
#===========标准晶胞与原始晶胞对比==============

print(f"\n"+"="*60)
print("标准晶胞")
print("="*60)

primitive = sga.get_primitive_standard_structure()
print(f"原始原子数:{len(si)}")

print(f"标准晶胞原子数:{len(primitive)}")

print(f"空间群不变，但晶胞可能更小")