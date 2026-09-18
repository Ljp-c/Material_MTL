from pymatgen.core import Structure, Lattice
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

""" 读取CIF文件"""


# si=Structure(
#     lattice=Lattice.cubic(5.43),#晶体结构
#     species=["Si","Si"],      #原子数量以及种类
#     coords=[[0,0,0],[0.25,0.25,0.25]] #原子坐标 分别按先后顺序进行列出
# )

structure = Structure.from_file("C:\\Users\\Dr.刘\\OneDrive\\Desktop\\新建文件夹\\project1Hello Crystal\\Zn(InS2)2.cif")

print("="*50)
print(" 原子坐标表格 ")
print("="*50)
print(f"{'Index':<8}{'Element':<10}{'a':<10}{'b':<10}{'c':<10}")
print("-"*50)

for i,site in enumerate(structure):#遍历该晶体中的每一个原子
    element = site.specie.symbol #获取当前原子的种类
    frac = site.frac_coords      #获取当前原子的分数坐标
    print(f"Atom{i} ({element}): ({frac[0]:.3f},{frac[1]:.3f},{frac[2]:.3f})")
                                #格式化打印原子编号、元素名、三个坐标值（各保留三位小数）

""" 从分数坐标转化为笛卡尔坐标"""
print("="*50)
print(" 分数坐标转化为笛卡尔坐标 ")
print("="*50)

cart_coords = structure.cart_coords #获取笛卡尔坐标
for i,(site,cart) in enumerate(zip(structure, cart_coords)):
    element = site.specie.symbol #获取当前原子的种类
    print(f"Atom{i} ({element}): ({cart[0]:.3f},{cart[1]:.3f},{cart[2]:.3f})")
                                #格式化打印原子编号、元素名、三个坐标值（各保留三位小数）   

"""计算最邻近距离""" 
print("="*50)
print(" 计算最邻近距离 ")
print("="*50)
dist_matrix = structure.distance_matrix #获取距离矩阵
np.fill_diagonal(dist_matrix,np.inf)#将对角线元素设置为无穷大，避免计算自身距离
min_dist = np.min(dist_matrix)#计算最小距离
print(f"最邻近距离为：{min_dist:.3f} Å")


# ========== 第4步：3D可视化 ==========
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 画晶格框
lattice = structure.lattice.matrix
origin = np.array([0, 0, 0])
# 晶格的8个顶点
vertices = np.array([
    origin, lattice[0], lattice[0]+lattice[1], lattice[1],
    lattice[2], lattice[0]+lattice[2], lattice[0]+lattice[1]+lattice[2], lattice[1]+lattice[2]
])
# 画晶格边（简化版，只画从原点出发的三条边）
for i in range(3):
    ax.plot([origin[0], lattice[i][0]], 
            [origin[1], lattice[i][1]], 
            [origin[2], lattice[i][2]], 'k-', linewidth=2, alpha=0.5)

# 画原子
elements = [site.specie.symbol for site in structure]
unique_elements = list(set(elements))
colors = {'Si': 'blue', 'O': 'red', 'Fe': 'orange', 'Li': 'green'}

for i, (site, cart) in enumerate(zip(structure, cart_coords)):
    elem = site.specie.symbol
    color = colors.get(elem, 'gray')
    ax.scatter(*cart, c=color, s=200, label=elem, edgecolors='black')
    ax.text(cart[0]+0.2, cart[1]+0.2, cart[2]+0.2, f'{elem}{i}', fontsize=10)

# 画晶胞边界框
for edge in [
    [vertices[0], vertices[1]], [vertices[0], vertices[3]], [vertices[0], vertices[4]],
    [vertices[1], vertices[2]], [vertices[1], vertices[5]], [vertices[3], vertices[2]],
    [vertices[3], vertices[7]], [vertices[4], vertices[5]], [vertices[4], vertices[7]],
    [vertices[2], vertices[6]], [vertices[5], vertices[6]], [vertices[7], vertices[6]]
]:
    ax.plot([edge[0][0], edge[1][0]], 
            [edge[0][1], edge[1][1]], 
            [edge[0][2], edge[1][2]], 'k-', linewidth=1, alpha=0.3)

ax.set_xlabel('X (Å)')
ax.set_ylabel('Y (Å)')
ax.set_zlabel('Z (Å)')
ax.set_title('Zn(InS2)2 Structure')
# 去重图例
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys())

plt.tight_layout()
plt.savefig('crystal_3d.png', dpi=150)
plt.show()
print("\n✅ 3D结构图已保存为 crystal_3d.png")