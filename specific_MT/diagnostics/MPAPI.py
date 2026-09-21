from mp_api.client import MPRester

try:
    # MPRester 会自动从环境变量中读取 API Key
    with MPRester() as mpr:
        # 尝试查询一个简单的材料（例如硅 mp-149）
        docs = mpr.materials.summary.search(material_ids=["mp-149"], fields=["material_id", "formula_pretty"])
        if docs:
            print("API Key 配置成功！")
            print(f"查询到材料: {docs[0].formula_pretty} ({docs[0].material_id})")
        else:
            print("API Key 有效，但未查询到数据。")
except Exception as e:
    print(f"配置失败，请检查 API Key: {e}")