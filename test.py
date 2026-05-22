import json
import numpy as np
import lance

ds_path = "/media/ubuntu/Data4/专业课教材合集/manifests/ingest.lance"
ds = lance.dataset(ds_path)
print("🎉 成功加载 Lance 数据集！")

indices = ds.list_indices()

counts = []

for idx in indices:
    idx_name = idx.get("name") if isinstance(idx, dict) else getattr(idx, "name", "")
    if not idx_name:
        idx_name = idx.get("index_name") or (list(idx.values())[0] if isinstance(idx, dict) and idx.values() else "")
        
    try:
        stats = ds.index_statistics(idx_name)  # type: ignore
        if isinstance(stats, str):
            stats = json.loads(stats)
            
        # 🎯 精准对齐你的真实结构：stats -> 'indices' -> 第0个元素 -> 'partitions'
        if isinstance(stats, dict) and "indices" in stats:
            for sub_idx in stats["indices"]:
                if "partitions" in sub_idx:
                    for part in sub_idx["partitions"]:
                        if "size" in part:
                            counts.append(part["size"])
                    break  # 提取成功，跳出当前索引的子循环
        if counts:
            break  # 成功拿到分桶数据，结束外层循环
            
    except Exception as e:
        print(f"读取索引 [{idx_name}] 失败: {e}")
        continue

# 3. 展现统计图表
if len(counts) > 0:
    counts_arr = np.array(counts)
    
    print("\n" + "="*40)
    print(f"📊 embedding 列 (IVF 索引) 桶大小最终统计")
    print("="*40)
    print(f"聚类总簇数 (总分区): {len(counts_arr)}")
    print(f"数据总条数:        {counts_arr.sum():,} 条")
    print(f"最大簇 (最大桶):   {counts_arr.max()} 条数据")
    print(f"最小簇 (最小桶):   {counts_arr.min()} 条数据")
    print(f"平均每簇包含数量:  {int(counts_arr.mean())} 条数据")
    print(f"空簇数量 (0条数据):  {(counts_arr == 0).sum()}")
    print(f"小于50条的小簇数:  {(counts_arr < 50).sum()}")
    print(f"大于3000条的大簇数: {(counts_arr > 3000).sum()}")
    print("="*40)
    print(f"💡 前 10 个桶的大小展示: {list(counts_arr[:10])}")
else:
    print("❌ 依然未能提取出分桶大小，请检查结构层级。")
