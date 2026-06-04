import pandas as pd
import lance
import pyarrow as pa

ds_path = "/media/ubuntu/Data4/专业课教材合集/manifests/ingest.lance"
ds = lance.dataset(ds_path)

print("🚀 开始执行【动态百分比·全域多样性采样】...")

# 1. 捞出 256 个桶的元数据（包含质心和每个桶的真实大小）
stats = ds.index_statistics("embedding")
indices_data = stats["indices"][0]
centroids = indices_data["centroids"]
partitions = indices_data["partitions"]  # 拿到每个桶的真实 size 列表

sampled_records = []

# 2. 动态游标遍历
for bucket_idx, (centroid, part_info) in enumerate(zip(centroids, partitions)):
    bucket_size = part_info["size"]
    
    # 如果桶太小（比如冷门小桶只有不到 100 条数据），全量保留，它们本身就是极端多样性的体现
    if bucket_size <= 100:
        scanner = ds.scanner(
            nearest={"column": "embedding", "q": centroid, "k": bucket_size, "metric_type": "l2"}
        )
        df_pool = scanner.to_table().to_pandas()
        df_pool["sample_type"] = "Rare_Thick"
        df_pool["bucket_id"] = bucket_idx
        sampled_records.append(df_pool)
        continue
        
    # 🔥 核心改动：动态设置 k 等于这个桶的完整大小！
    # 这样能确保我们把这个聚类里的数据从“最纯正”到“最边缘”排成一条直线
    scanner = ds.scanner(
        nearest={
            "column": "embedding",
            "q": centroid,
            "k": bucket_size, 
            "metric_type": "l2"
        }
    )
    df_pool = scanner.to_table().to_pandas()
    
    # 🎯 动态采样位置
    # 1. 核心代表（绝对前排）
    core_sample = df_pool.iloc[[0]].copy()
    core_sample["sample_type"] = "Core"
    
    # 2. 中坚力量（动态中位数，代表该学科的中等普遍知识）
    mid_idx = bucket_size // 2
    mid_sample = df_pool.iloc[[mid_idx]].copy()
    mid_sample["sample_type"] = "Median"
    
    # 3. 语义边界（绝对最后一名，真正的跨界/边缘长尾）
    boundary_sample = df_pool.iloc[[-1]].copy()
    boundary_sample["sample_type"] = "True_Boundary"
    
    # 标记桶编号并追加
    for sample in [core_sample, mid_sample, boundary_sample]:
        sample["bucket_id"] = bucket_idx
        sampled_records.append(sample)

# 3. 组装最终的多样性黄金集
diverse_df = pd.concat(sampled_records, ignore_index=True)

print("\n" + "="*40)
print(f"🎉 动态多样性采样成功！共获取样本：{len(diverse_df)} 条")
print(f" 核心代表 (Core):      {len(diverse_df[diverse_df['sample_type']=='Core'])} 条")
print(f" 中坚分布 (Median):    {len(diverse_df[diverse_df['sample_type']=='Median'])} 条")
print(f" 绝对边界 (Boundary):  {len(diverse_df[diverse_df['sample_type']=='True_Boundary'])} 条")
print(f" 稀有长尾 (Rare_Thick):{len(diverse_df[diverse_df['sample_type']=='Rare_Thick'])} 条")
print("="*40)

diverse_df.to_csv("dynamic_diverse_dataset.csv", index=False)