import lance
import numpy as np

# 【正确路径】你真实的 Lance 数据集路径
ds_path = "/media/ubuntu/Data4/专业课教材合集/manifests"

# 加载数据集
ds = lance.dataset(ds_path)

# 列出所有索引
indices = ds.list_indices()
print(f"找到索引数量：{len(indices)}")

# 找到 IVF 索引
ivf = None
for idx in indices:
    print(f"索引列: {idx.column}, 类型: {idx.index_type}")
    if hasattr(idx, "ivf"):
        ivf = idx.ivf
        break

# 输出每个簇的数量（你最想要的）
if ivf is not None:
    counts = ivf.inverted_list_sizes
    print("\n==== IVF 分区大小统计 ====")
    print(f"总簇数: {len(counts)}")
    print(f"最大簇: {counts.max()}")
    print(f"最小簇: {counts.min()}")
    print(f"空簇: {(counts == 0).sum()}")
    print(f"小于50条的小簇: {(counts < 50).sum()}")
else:
    print("未找到 IVF 索引")
