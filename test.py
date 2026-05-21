import duckdb
import lance

# 1. 加载你的 Lance 数据集
ds = lance.dataset("/media/ubuntu/Data4/专业课教材合集/manifests/ingest.lance")

# 2. 利用 DuckDB 配合 Lance 的底层流，直接按物理分区统计数量
# 注意：Lance 索引文件在被 DuckDB 扫描时，可以通过内置函数或将其转为带有隐藏分区属性的视图
# 如果你已经按我上一轮说的方案“显式写回了 cluster_id”，这就是终极精准的统计：
query = """
    SELECT 
        cluster_id AS partition_id, 
        COUNT(*) AS vector_count,
        ROUND(COUNT(*) * 100.0 / 2494825, 2) AS percentage
    FROM ds 
    GROUP BY cluster_id 
    ORDER BY vector_count DESC
"""

# 执行并打印结果
df = duckdb.query(query).df()
print(df)
