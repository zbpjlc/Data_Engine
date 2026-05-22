import os
import sys
import time
import torch
import lance

# ==================== 配置区域 ====================
DS_PATH = "/media/ubuntu/Data4/专业课教材合集/manifests/ingest.lance"
COLUMN_NAME = "embedding"
INDEX_NAME = "idx_embedding_ivf"

# 调优参数（针对百万级数据推荐）
NUM_PARTITIONS = 256    # 聚类桶数量：数据量大可设为 512，数据量小设为 128
NUM_SUB_VECTORS = 16   # PQ子向量空间数：CLIP 768维通常推荐 16 或 32
# ==================================================

print("="*40)
print("🚀 Lance 向量索引自动构建程序（GPU版）")
print("="*40)

# 1. 检测数据集
if not os.path.exists(DS_PATH):
    print(f"❌ 错误: 找不到数据集路径 {DS_PATH}，请检查路径或外置盘是否挂载！", file=sys.stderr)
    sys.exit(1)

ds = lance.dataset(DS_PATH)
print(f"📦 数据集总行数: {ds.count_rows():,} 行")
print(f"📂 数据集包含的列: {ds.schema.names}")

if COLUMN_NAME not in ds.schema.names:
    print(f"❌ 错误: 数据集中不包含名为 [{COLUMN_NAME}] 的向量列！", file=sys.stderr)
    sys.exit(1)

# 2. 智能探测 GPU / 显卡环境
accelerator = None
if torch.cuda.is_available():
    print(f"🎸 检测到可用 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 当前显卡可用显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    accelerator = "cuda"
else:
    print("⚠️ 未检测到 PyTorch CUDA 环境，将自动回退至 CPU 计算（速度会变慢）")
    accelerator = None

# 3. 执行索引构建
print(f"\n⚡ 开始构建向量索引...")
print(f" - 目标列名:  {COLUMN_NAME}")
print(f" - 索引名称:  {INDEX_NAME}")
print(f" - 聚类桶数:  {NUM_PARTITIONS} (partitions)")
print(f" - 硬件加速:  {accelerator if accelerator else 'CPU'}")
print("⏱️ 正在计算，请稍候...")

start_time = time.time()

try:
    ds.create_index(
        column=COLUMN_NAME,
        index_type="IVF_PQ",
        name=INDEX_NAME,
        num_partitions=NUM_PARTITIONS,
        num_sub_vectors=NUM_SUB_VECTORS,
        replace=True,          # 如果不满意，允许二次运行直接覆盖
        accelerator=accelerator # 传入硬件加速器配置
    )
    
    elapsed_time = time.time() - start_time
    print("\n" + "="*40)
    print(f"🎉 向量索引成功构建完成！")
    print(f"⏱️ 耗时: {elapsed_time:.2f} 秒")
    print(f"📈 数据集当前版本已自动推进至: Version {ds.version}")
    print("="*40)
    print("💡 提示：现在你可以重新运行 test1.py 来盘点每个桶的大小分布了！")

except Exception as e:
    print(f"\n❌ 索引构建失败！错误原因: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
