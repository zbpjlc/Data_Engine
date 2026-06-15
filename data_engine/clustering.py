from pathlib import Path
from typing import Any
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
import json

from data_engine.config import get_config


class KMeansClusterer:
    """MiniBatch K-means 聚类器（比标准 KMeans 快 10-100 倍）"""
    
    def __init__(self, n_clusters: int | None = None, random_state: int | None = None):
        self.n_clusters = n_clusters or get_config("clustering", "default_n_clusters", default=5)
        self.random_state = random_state or get_config("clustering", "random_state", default=42)
        batch_size = get_config("clustering", "batch_size", default=4096)
        self.model = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            batch_size=batch_size,
            n_init=3,
            max_iter=100,
        )
        self.cluster_centers = None
        self.labels = None
        self.silhouette_score = None
    
    def fit_predict(self, embeddings: list[list[float]], progress_callback=None) -> list[int]:
        """拟合模型并预测聚类标签
        
        大数据集（>50K）：采样 fit + 全量 predict，大幅加速。
        """
        if not embeddings:
            return []
        
        # 过滤有效 embedding 的索引
        valid_indices = [i for i, emb in enumerate(embeddings) if emb and len(emb) > 0]
        if not valid_indices:
            return []
        
        if progress_callback:
            progress_callback(0, 3, "构建特征矩阵...")
        
        # 只转换有效 embedding（节省内存）
        X_valid = np.array([embeddings[i] for i in valid_indices], dtype=np.float32)
        
        sample_limit = get_config("clustering", "fit_sample_limit", default=50000)
        
        if progress_callback:
            progress_callback(1, 3, f"KMeans 拟合 {len(X_valid)} 个 embedding...")
        
        if len(X_valid) > sample_limit:
            # 大数据集：采样 fit 学习质心，全量 predict 分配标签
            rng = np.random.RandomState(self.random_state)
            fit_idx = rng.choice(len(X_valid), sample_limit, replace=False)
            X_fit = X_valid[fit_idx]
            self.model.fit(X_fit)
            self.labels = self.model.predict(X_valid)
        else:
            self.labels = self.model.fit_predict(X_valid)
        
        self.cluster_centers = self.model.cluster_centers_
        
        if progress_callback:
            progress_callback(2, 3, "计算轮廓系数...")
        
        # 计算轮廓系数（用采样数据加速）
        if len(set(self.labels)) > 1:
            if len(X_valid) > 10000:
                sample_idx = np.random.choice(len(X_valid), 10000, replace=False)
                self.silhouette_score = silhouette_score(X_valid[sample_idx], self.labels[sample_idx])
            else:
                self.silhouette_score = silhouette_score(X_valid, self.labels)
        else:
            self.silhouette_score = 0.0
        
        if progress_callback:
            progress_callback(3, 3, "聚类完成")
        
        # 返回完整标签列表（包含无效embedding的标签）
        full_labels = [-1] * len(embeddings)  # -1表示无效embedding
        for i, idx in enumerate(valid_indices):
            full_labels[idx] = int(self.labels[i])
        
        # 释放内存
        if valid_indices and len(valid_indices) > sample_limit:
            del X_fit
        del X_valid
        
        return full_labels
    
    def get_cluster_stats(self) -> dict[str, Any]:
        """获取聚类统计信息"""
        if self.labels is None:
            return {}
        
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        return {
            "n_clusters": self.n_clusters,
            "cluster_sizes": {str(int(label)): int(count) for label, count in zip(unique_labels, counts)},
            "silhouette_score": float(self.silhouette_score) if self.silhouette_score else 0.0,
            "cluster_centers": self.cluster_centers.tolist() if self.cluster_centers is not None else None
        }


def find_optimal_clusters(embeddings: list[list[float]], max_clusters: int | None = None, progress_callback=None) -> int:
    """使用轮廓系数找到最优聚类数量（MiniBatchKMeans 加速）
    
    大数据集（>50K）自动采样以加速最优 K 搜索。
    """
    if max_clusters is None:
        max_clusters = get_config("clustering", "max_clusters", default=10)
    random_state = get_config("clustering", "random_state", default=42)
    batch_size = get_config("clustering", "batch_size", default=4096)
    sample_limit = get_config("clustering", "find_k_sample_limit", default=50000)
    
    if not embeddings or len(embeddings) < 2:
        return 1
    
    X = np.array(embeddings, dtype=np.float32)
    valid_indices = [i for i, emb in enumerate(embeddings) if emb and len(emb) > 0]
    if len(valid_indices) < 2:
        return 1
    
    X_valid = X[valid_indices]
    
    # 大数据集采样加速最优 K 搜索
    if len(X_valid) > sample_limit:
        rng = np.random.RandomState(random_state)
        sample_idx = rng.choice(len(X_valid), sample_limit, replace=False)
        X_search = X_valid[sample_idx]
    else:
        X_search = X_valid
    
    best_score = -1
    best_k = 1
    k_range = list(range(2, min(max_clusters + 1, len(X_search))))
    
    for k_idx, k in enumerate(k_range):
        if progress_callback:
            progress_callback(k_idx, len(k_range), f"搜索最优 K ({k}/{len(k_range)})...")
        kmeans = MiniBatchKMeans(
            n_clusters=k,
            random_state=random_state,
            batch_size=min(batch_size, len(X_search)),
            n_init=3,
            max_iter=50,
        )
        labels = kmeans.fit_predict(X_search)
        
        if len(set(labels)) > 1:
            # 采样计算轮廓系数
            if len(X_search) > 10000:
                sil_idx = np.random.choice(len(X_search), 10000, replace=False)
                score = silhouette_score(X_search[sil_idx], labels[sil_idx])
            else:
                score = silhouette_score(X_search, labels)
            if score > best_score:
                best_score = score
                best_k = k
    
    if progress_callback:
        progress_callback(len(k_range), len(k_range), f"最优 K={best_k}")
    
    return best_k


def cluster_records(
    records: list[dict[str, Any]],
    n_clusters: int = 5,
    auto_optimize: bool = True
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """为记录进行聚类"""
    # 提取embedding
    embeddings = []
    for record in records:
        embedding = record.get("embedding")
        if embedding and len(embedding) > 0:
            embeddings.append(embedding)
        else:
            embeddings.append(None)
    
    # 自动优化聚类数量
    if auto_optimize:
        valid_embeddings = [emb for emb in embeddings if emb and len(emb) > 0]
        n_clusters = find_optimal_clusters(valid_embeddings, max_clusters=min(10, len(valid_embeddings)))
    
    # 执行聚类
    clusterer = KMeansClusterer(n_clusters=n_clusters)
    cluster_labels = clusterer.fit_predict(embeddings)
    
    # 更新记录的cluster_id
    updated_records = []
    for record, label in zip(records, cluster_labels):
        record_copy = record.copy()
        if label >= 0:  # 有效聚类
            record_copy["cluster_id"] = f"cluster_{label}"
        else:  # 无效embedding
            record_copy["cluster_id"] = None
        updated_records.append(record_copy)
    
    # 获取聚类统计
    cluster_stats = clusterer.get_cluster_stats()
    cluster_stats["total_records"] = len(records)
    cluster_stats["valid_embeddings"] = len([r for r in records if r.get("embedding")])
    
    return updated_records, cluster_stats
