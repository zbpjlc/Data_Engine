from pathlib import Path
from typing import Any
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import json


class KMeansClusterer:
    """K-means聚类器"""
    
    def __init__(self, n_clusters: int = 5, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.cluster_centers = None
        self.labels = None
        self.silhouette_score = None
    
    def fit_predict(self, embeddings: list[list[float]]) -> list[int]:
        """拟合模型并预测聚类标签"""
        if not embeddings:
            return []
        
        # 转换为numpy数组
        X = np.array(embeddings)
        
        # 过滤掉无效的embedding（None或空）
        valid_indices = [i for i, emb in enumerate(embeddings) if emb and len(emb) > 0]
        if not valid_indices:
            return []
        
        X_valid = X[valid_indices]
        
        # K-means聚类
        self.labels = self.model.fit_predict(X_valid)
        self.cluster_centers = self.model.cluster_centers_
        
        # 计算轮廓系数
        if len(set(self.labels)) > 1:  # 需要至少2个聚类
            self.silhouette_score = silhouette_score(X_valid, self.labels)
        else:
            self.silhouette_score = 0.0
        
        # 返回完整标签列表（包含无效embedding的标签）
        full_labels = [-1] * len(embeddings)  # -1表示无效embedding
        for i, idx in enumerate(valid_indices):
            full_labels[idx] = int(self.labels[i])
        
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


def find_optimal_clusters(embeddings: list[list[float]], max_clusters: int = 10) -> int:
    """使用轮廓系数找到最优聚类数量"""
    if not embeddings or len(embeddings) < 2:
        return 1
    
    X = np.array(embeddings)
    valid_indices = [i for i, emb in enumerate(embeddings) if emb and len(emb) > 0]
    if len(valid_indices) < 2:
        return 1
    
    X_valid = X[valid_indices]
    
    best_score = -1
    best_k = 1
    
    for k in range(2, min(max_clusters + 1, len(X_valid))):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_valid)
        
        if len(set(labels)) > 1:
            score = silhouette_score(X_valid, labels)
            if score > best_score:
                best_score = score
                best_k = k
    
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
