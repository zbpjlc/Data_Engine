"""
Embedding 提取模块 - 通过外部 HTTP 服务调用

使用独立的 embedding server 避免 web_app 线程中的 cuBLAS LT 崩溃问题。
Embedding server 在独立进程中运行，拥有自己的 CUDA 上下文。
"""
import io
import sys
import base64
import requests
from pathlib import Path

from data_engine.config import get_config


def _coerce_image_bytes(img) -> bytes | None:
    """将 Lance 返回的各种图像数据格式转换为 bytes"""
    if img is None:
        return None
    if isinstance(img, bytes):
        return img
    if isinstance(img, bytearray):
        return bytes(img)
    if isinstance(img, memoryview):
        return bytes(img)
    # numpy array 或其他类型
    try:
        return bytes(img)
    except Exception:
        return None


class CLIPEmbeddingExtractor:
    """
    ✅ 通过 HTTP 调用外部 embedding server
    ✅ 无本地 CUDA 依赖，线程安全
    ✅ 支持单张和批量提取
    """

    def __init__(self, server_url: str | None = None):
        self.server_url = server_url or get_config(
            "embedding", "server_url",
            default="http://127.0.0.1:8099"
        )
        # 移除末尾斜杠
        self.server_url = self.server_url.rstrip("/")
        self.device = "http"  # 兼容旧代码的设备检查

        # 连接超时和读取超时
        self.timeout = (5.0, 60.0)  # (connect, read)

        # 验证服务可用性
        self._check_server()

    def _check_server(self):
        """检查 embedding server 是否可用"""
        try:
            resp = requests.get(
                f"{self.server_url}/health",
                timeout=self.timeout
            )
            if resp.ok:
                info = resp.json()
                print(f"[Embedding] 服务已连接: {self.server_url} ({info.get('model', 'unknown')})", file=sys.stderr)
            else:
                print(f"[Embedding] 服务响应异常: {resp.status_code}", file=sys.stderr)
        except requests.RequestException as e:
            print(f"[Embedding] 无法连接服务 {self.server_url}: {e}", file=sys.stderr)

    # ------------------------------------------------------------------
    # ✅ 单张（文件路径）
    # ------------------------------------------------------------------
    def extract_embedding(self, image_path: Path) -> list[float]:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        return self.extract_embedding_from_bytes(image_bytes)

    # ------------------------------------------------------------------
    # ✅ 单张（bytes）
    # ------------------------------------------------------------------
    def extract_embedding_from_bytes(self, image_bytes: bytes) -> list[float]:
        try:
            # 发送 base64 编码的图片
            payload = {
                "image": base64.b64encode(image_bytes).decode("ascii")
            }
            resp = requests.post(
                f"{self.server_url}/embed",
                json=payload,
                timeout=self.timeout
            )
            resp.raise_for_status()
            data = resp.json()
            return data["embedding"]
        except Exception as e:
            print(f"[Embedding] 单张提取失败: {e}", file=sys.stderr)
            raise

    # ------------------------------------------------------------------
    # ✅ 批量（通过 HTTP 批量请求）
    # ------------------------------------------------------------------
    def extract_embeddings_from_bytes_batch(
        self,
        image_bytes_list: list[bytes],
        batch_size: int = 32,
    ) -> list[list[float] | None]:
        """批量提取 embedding，内部自动分批发送到 server"""
        results = [None] * len(image_bytes_list)

        for start in range(0, len(image_bytes_list), batch_size):
            end = min(start + batch_size, len(image_bytes_list))
            batch_bytes = []
            batch_idxs = []

            for i in range(start, end):
                img_bytes = image_bytes_list[i]
                if img_bytes:
                    batch_bytes.append(img_bytes)
                    batch_idxs.append(i)

            if not batch_bytes:
                continue

            try:
                # base64 编码批量图片
                payload = {
                    "images": [
                        base64.b64encode(b).decode("ascii")
                        for b in batch_bytes
                    ]
                }
                resp = requests.post(
                    f"{self.server_url}/embed/batch",
                    json=payload,
                    timeout=(5.0, 300.0)  # 批量允许更长超时
                )
                resp.raise_for_status()
                data = resp.json()
                embeddings = data["embeddings"]

                for j, idx in enumerate(batch_idxs):
                    if j < len(embeddings) and embeddings[j] is not None:
                        results[idx] = embeddings[j]

            except Exception as e:
                print(f"[Embedding] 批量提取失败 ({len(batch_bytes)} 张): {e}", file=sys.stderr)
                # 批量失败时尝试单张重试
                for j, (idx, img_bytes) in enumerate(zip(batch_idxs, batch_bytes)):
                    if results[idx] is None:
                        try:
                            results[idx] = self.extract_embedding_from_bytes(img_bytes)
                        except Exception:
                            pass

        return results


def extract_embeddings_for_records(
    records: list[dict],
    batch_dir,
    extractor: CLIPEmbeddingExtractor | None = None,
    task_id: str | None = None,
    offset: int = 0,
) -> list[dict]:
    """
    为记录列表提取 embedding，返回带有 embedding 字段的记录。
    
    Args:
        records: 记录列表，每条包含 sample_id 和 image_data
        batch_dir: 批次目录（用于读取图像，但当前 HTTP 模式不需要）
        extractor: CLIPEmbeddingExtractor 实例，为 None 时自动创建
        task_id: 任务 ID（用于进度跟踪）
        offset: 当前偏移量（用于进度跟踪）
    """
    if extractor is None:
        extractor = CLIPEmbeddingExtractor()

    # 收集图像 bytes
    image_bytes_list = []
    valid_indices = []
    for i, record in enumerate(records):
        img_data = record.get("image_data")
        if img_data:
            img_bytes = _coerce_image_bytes(img_data)
            if img_bytes:
                image_bytes_list.append(img_bytes)
                valid_indices.append(i)
            else:
                image_bytes_list.append(None)
                valid_indices.append(i)
        else:
            image_bytes_list.append(None)
            valid_indices.append(i)

    # 批量提取
    embeddings = extractor.extract_embeddings_from_bytes_batch(image_bytes_list, batch_size=32)

    # 写回记录
    for i, emb in zip(valid_indices, embeddings):
        records[i]["embedding"] = emb

    return records
