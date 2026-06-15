"""Embedding HTTP 服务 - 独立进程，加载 SigLIP2 模型，提供 embedding 推理 API。

用法:
    python -m data_engine.ocr.embedding_server
    # 或
    python data_engine/ocr/embedding_server.py

API:
    POST /embed     body: {"images": [base64_str...], "batch_size": 32}
    GET  /health    → {"status": "ready", "gpu": "cuda:0"}
"""
from __future__ import annotations

import os
import sys
import base64
import io
import time
from pathlib import Path

# 限制线程数，避免多层线程嵌套崩溃
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1"

# 设置默认 GPU（必须在 torch 导入之前）
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from data_engine.config import get_config

gpu_id = get_config("embedding", "server", "gpu_id", default=1)
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Embedding Server", version="1.0.0")

# 全局 extractor 实例
_extractor = None
_model_info = {}


class EmbedRequest(BaseModel):
    images: list[str]  # base64 编码的图片字节列表
    batch_size: int = 32


class EmbedResponse(BaseModel):
    embeddings: list[list[float] | None]
    count: int
    elapsed_ms: float


@app.on_event("startup")
async def startup():
    global _extractor, _model_info
    from data_engine.embedding import CLIPEmbeddingExtractor

    t0 = time.time()
    _extractor = CLIPEmbeddingExtractor()
    _model_info = {
        "device": str(_extractor.device),
        "load_time": round(time.time() - t0, 2),
    }
    print(f"[EmbeddingServer] 模型加载完成: {_model_info}", file=sys.stderr)

    # 预热：运行一次推理
    from PIL import Image
    warmup_imgs = []
    for _ in range(4):
        img = Image.new("RGB", (224, 224), color="gray")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        warmup_imgs.append(buf.getvalue())
    _ = _extractor.extract_embeddings_from_bytes_batch(warmup_imgs, batch_size=4)
    del warmup_imgs
    print("[EmbeddingServer] 预热完成", file=sys.stderr)


@app.get("/health")
async def health():
    return {
        "status": "ready" if _extractor else "loading",
        **_model_info,
    }


@app.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest):
    if _extractor is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    try:
        t0 = time.time()
        image_bytes_list = [base64.b64decode(img) for img in req.images]
        results = _extractor.extract_embeddings_from_bytes_batch(
            image_bytes_list, batch_size=req.batch_size
        )
        elapsed_ms = round((time.time() - t0) * 1000)
        return EmbedResponse(
            embeddings=results,
            count=len(results),
            elapsed_ms=elapsed_ms,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main():
    import uvicorn
    host = get_config("embedding", "server", "host", default="127.0.0.1")
    port = get_config("embedding", "server", "port", default=8090)
    print(f"[EmbeddingServer] 启动于 {host}:{port}", file=sys.stderr)
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
