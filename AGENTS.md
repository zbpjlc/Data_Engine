# AGENTS.md

This file provides guidance to Qoder (qoder.com) when working with code in this repository.

## Project Overview

Data Engine is a document parsing training data pipeline for OCR. It processes PDFs/images through a multi-stage pipeline: ingest → embedding → clustering → multi-model OCR → CMCV (cross-model consistency verification) → sampling → export.

**No database.** The system uses a federated manifest architecture: `sources.yaml + Lance files` is the single source of truth. Data sources may live on different disks/mount points.

## Commands

```bash
# Run the web server (Flask, production entry point)
python -m data_engine

# Run CLI commands
python -m data_engine register --path /data/finance --source-id disk_01 --category finance
python -m data_engine scan
python -m data_engine ingest --source-id disk_01 --batch batch_001
python -m data_engine embed --source-id disk_01 --batch batch_001
python -m data_engine cluster --source-id disk_01 --batch batch_001 --auto-optimize
python -m data_engine element-sample --source-id disk_01 --batch batch_001
python -m data_engine cmcv --source-id disk_01 --batch batch_001

# Run tests
python -m pytest tests/

# Run a single test
python -m pytest tests/test_registry_and_status.py::RegistryStatusTests::test_register_and_scan
```

## Architecture

### Three-Layer Architecture

1. **Source Registry** (`registry.py`, `sources.yaml`): Maps `source_id → root_path`. Physical path can change (disk remount) without breaking manifests.
2. **Batch Directory Organization**: `<root_path>/<batch_id>/manifests/` + `<root_path>/<batch_id>/artifacts/`
3. **Manifest Federation**: Each batch stores its own stage results as Lance files. Global operations aggregate manifests at runtime.

### Key Data Files

| File | Purpose |
|------|---------|
| `sources.yaml` | Global data source registry |
| `config.yaml` | All tunable parameters (embedding model, clustering, OCR, CMCV) |
| `progress_state.json` | Runtime task progress (auto-managed, gitignored) |

### Lance Files (per batch, under `manifests/`)

| File | Content |
|------|---------|
| `ingest.lance` | Page-level records with `image_data`, `embedding`, `difficulty` |
| `text.lance` / `formula.lance` / `table.lance` | Block-level records split by type from layout detection |
| `element.lance` | Multi-model OCR results per block |

### Pipeline Stages & Key Modules

```
ingest.lance (page images + metadata)
    ↓ embedding.py (ViT: SigLIP2)
    ↓ clustering.py (MiniBatchKMeans)
    ↓ page-sample (per-cluster sampling → OCR → CMCV → difficulty sampling)
    ↓ web_app.py layout detection → split to text/formula/table.lance
    ↓ ocr/layout_features.py (element-level ViT clustering per type)
    ↓ element-sample (per-cluster sampling, output → element_samples.json)
    ↓ Element OCR: multi-model OCR on ALL blocks in text/formula/table.lance
    ↓ Element CMCV: cross-model consistency on all OCR results
    ↓ (element-type level: text/formula/table independently)
```

> **Note:** Element-Sample 将抽样结果写入 `artifacts/element_samples.json`，但 Element OCR **不读取该文件**，而是直接扫描 `text.lance`/`formula.lance`/`table.lance` 中的全部 block（支持断点续跑）。Element-Sample 当前主要用于统计/预览用途，不影响 OCR 处理范围。

### Two Clustering Levels

- **Page-Level** (`clustering.py` → `cluster_records`): Clusters page images using ViT embeddings + MiniBatchKMeans. Operates on `ingest.lance`. Sub-pipeline: **cluster → per-cluster sampling → multi-model OCR → CMCV → difficulty-aware sampling**，只对抽样的页面做 OCR。
- **Element-Level** (`ocr/layout_features.py` → `cluster_all_types`): Clusters blocks (text/formula/table) separately from `text.lance`/`formula.lance`/`table.lance`。聚类后：**element-sample 按 cluster 抽样（仅写 element_samples.json）**，然后 **Element OCR 扫描所有 block**（不按抽样过滤），最后 **Element CMCV 对所有 OCR 结果做一致性验证**。All operations are scoped to the **element-type** level (text, formula, table independently).

### OCR Subsystem (`data_engine/ocr/`)

- `layout_provider.py`: PP-DocLayout for block detection
- `paddle_ocr.py`, `glm_ocr.py`, `self_ocr.py`: Three OCR engines (HTTP API services)
- `cmcv.py`: Cross-Model Consistency Verification engine
- `normalizer.py`: Converts engine-specific outputs to unified format
- `__init__.py`: Lance schemas for text/formula/table blocks + block type mappings

### Web App (`web_app.py`)

Flask application (~3600 lines). Serves the dashboard UI and handles all async batch operations. On startup:
- Pre-initializes CUDA/cuBLAS to avoid malloc deadlocks
- Pre-loads SigLIP2 model as `_shared_extractor`
- Starts `EmbeddingWorker` on GPU 1 (separate from main GPU 0)

### Progress Tracking (`progress_tracker.py`)

Global singleton `progress_tracker` manages async task state. Tasks have `start_task`/`update_progress`/`complete_task`/`fail_task`/`stop_task` lifecycle. Frontend polls for status.

## Critical Constraints

### Lance Write Lock
All Lance writes (`merge_insert`, `write_dataset`) must hold `_lance_write_lock` from `manifests.py`. Concurrent writes cause glibc malloc deadlocks. Reads are lock-free.

### PyArrow Types
- Strings: always `pa.large_string()` (not `pa.string()`)
- Lists that may exceed 2^31 elements: `pa.large_list()`
- Embeddings: `pa.list_(pa.float32(), 768)` — dimension from config, immutable once set

### Threading
`web_app.py` sets `OMP_NUM_THREADS=1` etc. at import time to prevent multi-layer thread nesting crashes. Do not remove these.

### OCR Services
OCR engines run as external HTTP services on fixed ports (configured in `config.yaml`). They must be started before running Element OCR.

## Configuration

All tunable parameters live in `config.yaml`. Key sections:
- `embedding`: Model name, GPU ID, batch sizes
- `clustering`: MiniBatchKMeans params, sampling limits
- `layout`: Flush sizes for block lance writes
- `ocr.engines`: API URLs and timeouts for each OCR service
- `ocr.sampling`: Difficulty-aware sampling ratios (percentages, not absolute counts)

Use `get_config(section, key, default)` from `data_engine.config` to read values.
