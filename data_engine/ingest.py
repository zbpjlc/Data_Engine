from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

from data_engine.hashing import sample_id_for_page, sha256_file
from data_engine.manifests import write_json, write_jsonl, read_jsonl
from data_engine.models import InputType, SourceMetadata, SourceRegistryModel, StageStatus, UnifiedSampleRecord
from data_engine.registry import SourceRegistry


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class IngestResult:
    records: list[UnifiedSampleRecord]
    stats: dict[str, Any]


def run_ingest(registry: SourceRegistry, source_id: str, batch_id: str) -> IngestResult:
    import sys
    import time
    from tqdm import tqdm
    from data_engine.progress_tracker import progress_tracker
    from data_engine.models import BatchMetadata
    
    source = registry.get(source_id)
    batch_dir = Path(source.root_path) / batch_id
    
    # 去中心化元数据管理：从批次目录读取元数据
    try:
        batch_meta = BatchMetadata.from_batch_dir(batch_dir)
        category = batch_meta.category
        print(f"从.engine_meta.yaml读取category: {category}", file=sys.stderr)
    except FileNotFoundError:
        # 回退到旧逻辑：从source获取category
        print(f"未找到.engine_meta.yaml，使用source配置的category", file=sys.stderr)
        category = source.category
    
    page_images_dir = batch_dir / "page_images"
    manifests_dir = batch_dir / "manifests"
    artifacts_dir = batch_dir / "artifacts"

    page_images_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 检查已存在的manifest，支持断点续跑
    manifest_path = manifests_dir / "ingest.jsonl"
    file_list_path = manifests_dir / "input_files.json"
    existing_records = []
    processed_files = set()
    stored_input_files = []
    
    # 尝试读取存储的输入文件列表
    if file_list_path.exists():
        try:
            with open(file_list_path, 'r') as f:
                stored_input_files = json.load(f)
            print(f"读取存储的输入文件列表: {len(stored_input_files)} 个文件", file=sys.stderr)
        except Exception as e:
            print(f"读取文件列表失败: {e}", file=sys.stderr)
    
    if manifest_path.exists():
        try:
            existing_records = read_jsonl(manifest_path)
            # 提取已处理的文件路径（使用relative_path字段）
            for record in existing_records:
                if "relative_path" in record:
                    # 构建完整的文件路径
                    full_path = str(batch_dir / record["relative_path"])
                    processed_files.add(full_path)
            print(f"发现已有 {len(existing_records)} 条记录，已处理 {len(processed_files)} 个文件", file=sys.stderr)
        except Exception as e:
            print(f"读取现有manifest失败: {e}", file=sys.stderr)
    
    # 如果manifest为空，基于page_images目录推断已处理的文件
    if not processed_files and page_images_dir.exists():
        # 检查page_images目录中的文件，推断已处理的源文件
        for page_image in page_images_dir.rglob("*"):
            if page_image.is_file() and page_image.suffix.lower() in IMAGE_EXTENSIONS:
                # 从图像文件名推断源文件
                # 格式：images__pdf_xxx_page_001.png -> pdf_xxx
                try:
                    filename = page_image.stem
                    if "__" in filename:
                        # 提取源文件标识
                        parts = filename.split("__")
                        if len(parts) >= 2:
                            source_prefix = parts[1]  # pdf_xxx
                            # 查找对应的源文件
                            for source_file in batch_dir.rglob(f"{source_prefix}*"):
                                if source_file.is_file() and source_file.suffix.lower() in [".pdf"] + list(IMAGE_EXTENSIONS):
                                    processed_files.add(str(source_file))
                                    break
                except Exception:
                    continue
        
        if processed_files:
            print(f"基于page_images推断已处理 {len(processed_files)} 个文件", file=sys.stderr)

    # 扫描所有输入文件（只统计需要处理的文件类型，只扫描原始输入目录）
    if stored_input_files:
        # 使用存储的文件列表，确保分母一致
        input_files = [Path(f) for f in stored_input_files]
        print(f"使用存储的输入文件列表: {len(input_files)} 个文件", file=sys.stderr)
    else:
        # 扫描并存储输入文件列表（只扫描raw_input目录）
        raw_input_dir = batch_dir / "raw_input"
        if raw_input_dir.exists():
            input_files = sorted(p for p in raw_input_dir.rglob("*") 
                               if p.is_file() 
                               and p.suffix.lower() in [".pdf"] + list(IMAGE_EXTENSIONS))
        else:
            # 如果没有raw_input目录，扫描batch_dir但排除输出目录
            input_files = sorted(p for p in batch_dir.rglob("*") 
                               if p.is_file() 
                               and p.suffix.lower() in [".pdf"] + list(IMAGE_EXTENSIONS)
                               and "page_images" not in str(p)
                               and "manifests" not in str(p)
                               and "artifacts" not in str(p))
        
        # 存储输入文件列表
        try:
            with open(file_list_path, 'w') as f:
                json.dump([str(f) for f in input_files], f)
            print(f"存储输入文件列表: {len(input_files)} 个文件", file=sys.stderr)
        except Exception as e:
            print(f"存储文件列表失败: {e}", file=sys.stderr)
    
    # 过滤出需要处理的文件（跳过已处理的）
    files_to_process = [f for f in input_files if str(f) not in processed_files]
    
    skipped_count = len(input_files) - len(files_to_process)
    if skipped_count > 0:
        print(f"跳过 {skipped_count} 个已处理文件，将处理 {len(files_to_process)} 个新文件", file=sys.stderr)
    
    # 使用固定的task_id，支持断点续跑时复用同一个任务
    task_id = f"ingest_{source_id}_{batch_id}"
    
    # 检查是否已有任务
    existing_task = progress_tracker.get_task(task_id)
    if existing_task and existing_task.status.value in ["running", "pending"]:
        # 复用现有任务，但保持原有的total
        print(f"复用现有任务: {task_id}, 原total: {existing_task.total}", file=sys.stderr)
        # 更新current为已处理数量+1（从下一个文件开始）
        progress_tracker.update_progress(
            task_id=task_id,
            current=skipped_count + 1,  # 从下一个文件开始
            message=f"继续处理 {len(files_to_process)} 个文件 (已处理 {skipped_count} 个)"
        )
    else:
        # 创建新任务，使用固定的input_files数量作为分母
        progress_tracker.start_task(
            task_id=task_id,
            task_type="ingest",
            source_id=source_id,
            batch_id=batch_id,
            total=len(input_files),  # 使用总输入文件数量作为分母
            message=f"开始处理 {len(files_to_process)} 个文件 (跳过 {skipped_count} 个已处理)"
        )
        # 立即更新current为已跳过的数量
        progress_tracker.update_progress(
            task_id=task_id,
            current=skipped_count
        )
    
    print(f"发现 {len(files_to_process)} 个待处理文件，开始处理...", file=sys.stderr)
    
    try:
        new_records: list[UnifiedSampleRecord] = []
        # 实时更新manifest，支持断点续跑
        batch_update_interval = 100  # 每处理100个文件更新一次manifest
        with tqdm(files_to_process, desc="处理文件", unit="file", total=len(input_files), initial=skipped_count) as pbar:
            for i, input_path in enumerate(pbar):
                ext = input_path.suffix.lower()
                pbar.set_postfix({"文件": input_path.name})
                
                # 更新进度
                progress_tracker.update_progress(
                    task_id=task_id,
                    current=skipped_count + i,  # 使用已跳过文件数 + 当前处理索引
                    message=f"正在处理: {input_path.name}"
                )
                
                try:
                    if ext == ".pdf":
                        pdf_records = _ingest_pdf(input_path, batch_dir, page_images_dir, source_id, category, batch_id)
                        new_records.extend(pdf_records)
                    elif ext in IMAGE_EXTENSIONS:
                        record = _ingest_image(input_path, batch_dir, page_images_dir, source_id, category, batch_id)
                        new_records.append(record)
                    
                    # 定期更新manifest，支持断点续跑
                    if len(new_records) > 0 and len(new_records) % batch_update_interval == 0:
                        current_all_records = existing_records + new_records
                        write_jsonl(manifest_path, current_all_records)
                        print(f"已处理 {len(new_records)} 个样本，更新manifest", file=sys.stderr)
                        
                except Exception as e:
                    print(f"处理文件 {input_path} 时出错: {e}", file=sys.stderr)
                    continue
        
        # 完成任务
        progress_tracker.complete_task(
            task_id=task_id,
            message=f"成功处理 {len(new_records)} 个样本"
        )
        
    except Exception as e:
        progress_tracker.fail_task(
            task_id=task_id,
            error_message=str(e)
        )
        raise

    # 合并新旧记录
    all_records = existing_records + new_records
    
    # 写入manifest
    write_jsonl(manifest_path, all_records)

    stats = _build_ingest_stats(all_records)
    write_json(artifacts_dir / "stats.json", stats)
    return IngestResult(records=all_records, stats=stats)


def _ingest_pdf(
    pdf_path: Path,
    raw_input_dir: Path,
    page_images_dir: Path,
    source_id: str,
    category: str,
    batch_id: str,
) -> list[UnifiedSampleRecord]:
    pdf_hash = sha256_file(pdf_path)
    page_count = _pdf_page_count(pdf_path)
    text_excerpt = _pdf_text_excerpt(pdf_path)
    relative_input = pdf_path.relative_to(raw_input_dir).as_posix()
    stem_dir = page_images_dir / relative_input.replace("/", "__").removesuffix(pdf_path.suffix)
    stem_dir.mkdir(parents=True, exist_ok=True)
    prefix = stem_dir / "page"

    command = [
        "pdftoppm",
        "-png",
        str(pdf_path),
        str(prefix),
    ]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    rendered_pages = sorted(stem_dir.glob("page-*.png"))
    records: list[UnifiedSampleRecord] = []
    for index, page_path in enumerate(rendered_pages, start=1):
        page_id = _hash_text(f"{pdf_hash}:{index}")
        normalized_relative = page_path.relative_to(page_images_dir.parent).as_posix()
        metadata = SourceMetadata(pdf_page_count=page_count, pdf_text_excerpt=text_excerpt)
        records.append(
            UnifiedSampleRecord(
                sample_id=sample_id_for_page(source_id, category, batch_id, page_id),
                source_id=source_id,
                category=category,
                batch_id=batch_id,
                input_type=InputType.PDF,
                original_ext=".pdf",
                page_id=page_id,
                relative_path=relative_input,
                page_image=normalized_relative,
                stage_status=StageStatus.INGESTED,
                page_image_sha256=sha256_file(page_path),
                source_metadata=metadata,
            )
        )
    return records


def _ingest_image(
    image_path: Path,
    raw_input_dir: Path,
    page_images_dir: Path,
    source_id: str,
    category: str,
    batch_id: str,
) -> UnifiedSampleRecord:
    image_hash = sha256_file(image_path)
    page_id = image_hash
    relative_input = image_path.relative_to(raw_input_dir).as_posix()
    output_name = relative_input.replace("/", "__").rsplit(".", 1)[0] + ".png"
    output_path = page_images_dir / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    info_before = _ffprobe_image(image_path)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(image_path),
        "-frames:v",
        "1",
        str(output_path),
    ]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    info_after = _ffprobe_image(output_path)

    metadata = SourceMetadata(
        original_width=info_before.get("width"),
        original_height=info_before.get("height"),
        normalized_width=info_after.get("width"),
        normalized_height=info_after.get("height"),
        scale_x=_scale(info_before.get("width"), info_after.get("width")),
        scale_y=_scale(info_before.get("height"), info_after.get("height")),
        original_dpi=info_before.get("dpi"),
        normalized_dpi=info_after.get("dpi"),
        color_mode=info_after.get("pix_fmt"),
        extra={"normalization_applied": True},
    )

    return UnifiedSampleRecord(
        sample_id=sample_id_for_page(source_id, category, batch_id, page_id),
        source_id=source_id,
        category=category,
        batch_id=batch_id,
        input_type=InputType.IMAGE,
        original_ext=image_path.suffix.lower(),
        page_id=page_id,
        relative_path=relative_input,
        page_image=output_path.relative_to(page_images_dir.parent).as_posix(),
        stage_status=StageStatus.INGESTED,
        page_image_sha256=sha256_file(output_path),
        source_metadata=metadata,
    )


def _pdf_page_count(pdf_path: Path) -> int | None:
    try:
        proc = subprocess.run(
            ["pdfinfo", str(pdf_path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError:
        return None
    for line in proc.stdout.splitlines():
        if line.startswith("Pages:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except ValueError:
                return None
    return None


def _pdf_text_excerpt(pdf_path: Path) -> str | None:
    pdftotext = shutil.which("pdftotext")
    if not pdftotext:
        return None
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        subprocess.run(
            [pdftotext, "-f", "1", "-l", "1", str(pdf_path), str(tmp_path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        text = tmp_path.read_text(encoding="utf-8", errors="ignore").strip()
        return text[:500] or None
    except subprocess.CalledProcessError:
        return None
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _ffprobe_image(path: Path) -> dict[str, Any]:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,pix_fmt",
            "-show_entries",
            "stream_tags=dpi",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    payload = json.loads(proc.stdout or "{}")
    streams = payload.get("streams", [])
    if not streams:
        return {}
    stream = streams[0]
    return {
        "width": stream.get("width"),
        "height": stream.get("height"),
        "pix_fmt": stream.get("pix_fmt"),
        "dpi": _safe_float(stream.get("tags", {}).get("dpi")),
    }


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _scale(original: int | None, normalized: int | None) -> float | None:
    if not original or not normalized:
        return None
    return normalized / original


def _hash_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _build_ingest_stats(records: list[dict | UnifiedSampleRecord]) -> dict[str, Any]:
    """构建统计信息，支持dict和UnifiedSampleRecord对象"""
    total_samples = len(records)
    valid_samples = 0
    pdf_count = 0
    image_count = 0
    
    for r in records:
        # 处理dict和UnifiedSampleRecord两种类型
        if isinstance(r, dict):
            is_active = r.get("is_active", True)
            input_type = r.get("input_type")
        else:
            is_active = r.is_active
            input_type = r.input_type
        
        if is_active:
            valid_samples += 1
        
        if input_type == "pdf" or input_type == InputType.PDF:
            pdf_count += 1
        elif input_type == "image" or input_type == InputType.IMAGE:
            image_count += 1
    
    return {
        "updated_at": __import__("datetime").datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "total_samples": total_samples,
        "valid_samples": valid_samples,
        "stage_status_counts": {"ingested": total_samples},
        "difficulty_histogram": {"easy": 0, "medium": 0, "hard": 0, "invalid": 0},
        "source_counts": {
            "pdf": pdf_count,
            "image": image_count,
        },
    }
