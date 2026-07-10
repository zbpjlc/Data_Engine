from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc

from data_engine.clustering import cluster_records
from data_engine.embedding import extract_embeddings_for_records
from data_engine.ingest import run_ingest
from data_engine.manifests import (
    read_manifest, write_manifest, find_stage_manifest,
    safe_merge, read_table_streaming, open_dataset, _lance_write_lock,
    ensure_lance_indexes, ocr_complete_filter,
)
from data_engine.progress_tracker import progress_tracker
from data_engine.registry import DEFAULT_REGISTRY_PATH, SourceRegistry
from data_engine.status import collect_global_status, format_status_report

logger = logging.getLogger("data_engine.cli")

CATEGORIES = ("text", "formula", "table")
STAGE_INGEST = "ingest"
FIELD_CONSISTENCY = "consistency_pattern"
FIELD_BLOCK_DIFF = "block_diff_json"
OCR_ENGINE_PREFIXES = ("paddle", "glm", "self")
OCR_SUFFIXES = ("_text", "_confidence", "_table", "_formula")
OCR_JSON_FIELDS = (
    "paddle_table", "glm_table", "self_table",
    "paddle_formula", "glm_formula", "self_formula",
)


def _positive_int(value: str) -> int:
    iv = int(value)
    if iv < 2:
        raise argparse.ArgumentTypeError(f"必须 >= 2，当前值: {iv}")
    return iv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="data_engine")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH), help="Path to sources.yaml")
    parser.add_argument("--log-file", help="日志输出文件路径")
    parser.add_argument("--log-level", default="WARNING",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="日志级别 (default: WARNING)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    register = subparsers.add_parser("register", help="Register a data source")
    register.add_argument("--path", required=True, help="Root path of the source")
    register.add_argument("--source-id", help="Explicit source id")
    register.add_argument("--category", help="Category name")

    scan = subparsers.add_parser("scan", help="Scan registered sources")
    scan.add_argument("--json", action="store_true", help="Emit JSON")

    ingest = subparsers.add_parser("ingest", help="Ingest a batch")
    ingest.add_argument("--source-id", required=True)
    ingest.add_argument("--batch", required=True)
    ingest.add_argument("--json", action="store_true", help="Emit JSON summary")

    embed = subparsers.add_parser("embed", help="Extract embeddings for a batch")
    embed.add_argument("--source-id", required=True)
    embed.add_argument("--batch", required=True)
    embed.add_argument("--json", action="store_true", help="Emit JSON summary")

    cluster = subparsers.add_parser("cluster", help="Cluster samples using K-means")
    cluster.add_argument("--source-id", required=True)
    cluster.add_argument("--batch", required=True)
    cluster.add_argument("--clusters", type=_positive_int, default=5, help="Number of clusters (default: 5, min: 2)")
    cluster.add_argument("--auto-optimize", action="store_true", help="Auto-optimize cluster count")
    cluster.add_argument("--json", action="store_true", help="Emit JSON summary")

    status = subparsers.add_parser("status", help="Show batch progress")
    status.add_argument("--json", action="store_true", help="Emit JSON")

    cmcv = subparsers.add_parser("cmcv", help="Cross-Model Consistency Verification")
    cmcv.add_argument("--source-id", required=True)
    cmcv.add_argument("--batch", required=True)
    cmcv.add_argument("--json", action="store_true", help="Emit JSON summary")

    return parser


def _resolve_batch(registry: SourceRegistry, source_id: str, batch: str):
    source = registry.get(source_id)
    batch_dir = Path(source.root_path) / batch
    if not batch_dir.is_dir():
        raise FileNotFoundError(
            f"batch 目录不存在: source_id={source_id} batch={batch} 路径={batch_dir}"
        )
    manifests_dir = batch_dir / "manifests"
    return source, batch_dir, manifests_dir


def _require_manifest(manifests_dir: Path, stage: str, source_id: str, batch: str) -> Path:
    path = find_stage_manifest(manifests_dir, stage)
    if not path or not path.exists():
        raise FileNotFoundError(
            f"manifest 文件不存在: source_id={source_id} batch={batch} stage={stage} "
            f"(路径: {manifests_dir / stage}.lance)"
        )
    return path


def _start_progress_task(
    task_type: str, source_id: str, batch: str, total: int, message: str
) -> str:
    task_id = f"{task_type}_{source_id}_{batch}"
    progress_tracker.start_task(
        task_id=task_id,
        task_type=task_type,
        source_id=source_id,
        batch_id=batch,
        total=total,
        message=message,
    )
    return task_id


def _print_output(payload: dict, as_json: bool, human_msg: str = "") -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif human_msg:
        print(human_msg)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.log_file:
        file_handler = logging.FileHandler(args.log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(file_handler)
        logging.getLogger().addHandler(file_handler)

    log_level = getattr(logging, args.log_level, logging.WARNING)
    logger.setLevel(log_level)
    logging.getLogger().setLevel(log_level)

    registry = SourceRegistry(Path(args.registry))

    if hasattr(args, "source_id") and args.source_id:
        try:
            registry.get(args.source_id)
        except KeyError:
            logger.error(f"source_id 不存在: {args.source_id}")
            return 1

    dispatch = {
        "register": cmd_register,
        "scan": cmd_scan,
        "ingest": cmd_ingest,
        "embed": cmd_embed,
        "cluster": cmd_cluster,
        "status": cmd_status,
        "cmcv": cmd_cmcv,
    }
    handler = dispatch.get(args.command)
    if handler is None:
        parser.error("Unknown command")
        return 2
    try:
        return handler(args, registry)
    except Exception:
        logger.exception(f"Command '{args.command}' failed")
        return 1


def cmd_register(args: argparse.Namespace, registry: SourceRegistry) -> int:
    source = registry.register(Path(args.path), args.source_id, args.category)
    print(f"registered source_id={source.id} category={source.category} root_path={source.root_path}")
    return 0


def cmd_scan(args: argparse.Namespace, registry: SourceRegistry) -> int:
    scan_results = registry.scan()
    if args.json:
        print(json.dumps([item.model_dump(mode="json") for item in scan_results], ensure_ascii=False, indent=2))
    else:
        for item in scan_results:
            print(
                f"{item.source_id}\tcategory={item.category}\tonline={item.online}\t"
                f"enabled={item.enabled}\tbatches={item.batch_count}\tmanifests={item.manifest_count}"
            )
    return 0


def cmd_ingest(args: argparse.Namespace, registry: SourceRegistry) -> int:
    result = run_ingest(registry, args.source_id, args.batch)
    payload = {
        "records_written": len(result.records),
        "stats": result.stats,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"wrote {len(result.records)} records")
        print(json.dumps(result.stats, ensure_ascii=False, indent=2))
    return 0


def cmd_embed(args: argparse.Namespace, registry: SourceRegistry) -> int:
    _, batch_dir, manifests_dir = _resolve_batch(registry, args.source_id, args.batch)
    manifest_path = _require_manifest(manifests_dir, STAGE_INGEST, args.source_id, args.batch)

    records = read_manifest(manifest_path)
    task_id = _start_progress_task(
        "embed", args.source_id, args.batch, len(records),
        f"开始提取 {len(records)} 个样本的embedding"
    )

    try:
        updated_records = extract_embeddings_for_records(records, batch_dir, task_id=task_id)
        write_manifest(manifest_path, updated_records)
        progress_tracker.complete_task(
            task_id=task_id,
            message=f"成功提取 {len(updated_records)} 个样本的embedding"
        )
        _print_output(
            {"records_updated": len(updated_records), "batch_id": args.batch, "source_id": args.source_id},
            args.json,
            f"已更新 {len(updated_records)} 个样本的embedding",
        )
        return 0
    except Exception as e:
        progress_tracker.fail_task(task_id=task_id, error_message=str(e))
        raise


def cmd_cluster(args: argparse.Namespace, registry: SourceRegistry) -> int:
    _, batch_dir, manifests_dir = _resolve_batch(registry, args.source_id, args.batch)
    manifest_path = _require_manifest(manifests_dir, STAGE_INGEST, args.source_id, args.batch)

    records = read_manifest(manifest_path)
    task_id = _start_progress_task(
        "cluster", args.source_id, args.batch, len(records),
        f"开始聚类 {len(records)} 个样本"
    )

    try:
        updated_records, cluster_stats = cluster_records(
            records,
            n_clusters=args.clusters,
            auto_optimize=args.auto_optimize
        )
        write_manifest(manifest_path, updated_records)

        artifacts_dir = batch_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        with open(artifacts_dir / "cluster_stats.json", "w") as f:
            json.dump(cluster_stats, f, indent=2, ensure_ascii=False)

        n_clusters = cluster_stats.get("n_clusters", args.clusters)
        progress_tracker.complete_task(
            task_id=task_id,
            message=f"成功聚类为 {n_clusters} 个簇"
        )

        human_msg = (
            f"已聚类 {len(updated_records)} 个样本\n"
            f"聚类数量: {n_clusters}\n"
            f"轮廓系数: {cluster_stats.get('silhouette_score', 0):.3f}\n"
            f"聚类分布: {cluster_stats.get('cluster_sizes', {})}"
        )
        _print_output(
            {"records_updated": len(updated_records), "cluster_stats": cluster_stats,
             "batch_id": args.batch, "source_id": args.source_id},
            args.json,
            human_msg,
        )
        return 0
    except Exception as e:
        progress_tracker.fail_task(task_id=task_id, error_message=str(e))
        raise


def cmd_status(args: argparse.Namespace, registry: SourceRegistry) -> int:
    global_status = collect_global_status(registry)
    if args.json:
        print(
            json.dumps(
                {
                    "sources": [item.model_dump(mode="json") for item in global_status.sources],
                    "batches": [item.model_dump(mode="json") for item in global_status.batches],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(format_status_report(global_status))
    return 0


def cmd_cmcv(args: argparse.Namespace, registry: SourceRegistry) -> int:
    from data_engine.ocr.cmcv import CMCVEngine

    source, batch_dir, manifests_dir = _resolve_batch(registry, args.source_id, args.batch)

    read_errors: list[str] = []
    write_errors: list[str] = []
    arrow_tables: list[pa.Table] = []
    block_idx_type = pa.int32()

    for cat in CATEGORIES:
        lp = manifests_dir / f"{cat}.lance"
        if not lp.exists():
            continue
        try:
            with _lance_write_lock:
                ds = open_dataset(lp)
                block_idx_type = ds.schema.field("block_idx").type
                all_cols = ds.schema.names
                if "consistency_pattern" not in all_cols:
                    continue
                read_cols = ["sample_id", "block_idx", "block_type"]
                for prefix in OCR_ENGINE_PREFIXES:
                    for suffix in OCR_SUFFIXES:
                        col = f"{prefix}{suffix}"
                        if col in all_cols:
                            read_cols.append(col)
                cols = [c for c in read_cols if c in all_cols]
                ocr_filter = ocr_complete_filter(set(all_cols))
                filters = ["consistency_pattern IS NULL"]
                if ocr_filter:
                    filters.append(ocr_filter)
                batches = list(ds.to_batches(columns=cols, filter=" AND ".join(filters)))
            if batches:
                arrow_tables.append(pa.Table.from_batches(batches))
        except Exception as e:
            read_errors.append(f"{cat}.lance: {e}")
            logger.error(f"[CMCV] 读取 {cat}.lance 失败: source_id={args.source_id} "
                         f"batch={args.batch} path={lp} error={e}")

    if not arrow_tables:
        logger.error(f"[CMCV] 无可用 block: source_id={args.source_id} batch={args.batch} "
                     f"检查 {', '.join(str(manifests_dir / f'{c}.lance') for c in CATEGORIES)}")
        return 1

    combined = pa.concat_tables(arrow_tables)
    del arrow_tables
    total_rows = len(combined)

    task_id = _start_progress_task(
        "cmcv", args.source_id, args.batch, total_rows,
        f"开始一致性比较: {total_rows} 个 block",
    )

    cmcv = CMCVEngine(use_visual_cdm=True)

    def _cmcv_progress(cur, tot, msg):
        progress_tracker.update_progress(task_id=task_id, current=cur, message=f"[{cur}/{tot}] {msg}", total=tot)

    result_table, page_tiers = cmcv.process_element_batch_arrow(combined, progress_callback=_cmcv_progress)
    del combined

    up_sids = result_table.column("sample_id")
    up_bidxs = result_table.column("block_idx")
    up_pats = result_table.column(FIELD_CONSISTENCY)
    up_diffs = result_table.column(FIELD_BLOCK_DIFF)
    update_table = pa.table({
        "sample_id": up_sids,
        "block_idx": up_bidxs,
        "_new_pattern": up_pats,
        "_new_diff": up_diffs,
    })
    del result_table

    for cat in CATEGORIES:
        lp = manifests_dir / f"{cat}.lance"
        if not lp.exists():
            continue
        try:
            with _lance_write_lock:
                ds = open_dataset(lp)
                col_names = set(ds.schema.names)
                if FIELD_CONSISTENCY not in col_names or FIELD_BLOCK_DIFF not in col_names:
                    continue
                # 流式读取，避免全表加载
                read_cols = ["sample_id", "block_idx", FIELD_CONSISTENCY, FIELD_BLOCK_DIFF]
                scanner = ds.scanner(columns=read_cols)
                current_batches = list(scanner.to_batches())
                if not current_batches:
                    continue
                current_table = pa.concat_tables(current_batches)

                joined = current_table.join(
                    update_table,
                    keys=["sample_id", "block_idx"],
                    join_type="left outer",
                )
                is_updated = pc.or_(
                    pc.is_valid(joined.column("_new_pattern")),
                    pc.is_valid(joined.column("_new_diff")),
                )
                filtered = joined.filter(is_updated)
                if len(filtered) == 0:
                    continue

                final_consistency = pc.coalesce(filtered.column("_new_pattern"), filtered.column(FIELD_CONSISTENCY))
                final_diff = pc.coalesce(filtered.column("_new_diff"), filtered.column(FIELD_BLOCK_DIFF))
                merged = pa.table({
                    "sample_id": filtered.column("sample_id"),
                    "block_idx": filtered.column("block_idx"),
                    FIELD_CONSISTENCY: final_consistency,
                    FIELD_BLOCK_DIFF: final_diff,
                })

                safe_merge(ds, merged, ["sample_id", "block_idx"], context=f"cmcv/{cat}")
        except Exception as e:
            write_errors.append(f"{cat}.lance: {e}")
            logger.error(f"[CMCV] 写回 {cat}.lance 失败: source_id={args.source_id} "
                         f"batch={args.batch} path={lp} error={e}")

    for cat in CATEGORIES:
        lp = manifests_dir / f"{cat}.lance"
        if lp.exists():
            ensure_lance_indexes(lp, ["consistency_pattern", "block_idx"])

    progress_tracker.complete_task(
        task_id=task_id,
        message=f"完成: {len(update_table)} block, {len(page_tiers)} 页面",
    )

    tier_hist: dict[str, int] = {}
    for t in page_tiers.values():
        tier_hist[t] = tier_hist.get(t, 0) + 1

    payload = {
        "blocks_processed": len(update_table),
        "pages_tiered": len(page_tiers),
        "tier_histogram": tier_hist,
    }
    if read_errors:
        payload["read_errors"] = read_errors
    if write_errors:
        payload["write_errors"] = write_errors

    human_msg = f"CMCV 完成: {len(update_table)} block, 页面分桶 {tier_hist}"
    if read_errors:
        human_msg += f"\n读取失败 {len(read_errors)} 个: {'; '.join(read_errors)}"
    if write_errors:
        human_msg += f"\n写回失败 {len(write_errors)} 个: {'; '.join(write_errors)}"

    _print_output(payload, args.json, human_msg)

    if len(update_table) > 100_000:
        for cat in CATEGORIES:
            lp = manifests_dir / f"{cat}.lance"
            if not lp.exists():
                continue
            try:
                with _lance_write_lock:
                    ds = open_dataset(lp)
                    ds.compact_files()
                    ds.cleanup_old_versions(retain_versions=2)
                logger.info(f"[CMCV] 已压缩 {cat}.lance")
            except Exception as e:
                logger.warning(f"[CMCV] 压缩 {cat}.lance 失败: {e}")

    if read_errors or write_errors:
        return 1
    return 0
