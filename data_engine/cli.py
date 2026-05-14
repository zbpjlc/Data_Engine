from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_engine.clustering import cluster_records
from data_engine.embedding import extract_embeddings_for_records
from data_engine.ingest import run_ingest
from data_engine.manifests import read_manifest, write_manifest, find_stage_manifest
from data_engine.progress_tracker import progress_tracker
from data_engine.registry import DEFAULT_REGISTRY_PATH, SourceRegistry
from data_engine.status import collect_global_status, format_status_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="data_engine")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH), help="Path to sources.yaml")
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
    cluster.add_argument("--clusters", type=int, default=5, help="Number of clusters (default: 5)")
    cluster.add_argument("--auto-optimize", action="store_true", help="Auto-optimize cluster count")
    cluster.add_argument("--json", action="store_true", help="Emit JSON summary")

    status = subparsers.add_parser("status", help="Show batch progress")
    status.add_argument("--json", action="store_true", help="Emit JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    registry = SourceRegistry(Path(args.registry))

    if args.command == "register":
        source = registry.register(Path(args.path), args.source_id, args.category)
        print(f"registered source_id={source.id} category={source.category} root_path={source.root_path}")
        return 0

    if args.command == "scan":
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

    if args.command == "ingest":
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

    if args.command == "embed":
        import sys
        import time
        
        source = registry.get(args.source_id)
        batch_dir = Path(source.root_path) / args.batch
        manifests_dir = batch_dir / "manifests"
        manifest_path = find_stage_manifest(manifests_dir, "ingest")
        
        if not manifest_path or not manifest_path.exists():
            print(f"错误: manifest文件不存在", file=sys.stderr)
            return 1
        
        # 读取现有记录
        records = read_manifest(manifest_path)
        task_id = f"embed_{args.source_id}_{args.batch}"
        
        # 开始任务
        progress_tracker.start_task(
            task_id=task_id,
            task_type="embed",
            source_id=args.source_id,
            batch_id=args.batch,
            total=len(records),
            message=f"开始提取 {len(records)} 个样本的embedding"
        )
        
        try:
            # 提取embedding
            updated_records = extract_embeddings_for_records(records, batch_dir, task_id=task_id)
            
            # 写入更新后的记录
            write_manifest(manifest_path, updated_records)
            
            # 完成任务
            progress_tracker.complete_task(
                task_id=task_id,
                message=f"成功提取 {len(updated_records)} 个样本的embedding"
            )
            
            payload = {
                "records_updated": len(updated_records),
                "batch_id": args.batch,
                "source_id": args.source_id
            }
            
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, indent=2))
            else:
                print(f"已更新 {len(updated_records)} 个样本的embedding")
            return 0
            
        except Exception as e:
            progress_tracker.fail_task(
                task_id=task_id,
                error_message=str(e)
            )
            raise

    if args.command == "cluster":
        import sys
        
        source = registry.get(args.source_id)
        batch_dir = Path(source.root_path) / args.batch
        manifests_dir = batch_dir / "manifests"
        manifest_path = find_stage_manifest(manifests_dir, "ingest")
        
        if not manifest_path or not manifest_path.exists():
            print(f"错误: manifest文件不存在", file=sys.stderr)
            return 1
        
        # 读取现有记录
        records = read_manifest(manifest_path)
        task_id = f"cluster_{args.source_id}_{args.batch}"
        
        # 开始任务
        progress_tracker.start_task(
            task_id=task_id,
            task_type="cluster",
            source_id=args.source_id,
            batch_id=args.batch,
            total=len(records),
            message=f"开始聚类 {len(records)} 个样本"
        )
        
        try:
            # 执行聚类
            updated_records, cluster_stats = cluster_records(
                records, 
                n_clusters=args.clusters,
                auto_optimize=args.auto_optimize
            )
            
            # 写入更新后的记录
            write_manifest(manifest_path, updated_records)
            
            # 保存聚类统计
            artifacts_dir = batch_dir / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            with open(artifacts_dir / "cluster_stats.json", "w") as f:
                json.dump(cluster_stats, f, indent=2, ensure_ascii=False)
            
            # 完成任务
            progress_tracker.complete_task(
                task_id=task_id,
                message=f"成功聚类为 {cluster_stats.get('n_clusters', args.clusters)} 个簇"
            )
            
            payload = {
                "records_updated": len(updated_records),
                "cluster_stats": cluster_stats,
                "batch_id": args.batch,
                "source_id": args.source_id
            }
            
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, indent=2))
            else:
                print(f"已聚类 {len(updated_records)} 个样本")
                print(f"聚类数量: {cluster_stats.get('n_clusters', args.clusters)}")
                print(f"轮廓系数: {cluster_stats.get('silhouette_score', 0):.3f}")
                print(f"聚类分布: {cluster_stats.get('cluster_sizes', {})}")
            return 0
            
        except Exception as e:
            progress_tracker.fail_task(
                task_id=task_id,
                error_message=str(e)
            )
            raise

    if args.command == "status":
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

    parser.error("Unknown command")
    return 2
