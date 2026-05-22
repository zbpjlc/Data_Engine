#!/usr/bin/env python3
"""按 sample_id 去重 Lance manifest（增量写入，低内存）"""

import sys
from pathlib import Path
import lance
import pyarrow as pa


def check_duplicates(manifest_path: str | Path) -> dict:
    """检查是否有重复的 sample_id"""
    path = Path(manifest_path)
    if not path.exists():
        return {"error": f"文件不存在: {path}"}

    ds = lance.dataset(str(path))
    table = ds.to_table(columns=["sample_id"])
    ids = [r["sample_id"] for r in table.to_pylist()]
    total = len(ids)
    unique = len(set(ids))
    duplicates = total - unique

    return {
        "path": str(path),
        "total": total,
        "unique": unique,
        "duplicates": duplicates,
        "has_duplicates": duplicates > 0,
    }


def dedup_manifest(manifest_path: str | Path) -> dict:
    path = Path(manifest_path)
    temp_path = path.parent / "ingest_dedup.lance"

    if not path.exists():
        return {"error": f"文件不存在: {path}"}

    ds = lance.dataset(str(path))
    total_before = ds.count_rows()
    schema = ds.schema

    # 用 set 追踪已见的 sample_id（只存ID，省内存）
    seen_ids: set[str] = set()
    removed = 0
    kept = 0
    first_write = True

    # 删除旧的临时文件
    if temp_path.exists():
        import shutil
        shutil.rmtree(temp_path)

    chunk_size = 50000
    for offset in range(0, total_before, chunk_size):
        chunk = ds.to_table(offset=offset, limit=chunk_size)
        kept_records = []
        for row in chunk.to_pylist():
            sid = row.get("sample_id")
            if sid and sid not in seen_ids:
                seen_ids.add(sid)
                kept_records.append(row)
                kept += 1
            else:
                removed += 1

        # 增量写入当前chunk
        if kept_records:
            table = pa.Table.from_pylist(kept_records, schema=schema)
            mode = "overwrite" if first_write else "append"
            lance.write_dataset(table, str(temp_path), mode=mode)
            first_write = False
            kept_records = None  # 释放内存

        print(f"  处理 {min(offset + chunk_size, total_before)}/{total_before}, 已保留: {kept}, 已删除: {removed}", flush=True)

    # 替换原文件
    import shutil
    shutil.rmtree(path)
    shutil.move(str(temp_path), str(path))

    return {"path": str(path), "before": total_before, "after": kept, "removed": removed}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法:")
        print("  python dedup_manifest.py check <manifest.lance>   # 检查是否有重复")
        print("  python dedup_manifest.py dedup <manifest.lance>   # 去重")
        sys.exit(1)

    cmd = sys.argv[1]
    if len(sys.argv) < 3:
        print("错误: 请提供 manifest 路径")
        sys.exit(1)

    manifest = sys.argv[2]

    if cmd == "check":
        result = check_duplicates(manifest)
        print(f"总行数: {result['total']}")
        print(f"唯一数: {result['unique']}")
        print(f"重复数: {result['duplicates']}")
        if result['has_duplicates']:
            print("需要去重")
        else:
            print("无需去重")
    elif cmd == "dedup":
        print(f"开始去重: {manifest}", flush=True)
        result = dedup_manifest(manifest)
        print(f"去重前: {result['before']}", flush=True)
        print(f"去重后: {result['after']}", flush=True)
        print(f"删除:  {result['removed']}", flush=True)
    else:
        print(f"未知命令: {cmd}")
        sys.exit(1)
