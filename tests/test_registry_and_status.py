from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from data_engine.hashing import sample_id_for_page
from data_engine.manifests import write_json, write_manifest
from data_engine.models import InputType, SourceMetadata, StageStatus, UnifiedSampleRecord
from data_engine.registry import SourceRegistry
from data_engine.status import collect_global_status


class RegistryStatusTests(unittest.TestCase):
    def test_register_and_scan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry = SourceRegistry(root / "sources.yaml")
            source_dir = root / "source_a"
            (source_dir / "batch_001" / "manifests").mkdir(parents=True)
            registry.register(source_dir, "disk_01", "finance")
            scanned = registry.scan()
            self.assertEqual(len(scanned), 1)
            self.assertTrue(scanned[0].online)
            self.assertEqual(scanned[0].batch_count, 1)

    def test_status_aggregates_lance_and_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "source_a"
            batch_dir = source_dir / "batch_001"
            (batch_dir / "manifests").mkdir(parents=True)
            (batch_dir / "artifacts").mkdir(parents=True)
            registry = SourceRegistry(root / "sources.yaml")
            registry.register(source_dir, "disk_01", "finance")

            record = UnifiedSampleRecord(
                sample_id=sample_id_for_page("disk_01", "finance", "batch_001", "a" * 64),
                source_id="disk_01",
                category="finance",
                batch_id="batch_001",
                input_type=InputType.IMAGE,
                original_ext=".png",
                page_id="a" * 64,
                relative_path="single.png",
                page_image="page_images/single.png",
                stage_status=StageStatus.INGESTED,
                page_image_sha256="b" * 64,
                source_metadata=SourceMetadata(normalized_width=10, normalized_height=20),
            )
            write_manifest(batch_dir / "manifests" / "ingest.lance", [record.model_dump(mode="json")])
            write_json(
                batch_dir / "artifacts" / "stats.json",
                {
                    "total_samples": 1,
                    "valid_samples": 1,
                    "difficulty_histogram": {"easy": 1, "medium": 0, "hard": 0, "invalid": 0},
                    "updated_at": "2026-01-01T00:00:00Z",
                },
            )

            status = collect_global_status(registry)
            self.assertEqual(len(status.batches), 1)
            self.assertEqual(status.batches[0].stage_status, "ingested")
            self.assertEqual(status.batches[0].sample_count, 1)
            self.assertEqual(status.batches[0].difficulty_histogram["easy"], 1)

    def test_sample_id_is_hex_and_deterministic(self) -> None:
        first = sample_id_for_page("s1", "finance", "batch_001", "f" * 64)
        second = sample_id_for_page("s1", "finance", "batch_001", "f" * 64)
        self.assertEqual(first, second)
        int(first, 16)


if __name__ == "__main__":
    unittest.main()
