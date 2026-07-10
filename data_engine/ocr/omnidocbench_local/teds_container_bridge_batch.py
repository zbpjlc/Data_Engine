from __future__ import annotations

import json
import sys

sys.path.insert(0, "/workspace")

from src.metrics.table_metric import TEDS


def main() -> int:
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw)
    except Exception as exc:
        print(json.dumps({"results": [{"score": 0.0, "error": f"bad_json:{exc}"}]}, ensure_ascii=False))
        return 0

    pairs = payload.get("pairs") or []
    structure_only = payload.get("structure_only", False)
    results: list[dict] = []
    teds = TEDS(structure_only=structure_only, n_jobs=1)
    for idx, item in enumerate(pairs):
        html_a = (item.get("html_a") or "").strip()
        html_b = (item.get("html_b") or "").strip()
        if not html_a and not html_b:
            results.append({"score": 1.0, "error": None})
            continue
        if not html_a or not html_b:
            results.append({"score": 0.0, "error": None})
            continue
        try:
            score = teds.evaluate(html_a, html_b)
            results.append({"score": float(score), "error": None})
        except Exception as exc:
            results.append({"score": 0.0, "error": f"{type(exc).__name__}: {exc}"})
    print(json.dumps({"results": results}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
