from __future__ import annotations

import json
import os
import sys
from typing import Any

sys.path.insert(0, "/workspace")

from src.metrics.cdm_metric import CDM


def _score_pair(cdm: CDM, latex_a: str, latex_b: str, img_id: str) -> dict[str, Any]:
    try:
        metrics = cdm.evaluate(latex_a, latex_b, img_id)
        metrics.setdefault("F1_score", 0.0)
        return {"score": float(metrics.get("F1_score", 0.0)), "error": metrics.get("cdm_eval_error")}
    except Exception as exc:
        return {"score": 0.0, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw)
    except Exception as exc:
        print(json.dumps({"results": [{"score": 0.0, "error": f"bad_json:{exc}"}]}, ensure_ascii=False))
        return 0

    pairs = payload.get("pairs") or []
    timeout_sec = payload.get("timeout_sec", 120)
    results: list[dict[str, Any]] = []
    cdm = CDM()
    for idx, item in enumerate(pairs):
        latex_a = (item.get("latex_a") or "").strip()
        latex_b = (item.get("latex_b") or "").strip()
        if not latex_a and not latex_b:
            results.append({"score": 1.0, "error": None})
            continue
        if not latex_a or not latex_b:
            results.append({"score": 0.0, "error": None})
            continue
        results.append(_score_pair(cdm, latex_a, latex_b, f"batch_{idx}"))
    print(json.dumps({"results": results}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
