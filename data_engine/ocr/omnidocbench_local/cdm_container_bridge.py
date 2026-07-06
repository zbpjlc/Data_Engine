from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, "/workspace")

from src.metrics.cdm_metric import CDM


def main() -> int:
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw)
    except Exception as exc:
        print(json.dumps({"F1_score": 0.0, "cdm_eval_error": f"bad_json:{exc}"}, ensure_ascii=False))
        return 0

    latex_a = payload.get("latex_a") or ""
    latex_b = payload.get("latex_b") or ""
    timeout_sec = payload.get("timeout_sec", 120)
    img_id = payload.get("img_id", "bridge_fallback")

    cdm = CDM()
    try:
        metrics = cdm.evaluate(latex_a, latex_b, img_id)
        metrics.setdefault("F1_score", 0.0)
        print(json.dumps(metrics, ensure_ascii=False))
    except Exception as exc:
        print(json.dumps({"F1_score": 0.0, "cdm_eval_error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
