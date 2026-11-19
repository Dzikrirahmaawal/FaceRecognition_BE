#!/usr/bin/env python3
"""Read-only summary of liveness shadow-mode logs.

Reads JSON log lines (one per line) from a file argument or stdin, extracts the
liveness shadow `data.score` and `data.would_pass` fields, and prints:
  count, min, mean, median, p5, p95 of scores, and the would_pass rate.

No images, crops, or embeddings are ever read or printed. Safe to run against
production log exports.

Usage:
    python scripts/summarize_liveness_shadow_log.py /var/log/app/shadow.log
    cat /var/log/app/shadow.log | python scripts/summarize_liveness_shadow_log.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def _percentile(sorted_values, q):
    """Linear-interpolation percentile (q in 0..100)."""
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * (q / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    source = Path(argv[0]) if argv else None

    scores = []
    would_pass_count = 0
    total = 0

    lines = source.read_text(encoding="utf-8").splitlines() if source else sys.stdin

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        data = record.get("data")
        if not isinstance(data, dict):
            continue
        if "score" not in data or "would_pass" not in data:
            continue
        score = data["score"]
        if score is None:
            continue
        try:
            scores.append(float(score))
        except (TypeError, ValueError):
            continue
        total += 1
        if data["would_pass"] is True:
            would_pass_count += 1

    if not scores:
        print("No liveness shadow log entries found.")
        return 0

    scores.sort()
    n = len(scores)
    mean = sum(scores) / n
    median = _percentile(scores, 50)
    p5 = _percentile(scores, 5)
    p95 = _percentile(scores, 95)
    would_pass_rate = (would_pass_count / n) * 100.0

    print(f"entries: {n}")
    print(f"min_score: {min(scores):.6f}")
    print(f"max_score: {max(scores):.6f}")
    print(f"mean_score: {mean:.6f}")
    print(f"median_score: {median:.6f}")
    print(f"p5_score: {p5:.6f}")
    print(f"p95_score: {p95:.6f}")
    print(f"would_pass_rate: {would_pass_rate:.2f}%  ({would_pass_count}/{n})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
