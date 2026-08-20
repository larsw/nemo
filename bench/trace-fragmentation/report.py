#!/usr/bin/env python3
"""Compute the memo-fragmentation factor and apply the pre-registered rule.

The rule was committed before any numbers existed (design record, decision 16),
and is applied at K=8 on locality-ordered shards:

    < 1.5x   proceed exactly as designed
    1.5-3x   proceed, but pull the shared explained-set into v1 scope
    > 3x     the farm thesis has failed; pivot

Reporting also splits out the locality-versus-random comparison, which tests a
different claim: not whether the farm is viable, but whether trie-range batching
is worth its complexity. If locality does not beat random, small random batches
with a dynamic queue are simpler and just as good.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

BAND_PROCEED = 1.5
BAND_PIVOT = 3.0
DECISION_K = 8


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {Path(sys.argv[0]).name} results.tsv", file=sys.stderr)
        return 2

    rows = list(csv.DictReader(Path(sys.argv[1]).open(encoding="utf-8"), delimiter="\t"))
    if not rows:
        print("error: no rows in results file", file=sys.stderr)
        return 1

    baseline = next((r for r in rows if r["ordering"] == "baseline"), None)
    if baseline is None:
        print("error: no baseline row; the whole-set run did not complete", file=sys.stderr)
        return 1

    i1 = int(baseline["inferences"])
    t1 = int(baseline["trace_ms"])
    if i1 == 0:
        print(
            "error: the baseline produced 0 inferences. Nothing was explained, so "
            "there is no fragmentation to measure.\n"
            "Most likely the corpus CURIEs did not expand to the IRIs the rules "
            "match on, leaving every chaining rule dead. Check sssom_to_nt.py "
            "output for dropped rows and guessed prefixes.",
            file=sys.stderr,
        )
        return 1

    # Group shard rows by (ordering, K).
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        if row["ordering"] == "baseline":
            continue
        groups[(row["ordering"], int(row["k"]))].append(row)

    print("Baseline (whole set, one run)")
    print(f"  facts        {int(baseline['facts']):>12,}")
    print(f"  inferences   {i1:>12,}")
    print(f"  conclusions  {int(baseline['conclusions']):>12,}")
    print(f"  trace_ms     {t1:>12,}   (residual: wall - import - reasoning - export)")
    print()

    header = f"{'ord':<5} {'K':>3} {'inferences':>13} {'F':>7} {'work':>7} {'slowest':>9} {'speedup':>8}"
    print(header)
    print("-" * len(header))

    summary: dict[tuple[str, int], float] = {}
    for (ordering, k) in sorted(groups, key=lambda key: (key[0], key[1])):
        shards = groups[(ordering, k)]
        inf_sum = sum(int(s["inferences"]) for s in shards)
        trace_sum = sum(int(s["trace_ms"]) for s in shards)
        trace_max = max(int(s["trace_ms"]) for s in shards)
        frag = inf_sum / i1
        work = trace_sum / t1 if t1 else float("nan")
        speedup = t1 / trace_max if trace_max else float("nan")
        summary[(ordering, k)] = frag
        print(
            f"{ordering:<5} {k:>3} {inf_sum:>13,} {frag:>7.2f} {work:>7.2f} "
            f"{trace_max:>8,}m {speedup:>8.2f}"
        )

    print()
    print("  F       = fragmentation: sum(shard inferences) / baseline inferences")
    print("  work    = sum(shard trace_ms) / baseline trace_ms")
    print("  slowest = trace_ms of the slowest shard (the ideal parallel wall clock)")
    print("  speedup = baseline trace_ms / slowest shard, i.e. the best case at this K")
    print()

    # --- Locality versus random, where both were measured -------------------
    pairs = [
        (k, summary[("loc", k)], summary[("rnd", k)])
        for k in sorted({k for (o, k) in summary if o == "loc"})
        if ("rnd", k) in summary
    ]
    if pairs:
        print("Locality vs random")
        for k, loc, rnd in pairs:
            verdict = "locality helps" if loc < rnd else "locality does not help"
            ratio = rnd / loc if loc else float("nan")
            print(f"  K={k:<3} loc F={loc:.2f}  rnd F={rnd:.2f}  ({ratio:.2f}x)  {verdict}")
        print()
        print("  If locality does not clearly beat random, trie-range batching is not")
        print("  earning its complexity: use small random batches with a pull queue.")
        print()

    # --- The pre-registered rule -------------------------------------------
    key = ("loc", DECISION_K)
    print("=" * 62)
    if key not in summary:
        available = sorted(k for (o, k) in summary if o == "loc")
        print(f"Pre-registered rule needs K={DECISION_K}, locality-ordered.")
        print(f"Not measured. Locality K values present: {available or 'none'}")
        print("Re-run with -k including 8 before drawing a conclusion.")
        return 1

    frag = summary[key]
    print(f"Pre-registered rule, K={DECISION_K}, locality-ordered: F = {frag:.2f}")
    print()
    if frag < BAND_PROCEED:
        print("  VERDICT  proceed exactly as designed.")
        print("           Fragmentation is small; locality batching is sufficient.")
    elif frag <= BAND_PIVOT:
        print("  VERDICT  proceed, but pull the shared explained-set into v1 scope.")
        print("           Do not defer it to a fallback -- it is load-bearing.")
    else:
        print("  VERDICT  the farm thesis has FAILED at this scale.")
        print("           Pivot to a shared-memo design, or to partitioned inference.")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
