#!/usr/bin/env python3
"""Generate a tiny synthetic corpus for exercising the harness itself.

This is a fixture, not evidence. It emits `chains` independent chains of
owl:equivalentClass edges, so contiguous subject-ID ranges line up exactly with
connected components and locality-ordered sharding scores a perfect F = 1.00.
Real mapping graphs do not look like this -- they have hub terms and one giant
component -- so never quote a number obtained from this file as a result. Use it
to check that the harness runs, the metric responds, and the verdict path works.

The random ordering is the informative half here: F should climb with K, because
shuffling destroys the component alignment. If it does not climb, the harness is
not measuring what it claims to.
"""

from __future__ import annotations

import argparse
from pathlib import Path

EQUIVALENT_CLASS = "<http://www.w3.org/2002/07/owl#equivalentClass>"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=Path("smoke/mappings.nt"))
    parser.add_argument("--chains", type=int, default=8)
    parser.add_argument("--length", type=int, default=5, help="edges per chain")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for chain in range(args.chains):
            for step in range(args.length):
                subject = f"<http://example.org/chain{chain}/t{step}>"
                obj = f"<http://example.org/chain{chain}/t{step + 1}>"
                out.write(f"{subject} {EQUIVALENT_CLASS} {obj} .\n")

    total = args.chains * args.length
    print(f"wrote {total} triples -> {args.output}")
    print("fixture only: locality F = 1.00 here is an artifact of independent chains")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
