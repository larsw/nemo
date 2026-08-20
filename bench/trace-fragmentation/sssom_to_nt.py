#!/usr/bin/env python3
"""Convert an SSSOM TSV mapping set into N-Triples.

The SSSOM chain rules in sssom-chain.rls match on full IRIs (owl:equivalentClass
and friends are written out longhand), while SSSOM TSV stores CURIEs. So the
CURIEs have to be expanded, or the transitivity and role-chain rules simply never
fire and the whole experiment measures nothing.

Prefixes come from the `curie_map` in the commented YAML header when present,
falling back to the well-known vocabularies the rules actually reference. Rows
whose prefix cannot be resolved are dropped and counted, because silently
emitting a malformed IRI would corrupt the corpus rather than shrink it.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Vocabularies the rules in sssom-chain.rls match on directly. If a mapping set
# omits any of these from its curie_map, the corresponding rules would be dead.
FALLBACK_PREFIXES = {
    "owl": "http://www.w3.org/2002/07/owl#",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "semapv": "https://w3id.org/semapv/vocab/",
    "oboInOwl": "http://www.geneontology.org/formats/oboInOwl#",
    "obo": "http://purl.obolibrary.org/obo/",
}

# Anything not otherwise known is treated as an OBO-style prefix, which is how
# the overwhelming majority of biomedical mapping sets identify their terms.
OBO_TEMPLATE = "http://purl.obolibrary.org/obo/{prefix}_{local}"


def read_curie_map(path: Path) -> dict[str, str]:
    """Extract `curie_map` from the commented YAML header.

    Parsed by hand rather than with PyYAML so the harness has no dependencies
    beyond the standard library.
    """
    prefixes: dict[str, str] = {}
    in_map = False
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            if not raw.startswith("#"):
                break
            line = raw[1:].rstrip("\n")
            stripped = line.strip()
            if stripped.startswith("curie_map:"):
                in_map = True
                continue
            if in_map:
                # The block ends at the first non-indented, non-empty line.
                if stripped and not line.startswith((" ", "\t")):
                    in_map = False
                    continue
                if ":" not in stripped:
                    continue
                key, _, value = stripped.partition(":")
                value = value.strip().strip('"').strip("'")
                if value:
                    prefixes[key.strip()] = value
    return prefixes


def expand(curie: str, prefixes: dict[str, str]) -> str | None:
    """Expand a CURIE to a full IRI, or return None if it cannot be resolved."""
    curie = curie.strip()
    if not curie:
        return None
    if curie.startswith(("http://", "https://")):
        return curie
    prefix, sep, local = curie.partition(":")
    if not sep or not local:
        return None
    if prefix in prefixes:
        return prefixes[prefix] + local
    if prefix in FALLBACK_PREFIXES:
        return FALLBACK_PREFIXES[prefix] + local
    # Unknown prefix: assume OBO. Recorded by the caller so the fraction of
    # guessed IRIs is visible rather than hidden.
    return OBO_TEMPLATE.format(prefix=prefix, local=local)


def body_reader(path: Path):
    """Yield TSV rows, skipping the commented YAML header."""
    with path.open(encoding="utf-8", newline="") as handle:
        lines = [line for line in handle if not line.startswith("#")]
    yield from csv.DictReader(lines, delimiter="\t")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="SSSOM TSV file")
    parser.add_argument("output", type=Path, help="N-Triples output file")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="keep only the first N mappings (0 = all). Use this to build the "
        "smaller scales; transitive closure over a full mapping set can be "
        "very large.",
    )
    args = parser.parse_args()

    prefixes = read_curie_map(args.input)
    required = {"subject_id", "predicate_id", "object_id"}

    written = 0
    dropped = 0
    guessed_prefixes: set[str] = set()
    known = set(prefixes) | set(FALLBACK_PREFIXES)

    with args.output.open("w", encoding="utf-8") as out:
        rows = body_reader(args.input)
        first = next(rows, None)
        if first is None:
            print(f"error: {args.input} has no data rows", file=sys.stderr)
            return 1
        missing = required - set(first)
        if missing:
            print(
                f"error: {args.input} is missing SSSOM columns: {sorted(missing)}",
                file=sys.stderr,
            )
            return 1

        for row in [first, *rows]:
            if args.limit and written >= args.limit:
                break
            triple = []
            for column in ("subject_id", "predicate_id", "object_id"):
                raw = (row.get(column) or "").strip()
                prefix = raw.partition(":")[0]
                if prefix and prefix not in known and not raw.startswith("http"):
                    guessed_prefixes.add(prefix)
                iri = expand(raw, prefixes)
                if iri is None:
                    triple = []
                    break
                triple.append(iri)
            if len(triple) != 3:
                dropped += 1
                continue
            out.write(f"<{triple[0]}> <{triple[1]}> <{triple[2]}> .\n")
            written += 1

    print(f"wrote   {written} triples -> {args.output}")
    print(f"dropped {dropped} rows (unresolvable subject, predicate, or object)")
    print(f"prefixes from curie_map: {len(prefixes)}")
    if guessed_prefixes:
        preview = ", ".join(sorted(guessed_prefixes)[:12])
        print(
            f"note: {len(guessed_prefixes)} prefix(es) not in curie_map were "
            f"expanded as OBO: {preview}"
        )
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
