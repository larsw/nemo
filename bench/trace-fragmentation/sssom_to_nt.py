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


def scale_by_components(triples: list[tuple[str, str, str]], budget: int):
    """Select whole connected components, largest first, up to an edge budget.

    Taking the first N rows instead does not work and produces a silently
    useless corpus: SSSOM files are grouped by subject, so a prefix of one is
    bipartite -- measured at zero nodes appearing as both subject and object
    across 2k and 20k row slices of biomappings. With no such pivot node, no
    transitivity or role-chain rule in sssom-chain.rls can fire, the closure
    derives nothing, and the experiment measures nothing.

    Components are computed on the UNDIRECTED graph, because the role-chain
    rules traverse mappings in both directions.
    """
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:  # path compression
            parent[node], node = root, parent[node]
        return root

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for subject, _, obj in triples:
        union(subject, obj)

    components: dict[str, list[tuple[str, str, str]]] = {}
    for triple in triples:
        components.setdefault(find(triple[0]), []).append(triple)

    selected: list[tuple[str, str, str]] = []
    skipped_too_large = 0
    for edges in sorted(components.values(), key=len, reverse=True):
        if len(selected) + len(edges) > budget:
            skipped_too_large += 1
            continue
        selected.extend(edges)
        if len(selected) >= budget:
            break

    return selected, len(components), skipped_too_large


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
        help="approximate edge budget for a smaller scale (0 = keep everything). "
        "Whole connected components are taken, largest first, until the budget "
        "is reached -- never a prefix of the file. See scale_by_components().",
    )
    args = parser.parse_args()

    prefixes = read_curie_map(args.input)
    required = {"subject_id", "predicate_id", "object_id"}

    dropped = 0
    guessed_prefixes: set[str] = set()
    known = set(prefixes) | set(FALLBACK_PREFIXES)

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

    triples: list[tuple[str, str, str]] = []
    for row in [first, *rows]:
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
        triples.append((triple[0], triple[1], triple[2]))

    total_components = None
    skipped_too_large = 0
    if args.limit and len(triples) > args.limit:
        triples, total_components, skipped_too_large = scale_by_components(
            triples, args.limit
        )

    with args.output.open("w", encoding="utf-8") as out:
        for subject, predicate, obj in triples:
            out.write(f"<{subject}> <{predicate}> <{obj}> .\n")

    # A node appearing as both subject and object is what every chaining rule
    # needs. Report it, because zero pivots means the corpus is useless for this
    # experiment however many triples it has.
    subjects = {t[0] for t in triples}
    objects = {t[2] for t in triples}
    pivots = len(subjects & objects)

    print(f"wrote   {len(triples)} triples -> {args.output}")
    print(f"dropped {dropped} rows (unresolvable subject, predicate, or object)")
    print(f"prefixes from curie_map: {len(prefixes)}")
    if total_components is not None:
        print(
            f"scaled by whole components: {total_components} components in the "
            f"source, {skipped_too_large} skipped as too large for the budget"
        )
    print(f"pivot nodes (both subject and object): {pivots}")
    if not pivots:
        print(
            "warning: no pivot nodes, so no transitivity or role-chain rule can "
            "fire. This corpus will derive nothing and measure nothing.",
            file=sys.stderr,
        )
    return 0 if triples else 1


if __name__ == "__main__":
    raise SystemExit(main())
