#!/usr/bin/env bash
# Fetch a public SSSOM mapping set and build the corpus scales the experiment
# sweeps over.
#
# Decision 13 in the design record: use public mapping sets at 2-3 scales with
# the reporter's rules verbatim, so the result is reproducible by anyone and
# exercises the real hub skew and component structure of mapping data. Synthetic
# graphs would not.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORPUS_DIR="${CORPUS_DIR:-$HERE/corpus}"

# Biomappings: a public, actively maintained SSSOM mapping set with the CURIE
# map in its header. Any SSSOM TSV works; override with --from-file.
SOURCE_URL="${SOURCE_URL:-https://raw.githubusercontent.com/biopragmatics/biomappings/master/export/biomappings.sssom.tsv}"

# Scales. Small ones exist so a first pass finishes quickly and so the
# fragmentation factor can be seen as a function of corpus size, not just at one
# point.
SCALES=("${SCALES[@]:-2000 20000 0}")

from_file=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --from-file) from_file="$2"; shift 2 ;;
    --url)       SOURCE_URL="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
      echo
      echo "usage: $(basename "$0") [--from-file SSSOM.tsv] [--url URL]"
      echo "env:   CORPUS_DIR=$CORPUS_DIR  SCALES='2000 20000 0'  (0 = full set)"
      exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$CORPUS_DIR"
raw="$CORPUS_DIR/source.sssom.tsv"

if [[ -n "$from_file" ]]; then
  cp "$from_file" "$raw"
  echo "using local mapping set: $from_file"
elif [[ -s "$raw" ]]; then
  echo "reusing already-downloaded $raw (delete it to refetch)"
else
  echo "fetching $SOURCE_URL"
  if ! curl -fsSL "$SOURCE_URL" -o "$raw"; then
    cat >&2 <<EOF

Could not download the mapping set. Two ways forward:

  1. Pass a different source:  $(basename "$0") --url <URL>
  2. Supply a local SSSOM TSV: $(basename "$0") --from-file path/to/set.sssom.tsv

Any SSSOM TSV with subject_id / predicate_id / object_id columns will do.
EOF
    exit 1
  fi
fi

# shellcheck disable=SC2206
read -r -a scale_list <<<"${SCALES[*]}"
for limit in "${scale_list[@]}"; do
  if [[ "$limit" == "0" ]]; then
    name="full"
    args=()
  else
    name="$limit"
    args=(--limit "$limit")
  fi
  out="$CORPUS_DIR/mappings-$name.nt"
  echo
  echo "== scale $name =="
  python3 "$HERE/sssom_to_nt.py" "$raw" "$out" "${args[@]}"
done

echo
echo "corpus ready in $CORPUS_DIR"
ls -la "$CORPUS_DIR"/*.nt
