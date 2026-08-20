#!/usr/bin/env bash
# Step 0 of the trace-farm design record: measure how much the tracer's memo
# fragments when a fact set is split across independent runs.
#
# Why this exists: trace_ground_facts threads ONE ExecutionTrace through every
# fact in a batch (execution/execution_engine/tracing/simple.rs:253-273), and
# trace_recursive returns immediately on an already-known fact (:43-47). So
# explaining fact #2 reuses everything computed for fact #1 -- and splitting the
# fact set across N workers makes each of them re-derive the shared substructure.
# If that penalty approaches N even with locality-ordered shards, the whole
# parallel-explanation design fails, and it fails for the price of a few
# benchmark runs instead of an implementation.
#
# The metric is exact and needs no instrumentation: list_of_inferences
# deduplicates within a run (execution/tracing/trace.rs:369-411), so
#
#     F = sum(inferences across shards) / inferences(single whole-set run)
#
# IS the fragmentation factor.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

BIN="${BIN:-$REPO/target/release/nmo}"
DATA=""
OUT="${OUT:-$HERE/results}"
KS="2 4 8 16"
MAX_FACTS=0
ORDERINGS="loc rnd"

usage() {
  cat <<EOF
usage: $(basename "$0") -d CORPUS.nt [options]

  -d FILE   N-Triples corpus of asserted mappings (see fetch-corpus.sh)
  -k LIST   shard counts to sweep            (default: "$KS")
  -m N      trace only the first N facts     (default: 0 = all)
  -o DIR    output directory                 (default: $OUT)
  -b PATH   nmo binary                       (default: $BIN)
  -O LIST   orderings: loc, rnd, or both     (default: "$ORDERINGS")

The pre-registered decision rule is applied at K=8, locality-ordered:
  < 1.5x  proceed as designed
  1.5-3x  proceed, but pull the shared explained-set into v1 scope
  > 3x    the farm thesis has failed; pivot
EOF
}

while getopts "d:k:m:o:b:O:h" opt; do
  case "$opt" in
    d) DATA="$OPTARG" ;;
    k) KS="$OPTARG" ;;
    m) MAX_FACTS="$OPTARG" ;;
    o) OUT="$OPTARG" ;;
    b) BIN="$OPTARG" ;;
    O) ORDERINGS="$OPTARG" ;;
    h) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done

[[ -n "$DATA" ]] || { echo "error: -d CORPUS.nt is required" >&2; usage >&2; exit 2; }
[[ -s "$DATA" ]] || { echo "error: corpus not found or empty: $DATA" >&2; exit 1; }
[[ -x "$BIN"  ]] || { echo "error: nmo not found at $BIN (cargo build --release -p nemo-cli)" >&2; exit 1; }
for tool in jq python3 split shuf awk; do
  command -v "$tool" >/dev/null || { echo "error: $tool is required" >&2; exit 1; }
done

WORK="$OUT/work"
rm -rf "$WORK"
mkdir -p "$WORK"
RESULTS="$OUT/results.tsv"
printf 'ordering\tk\tshard\tfacts\twall_ms\timport_ms\treasoning_ms\texport_ms\ttrace_ms\tinferences\tconclusions\n' >"$RESULTS"

# Pull one figure out of nemo's short report. Absent lines count as 0, which is
# correct: "Data export:" is only printed when something was exported.
#
# Single awk pass on purpose. A `sed | awk | head` pipeline makes the upstream
# stage take SIGPIPE once the downstream one exits early, and under `set -o
# pipefail` that aborts the whole script with no diagnostic. nemo colours its
# report via the `colored` crate, so the escape sequences are stripped here
# rather than upstream.
report_ms() { # <logfile> <label>
  awk -v label="$2" '
    { gsub(/\033\[[0-9;]*[A-Za-z]/, "") }
    index($0, label) {
      for (i = 1; i <= NF; i++)
        if ($i ~ /ms$/) { sub(/ms$/, "", $i); print $i; exit }
    }
  ' "$1"
}

# Run nmo and publish its timings in R_WALL / R_IMPORT / R_REASONING /
# R_EXPORT / R_TRACE.
#
# Deliberately assigns globals rather than echoing for command substitution: a
# failing `exit` inside $( ) only kills the subshell, so a crashed nmo would be
# silently reported as an empty measurement.
#
# Tracing is NOT wrapped in a TimedCode block anywhere in the engine -- the timed
# blocks are only Reading & Preprocessing, Reasoning, Reasoning/Execution,
# Reasoning/Rules and Output & Final Materialization. So --report cannot isolate
# it, and trace time has to be taken as the residual:
#
#     trace_ms = wall - import - reasoning - export
#
# That residual also absorbs process startup, rule parsing and JSON writing, so
# it slightly overstates tracing. It overstates it by the same near-constant
# amount in every run, which is what matters for a ratio.
run_nmo() { # <logfile> <args...>
  local log="$1"; shift
  local t0 t1
  t0=$(date +%s%N)
  if ! "$BIN" "$@" >"$log" 2>&1; then
    echo "error: nmo failed; see $log" >&2
    tail -20 "$log" >&2
    exit 1
  fi
  t1=$(date +%s%N)
  R_WALL=$(( (t1 - t0) / 1000000 ))
  R_IMPORT=$(report_ms "$log" "Data import:");  R_IMPORT=${R_IMPORT:-0}
  R_REASONING=$(report_ms "$log" "Reasoning:"); R_REASONING=${R_REASONING:-0}
  R_EXPORT=$(report_ms "$log" "Data export:");  R_EXPORT=${R_EXPORT:-0}
  R_TRACE=$(( R_WALL - R_IMPORT - R_REASONING - R_EXPORT ))
  # An `if`, not `(( ... )) && ...`: a false arithmetic test exits non-zero, and
  # under `set -e` that would abort the run whenever the residual was positive.
  if (( R_TRACE < 0 )); then
    R_TRACE=0
  fi
}

# ---------------------------------------------------------------------------
# nemo rejects more than one rule file ("multiple rule files are currently
# unsupported"), so the verbatim program and the export overlay are concatenated
# into one program here. sssom-chain.rls itself stays byte-identical to the
# program reported in #763; the only addition is facts-export.rls, which contains
# a single @export directive and no rules.
PROGRAM="$WORK/program.rls"
cat "$HERE/sssom-chain.rls" "$HERE/facts-export.rls" >"$PROGRAM"

# --param values are parsed with GroundTerm::parse
# (nemo/src/execution/execution_parameters.rs:61), so a path has to be a quoted
# nemo string literal. Bare text is rejected as "invalid parameter".
PARAMS=(
  --param "importfile=\"$(realpath "$DATA")\""
  --param "exportfile=\"inferred.ttl\""
  --param "factsfile=\"inferred.nt\""
)

echo "== Phase A: materialize the model and export the fact list =="
run_nmo "$WORK/phase-a.log" "$PROGRAM" "${PARAMS[@]}" \
  --export-dir "$WORK" --overwrite-results --report short
echo "   import ${R_IMPORT}ms  reasoning ${R_REASONING}ms  export ${R_EXPORT}ms  (wall ${R_WALL}ms)"

NT="$WORK/inferred.nt"
[[ -s "$NT" ]] || { echo "error: no facts exported to $NT" >&2; exit 1; }

# N-Triples order is trie order: predicate_rows iterates the combined trie in
# ColumnOrder::default(), so contiguous LINES are a contiguous subject-ID range.
# That is the entire basis for locality-ordered sharding -- do not sort this file.
ALL="$WORK/all.facts"
awk '
  NF >= 4 && substr($1,1,1) == "<" && substr($2,1,1) == "<" && substr($3,1,1) == "<" {
    printf "inferredMapping(%s,%s,%s)\n", $1, $2, $3; kept++; next
  }
  { skipped++ }
  END { if (skipped) printf "skipped %d non-IRI triple(s)\n", skipped > "/dev/stderr" }
' "$NT" >"$ALL"

if (( MAX_FACTS > 0 )); then
  head -n "$MAX_FACTS" "$ALL" >"$ALL.capped" && mv "$ALL.capped" "$ALL"
fi
TOTAL=$(wc -l <"$ALL")
echo "   $TOTAL facts to explain"
(( TOTAL >= 8 )) || { echo "error: too few facts ($TOTAL) for a meaningful sweep" >&2; exit 1; }

# --trace-input-file splits its content on ';' only (nemo-cli/src/tracing.rs:23),
# never on newlines. So the newline-delimited file that `split` understands has to
# be rewritten as a ';'-joined one for nemo, with no trailing separator.
join_facts() { awk 'NR > 1 { printf ";" } { printf "%s", $0 }' "$1" >"$2"; }

trace_run() { # <tag> <facts-file> <ordering> <k> <shard-index>
  local tag="$1" facts="$2" ordering="$3" k="$4" idx="$5"
  local joined="$WORK/$tag.in" out="$WORK/$tag.json" log="$WORK/$tag.log"
  join_facts "$facts" "$joined"
  # --export none suppresses both export directives. Every trace run re-does
  # inference anyway (that is the cost this design removes), but there is no
  # reason to also re-serialize the whole inferred set each time -- that would
  # inflate the residual that stands in for trace time.
  run_nmo "$log" "$PROGRAM" "${PARAMS[@]}" \
    --export none --export-dir "$WORK" --overwrite-results --report short \
    --trace-input-file "$joined" --trace-output "$out"
  local inf conc n
  inf=$(jq '.inferences | length' "$out")
  conc=$(jq '.finalConclusion | length' "$out")
  n=$(wc -l <"$facts")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$ordering" "$k" "$idx" "$n" \
    "$R_WALL" "$R_IMPORT" "$R_REASONING" "$R_EXPORT" "$R_TRACE" "$inf" "$conc" >>"$RESULTS"
  echo "   $tag: $n facts, trace ${R_TRACE}ms, $inf inferences"
}

echo
echo "== Phase B: baseline, whole set in one run =="
trace_run "baseline" "$ALL" "baseline" 1 0

for ordering in $ORDERINGS; do
  src="$ALL"
  if [[ "$ordering" == "rnd" ]]; then
    src="$WORK/all.shuf"
    shuf "$ALL" >"$src"
  fi
  for k in $KS; do
    (( k <= TOTAL )) || { echo "   skipping K=$k (only $TOTAL facts)"; continue; }
    echo
    echo "== Phase C: $ordering, K=$k =="
    prefix="$WORK/shard.$ordering.$k."
    rm -f "$prefix"*
    split -n "l/$k" -d --suffix-length=3 "$src" "$prefix"
    i=0
    for shard in "$prefix"*; do
      [[ -s "$shard" ]] || continue
      trace_run "$ordering-$k-$i" "$shard" "$ordering" "$k" "$i"
      i=$((i + 1))
    done
  done
done

echo
python3 "$HERE/report.py" "$RESULTS"
echo
echo "raw rows: $RESULTS"
echo "logs:     $WORK"
