# Step 0 — memo-fragmentation measurement

Measures how much the tracer's memo fragments when a fact set is split across
independent runs. This is the falsification step for the parallel-explanation
design in issue #1; it deliberately requires **no engine changes**.

## Why

`trace_ground_facts` threads one `ExecutionTrace` through every fact in a batch
(`nemo/src/execution/execution_engine/tracing/simple.rs:253-273`), and
`trace_recursive` returns immediately on an already-known fact (`:43-47`). So
explaining fact #2 reuses everything computed for fact #1 — and splitting the
fact set across N workers makes each of them re-derive the shared substructure.

If that penalty approaches N even with locality-ordered shards, the whole
parallel-explanation design fails. Better to learn that for the price of a few
benchmark runs than after building a memory-mapped store.

## The metric

`list_of_inferences` deduplicates within a run
(`nemo/src/execution/tracing/trace.rs:369-411`), so the ratio

```
F = sum(inferences across shards) / inferences(single whole-set run)
```

**is** the fragmentation factor, exactly, with no instrumentation.

Two secondary figures come along: `work` (summed shard trace time over baseline
trace time) and `speedup` (baseline trace time over the slowest shard — the best
case achievable at that K).

## Running it

```bash
cargo build --release -p nemo-cli

# self-test on a synthetic fixture (see the caveat below)
python3 smoke-corpus.py smoke/mappings.nt
./run.sh -d smoke/mappings.nt -k "2 4 8"

# the real thing
./fetch-corpus.sh                                  # or --from-file your.sssom.tsv
./run.sh -d corpus/mappings-20000.nt -k "2 4 8 16"
```

Useful flags: `-m N` traces only the first N facts (keeps locality, so it is a
valid way to shrink a first pass); `-O loc` skips the random-ordering arm; `-b`
points at a different `nmo`.

## Pre-registered decision rule

Committed before any numbers existed (issue #1, decision 16). Applied at K=8 on
locality-ordered shards:

| F | Action |
|---|---|
| **< 1.5×** | Proceed exactly as designed. |
| **1.5–3×** | Proceed, but pull the shared explained-set into v1 scope rather than deferring it. |
| **> 3×** | The farm thesis has failed. Pivot to shared-memo, or to partitioned inference. |

`report.py` applies this itself and refuses to give a verdict if K=8 was not
measured, so the rule cannot be quietly reinterpreted after the fact.

The locality-versus-random comparison tests a *different* claim: not whether the
farm is viable, but whether trie-range batching earns its complexity. If locality
does not clearly beat random, small random batches with a pull queue are simpler
and just as good.

## Files

| File | Role |
|---|---|
| `sssom-chain.rls` | The reporter's program from knowsys/nemo#763, **verbatim**. Do not edit. |
| `facts-export.rls` | One `@export` directive, no rules. Concatenated onto the above at runtime. |
| `fetch-corpus.sh` | Downloads a public SSSOM set and builds the corpus scales. |
| `sssom_to_nt.py` | Expands SSSOM CURIEs to full IRIs and writes N-Triples. |
| `run.sh` | The experiment driver. |
| `report.py` | Computes F, the secondary figures, and the verdict. |
| `smoke-corpus.py` | Synthetic fixture for testing the harness. |

## Things the harness has to work around

Each of these was found by running it, and each is a place where a naive
implementation would silently produce a wrong number.

- **Tracing is not timed.** No `TimedCode` block wraps it anywhere — the timed
  blocks are only `Reading & Preprocessing`, `Reasoning`, `Reasoning/Execution`,
  `Reasoning/Rules` and `Output & Final Materialization`. So `--report` cannot
  isolate the tracing phase, and trace time is taken as the residual
  `wall - import - reasoning - export`. That residual also absorbs process
  startup, rule parsing and JSON writing, so it overstates tracing — by roughly
  the same constant in every run, which is what a ratio needs. Adding a timing
  block around `handle_tracing` would make this direct and is a small, separable
  change; it is deliberately not done here so Step 0 stays engine-code-free.
- **Only one rule file is accepted** ("multiple rule files are currently
  unsupported"), so `run.sh` concatenates the verbatim program and the export
  overlay into one file at runtime. The checked-in program stays byte-identical.
- **`--param` values are nemo ground terms**, parsed with `GroundTerm::parse`
  (`nemo/src/execution/execution_parameters.rs:61`). Paths must be quoted string
  literals; bare text is rejected as `invalid parameter`.
- **`--trace-input-file` splits on `;` only**, never on newlines
  (`nemo-cli/src/tracing.rs:23`). The harness keeps a newline-delimited file so
  `split` works, and rewrites each shard as a `;`-joined file for nemo.
- **N-Triples, not Turtle, for the fact list.** N-Triples is exactly one
  `<s> <p> <o> .` per line; Turtle serializers may group statements by subject
  with `;` continuations, which would break line-oriented sharding.
- **Do not sort the exported fact list.** `predicate_rows` iterates the combined
  trie in `ColumnOrder::default()`, so the export is *already* in trie order, and
  contiguous lines are a contiguous subject-ID range. That ordering is the entire
  basis for locality-ordered sharding.
- **`--export none` on trace runs.** Every trace run re-does inference — that is
  the cost this design exists to remove — but re-serializing the whole inferred
  set each time would inflate the residual standing in for trace time.
- **CURIE expansion is mandatory.** The rules match full IRIs longhand, while
  SSSOM TSV stores CURIEs. If expansion fails, every chaining rule is dead and
  the experiment measures nothing. `report.py` fails loudly on a zero-inference
  baseline for exactly this reason.

## Caveat on the fixture

`smoke-corpus.py` emits independent chains, so contiguous ranges align perfectly
with connected components and locality scores `F = 1.00`. That is a property of
the fixture, not a result — real mapping graphs have hub terms and a giant
component. The informative half of a fixture run is the random arm: F should
climb with K. If it does not, the harness is not measuring what it claims to.
