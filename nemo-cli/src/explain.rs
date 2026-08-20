//! Sharded explanation across worker processes.
//!
//! # Why this is worth doing
//!
//! Explaining a large fact set is the expensive half of the workflow that
//! motivated the model store: deriving the facts is comparatively quick, and
//! explaining them can take orders of magnitude longer. The work is also
//! embarrassingly parallel — each fact's derivation is independent — so the
//! obstacle was never the parallelism. It was that every process had to derive
//! the model before it could explain anything, so *n* processes did *n* times the
//! inference.
//!
//! With `--cache-dir`, a worker restores the model instead. That is why
//! `--explain` requires a cache directory: without one, sharding would reintroduce
//! the cost it exists to remove.
//!
//! # Why a row range is the unit of work
//!
//! A worker is told `(predicate, start, count)` and nothing else. `predicate_rows`
//! iterates the combined trie in `ColumnOrder::default()`, so a contiguous range
//! of rows is a contiguous range of leading values. That matters because the
//! tracer carries one memo across a batch: neighbouring facts share derivations,
//! so a contiguous shard keeps the sharing that a scattered one would break.
//!
//! Measured on 20,000 SSSOM mappings across 13,123 facts, contiguous shards
//! duplicate essentially nothing — summed shard inference counts came to 1.00×
//! the single-run total at 8 shards and 1.01× at 16 — where randomly assigned
//! shards reached 1.46× and 1.58×. That measurement is also why shards are plain
//! trace documents here: with nothing to deduplicate, a relational trace store
//! would be solving a problem this design does not have.
//!
//! # One binary
//!
//! The controller spawns *itself*. Model files are interpreted against the format
//! versions of the build that wrote them, so a controller and worker from
//! different builds could disagree about what a stored id means. Re-executing the
//! current executable makes that impossible rather than merely unlikely.

use std::{
    fs::{File, create_dir_all},
    path::{Path, PathBuf},
    process::Command,
};

use nemo::{execution::DefaultExecutionEngine, rule_model::components::tag::Tag};

use crate::{cli::CliApp, error::CliError};

/// Outcome of one worker.
#[derive(Debug)]
struct ShardOutcome {
    /// Index of the shard, matching its output file.
    index: usize,
    /// Rows it was asked to explain.
    range: (usize, usize),
    /// Where it wrote its trace.
    output: PathBuf,
    /// Whether it exited successfully.
    succeeded: bool,
}

/// Split `total` rows into at most `shards` contiguous ranges.
///
/// The remainder is spread one row at a time across the leading shards rather
/// than piled onto the last, so no shard is systematically larger. Returns fewer
/// ranges than requested when there are fewer rows than shards, and an empty
/// vector for no rows at all.
fn split_rows(total: usize, shards: usize) -> Vec<(usize, usize)> {
    if total == 0 || shards == 0 {
        return Vec::new();
    }

    let shards = shards.min(total);
    let base = total / shards;
    let remainder = total % shards;

    let mut ranges = Vec::with_capacity(shards);
    let mut start = 0;

    for index in 0..shards {
        let count = base + usize::from(index < remainder);
        ranges.push((start, count));
        start += count;
    }

    ranges
}

/// Explain the rows this process was assigned, writing one trace document.
///
/// The worker half. Reached only when the controller passes `--explain-range`.
pub(crate) async fn run_worker(
    cli: &CliApp,
    engine: &mut DefaultExecutionEngine,
    predicate: &str,
    range: (usize, usize),
) -> Result<(), CliError> {
    let (start, count) = range;
    let tag = Tag::new(predicate.to_string());

    let (trace, handles) = engine.trace_predicate_range(&tag, start, count).await?;

    let Some(output) = &cli.tracing.explain_output else {
        // Without a destination there is nothing useful to do with a shard: its
        // value is the document, and printing trees from many processes into one
        // terminal would interleave them.
        return Err(CliError::MissingExplainOutput);
    };

    let filename = output.to_string_lossy().to_string();
    let mut file = File::create(output)?;

    if serde_json::to_writer(&mut file, &trace.json(&handles)).is_err() {
        return Err(CliError::SerializationError { filename });
    }

    Ok(())
}

/// Shard the predicate's facts across workers and wait for them.
///
/// The controller half. Returns the number of facts distributed.
pub(crate) fn run_controller(
    cli: &CliApp,
    program: &Path,
    predicate: &str,
    total_rows: usize,
    output_directory: &Path,
) -> Result<usize, CliError> {
    let ranges = split_rows(total_rows, cli.tracing.explain_workers);

    if ranges.is_empty() {
        eprintln!("note: {predicate} has no facts to explain");
        return Ok(0);
    }

    create_dir_all(output_directory)?;

    // Re-executing this exact binary, not a name resolved from PATH: a worker from
    // a different build could read the stored model's ids differently.
    let executable = std::env::current_exe()?;

    eprintln!(
        "note: explaining {total_rows} facts of {predicate} across {} worker(s)",
        ranges.len()
    );

    let mut children = Vec::with_capacity(ranges.len());

    for (index, (start, count)) in ranges.iter().copied().enumerate() {
        let output = output_directory.join(format!("shard-{index:04}.json"));
        let mut command = Command::new(&executable);

        // `program` is passed in rather than read from `cli.rules`, which `run`
        // has already drained by the time a controller gets here. Reading it from
        // there produced workers invoked with no rule file at all.
        command
            .arg(program)
            .arg("--explain")
            .arg(predicate)
            .arg("--explain-range")
            .arg(format!("{start}:{count}"))
            .arg("--explain-output")
            .arg(&output)
            // Every worker restores from the cache the controller has already
            // populated, which is the whole reason sharding pays off.
            .arg("--cache-dir")
            .arg(
                cli.cache_dir
                    .as_ref()
                    .expect("--explain requires --cache-dir"),
            )
            .arg("--export")
            .arg("none")
            .arg("--report")
            .arg("none");

        if let Some(directory) = &cli.import_directory {
            command.arg("--import-dir").arg(directory);
        }

        for parameter in &cli.parameters {
            command
                .arg("--param")
                .arg(format!("{}={}", parameter.key, parameter.value));
        }

        children.push((index, (start, count), output, command.spawn()?));
    }

    // Spawned first, waited on afterwards: waiting inside the loop above would
    // run them one at a time.
    let mut outcomes = Vec::with_capacity(children.len());
    for (index, range, output, mut child) in children {
        let status = child.wait()?;
        outcomes.push(ShardOutcome {
            index,
            range,
            output,
            succeeded: status.success(),
        });
    }

    let failed: Vec<&ShardOutcome> = outcomes.iter().filter(|shard| !shard.succeeded).collect();

    for shard in &failed {
        eprintln!(
            "error: shard {} (rows {}..{}) failed; {} is missing or incomplete",
            shard.index,
            shard.range.0,
            shard.range.0 + shard.range.1,
            shard.output.display()
        );
    }

    if !failed.is_empty() {
        // Reported rather than partially accepted: a caller cannot tell a complete
        // set of explanations from one missing a shard by looking at the files.
        return Err(CliError::ExplainShardsFailed {
            failed: failed.len(),
            total: outcomes.len(),
        });
    }

    eprintln!(
        "note: wrote {} shard(s) to {}",
        outcomes.len(),
        output_directory.display()
    );

    Ok(total_rows)
}

#[cfg(test)]
mod test {
    use super::split_rows;

    #[test]
    fn splits_evenly_when_it_divides() {
        assert_eq!(split_rows(8, 4), vec![(0, 2), (2, 2), (4, 2), (6, 2)]);
    }

    #[test]
    fn spreads_the_remainder_rather_than_piling_it_on_one_shard() {
        // 10 rows over 4 shards: two shards of 3, two of 2. Appending the
        // remainder to the last shard instead would make it 50% larger than its
        // peers, and the run is only as fast as its slowest shard.
        assert_eq!(split_rows(10, 4), vec![(0, 3), (3, 3), (6, 2), (8, 2)]);
    }

    #[test]
    fn ranges_are_contiguous_and_cover_everything() {
        for total in [1usize, 2, 7, 100, 1013] {
            for shards in [1usize, 2, 3, 8, 16, 64] {
                let ranges = split_rows(total, shards);

                let mut next = 0;
                for (start, count) in &ranges {
                    assert_eq!(
                        *start, next,
                        "gap or overlap at {start} for {total}/{shards}"
                    );
                    assert!(*count > 0, "empty shard for {total}/{shards}");
                    next += count;
                }

                assert_eq!(next, total, "shards must cover every row");
            }
        }
    }

    #[test]
    fn never_produces_more_shards_than_rows() {
        // Otherwise a worker would be spawned with nothing to do, paying process
        // startup and a model restore to explain zero facts.
        assert_eq!(split_rows(3, 16), vec![(0, 1), (1, 1), (2, 1)]);
    }

    #[test]
    fn nothing_to_do_yields_no_shards() {
        assert!(split_rows(0, 8).is_empty());
        assert!(split_rows(100, 0).is_empty());
    }
}
