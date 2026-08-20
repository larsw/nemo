/*!
  Binary for the CLI of nemo: nmo
*/

#![deny(
    missing_debug_implementations,
    missing_copy_implementations,
    trivial_casts,
    trivial_numeric_casts
)]
#![warn(
    missing_docs,
    unused_import_braces,
    unused_qualifications,
    unused_extern_crates,
    variant_size_differences
)]

pub mod cli;
pub mod error;
pub mod tracing;

use std::{io::Write, io::stdout};

use clap::Parser;
use colored::Colorize;

use cli::{CliApp, FactPrinting, Reporting};

use error::CliError;
use std::{collections::BTreeMap, path::PathBuf};

use nemo::{
    datavalues::AnyDataValue,
    error::Error,
    execution::{
        DefaultExecutionEngine, ExecutionEngine, execution_parameters::ExecutionParameters,
        planning::normalization::program::NormalizedProgram,
    },
    io::{ImportManager, resource_providers::ResourceProviders},
    meta::timing::{TimedCode, TimedDisplay},
    model_store::{CacheKey, ModelCache, ModelStore},
    rule_file::RuleFile,
    rule_model::components::{fact::Fact, tag::Tag, term::Term},
};
use tracing::{handle_tracing, tracing_requested};

fn print_facts_for_table<W: Write>(
    writer: &mut W,
    mut table: impl Iterator<Item = Vec<AnyDataValue>>,
    predicate: Tag,
) -> Result<(), Error> {
    table
        .try_for_each(|row| {
            writeln!(
                writer,
                "{}",
                Fact::new(predicate.clone(), row.into_iter().map(Term::ground))
            )
        })
        .map_err(Error::IO)
}

fn predicates_to_print_facts_for(
    print_facts_setting: FactPrinting,
    program: &NormalizedProgram,
) -> Vec<Tag> {
    match print_facts_setting {
        FactPrinting::None => Vec::new(),
        FactPrinting::Idb => program.derived_predicates().iter().cloned().collect(),
        FactPrinting::Edb => program.import_predicates().collect(),
        FactPrinting::All => program.all_predicates().collect(),
    }
}

/// Prints short summary message.
fn print_finished_message(new_facts: usize, saving: bool, traced: bool) {
    let overall_time = TimedCode::instance().total_system_time().as_millis();
    let reading_time = TimedCode::instance()
        .sub("Reading & Preprocessing")
        .total_system_time()
        .as_millis();
    let loading_time = TimedCode::instance()
        .sub("Reasoning/Execution/Load Table")
        .total_system_time()
        .as_millis();
    let execution_time = TimedCode::instance()
        .sub("Reasoning")
        .total_system_time()
        .as_millis();

    // NOTE: for some reason the subtraction produced an overflow for me once when running the tests; so better safe than sorry now :)
    let loading_preprocessing = reading_time.saturating_add(loading_time);
    let reasoning_time = execution_time.saturating_sub(loading_time);

    let writing_time = if saving {
        TimedCode::instance()
            .sub("Output & Final Materialization")
            .total_system_time()
            .as_millis()
    } else {
        0
    };

    let tracing_time = if traced {
        TimedCode::instance()
            .sub("Tracing")
            .total_system_time()
            .as_millis()
    } else {
        0
    };

    let max_string_len = [
        loading_preprocessing,
        reading_time,
        writing_time,
        tracing_time,
    ]
    .iter()
    .map(|t| t.to_string().len())
    .max()
    .expect("Vector is not empty")
        + 2; // for the unit ms

    println!(
        "Reasoning completed in {}{}. Derived {} facts.",
        overall_time.to_string().green().bold(),
        "ms".green().bold(),
        new_facts.to_string().green().bold(),
    );

    println!(
        "   {0: <14} {1:>max_string_len$}ms",
        "Data import:", loading_preprocessing
    );
    println!(
        "   {0: <14} {1:>max_string_len$}ms",
        "Reasoning:", reasoning_time
    );

    if saving {
        println!(
            "   {0: <14} {1:>max_string_len$}ms",
            "Data export:", writing_time
        );
    }

    if traced {
        println!(
            "   {0: <14} {1:>max_string_len$}ms",
            "Tracing:", tracing_time
        );
    }
}

/// Prints detailed timing information.
fn print_timing_details() {
    println!(
        "\nTiming report:\n\n{}",
        TimedCode::instance().create_tree_string(
            "nemo",
            &[
                TimedDisplay::default(),
                TimedDisplay::default(),
                TimedDisplay::new(nemo::meta::timing::TimedSorting::LongestThreadTime, 0)
            ]
        )
    );
}

/// Prints detailed memory information.
fn print_memory_details(engine: &DefaultExecutionEngine) {
    println!("\nMemory report:\n\n{}", engine.memory_usage());
}

/// What to do about the model cache for this run.
enum CachePlan {
    /// A stored model applies; restore it instead of reasoning.
    Hit {
        /// The matching store.
        store: ModelStore,
    },
    /// Nothing applies; reason, then store the result at `target`.
    Miss {
        /// Free path to write the new store to.
        target: PathBuf,
        /// Key to record with it.
        key: CacheKey,
    },
}

/// Decide what the cache can do for this run.
///
/// `Ok(None)` means the cache is not in play: either no directory was given, or
/// the program cannot be keyed reliably. `ExecutionEngine::cache_key` declines to
/// key a program importing anything other than a local file, since an HTTP
/// resource has no cheap fingerprint and treating it as unchanged would serve a
/// stale model.
fn resolve_cache<Strategy: nemo::execution::selection_strategy::strategy::RuleSelectionStrategy>(
    cli: &CliApp,
    engine: &ExecutionEngine<Strategy>,
    parameters: BTreeMap<String, String>,
) -> Result<Option<CachePlan>, CliError> {
    let Some(directory) = &cli.cache_dir else {
        return Ok(None);
    };

    let Some(key) = engine.cache_key(parameters) else {
        eprintln!(
            "note: not using the model cache -- this program imports a resource that cannot be \
             fingerprinted, so a stored model could not be shown to still apply"
        );
        return Ok(None);
    };

    let cache = ModelCache::open(directory).map_err(Error::from)?;

    match cache.lookup(&key) {
        Some(store) => {
            eprintln!("note: reusing the cached model, skipping inference");
            Ok(Some(CachePlan::Hit { store }))
        }
        None => Ok(Some(CachePlan::Miss {
            target: cache.reserve(),
            key,
        })),
    }
}

async fn run(mut cli: CliApp) -> Result<(), CliError> {
    TimedCode::instance().start();
    TimedCode::instance().sub("Reading & Preprocessing").start();

    log::info!("Parsing rules ...");

    if cli.rules.len() > 1 {
        return Err(CliError::MultipleFilesNotImplemented);
    }

    let program_path = cli.rules.pop().ok_or(CliError::NoInput)?;
    let program_file = RuleFile::load(program_path)?;

    let export_manager = cli.output.export_manager()?;
    let import_manager = ImportManager::new(ResourceProviders::with_base_path(
        cli.import_directory.clone(),
    ));

    let mut execution_parameters = ExecutionParameters::default();
    execution_parameters.set_export_parameters(cli.output.export_setting.into());
    execution_parameters.set_import_manager(import_manager);

    // Kept before draining, because the cache key has to record which bindings
    // the program was run with: identical program text under a different
    // $importfile reads different data.
    let parameter_bindings: BTreeMap<String, String> = cli
        .parameters
        .iter()
        .map(|parameter| (parameter.key.clone(), parameter.value.clone()))
        .collect();

    if let Err(parameter) = execution_parameters.set_global(
        cli.parameters
            .drain(..)
            .map(|parameter| (parameter.key, parameter.value)),
    ) {
        return Err(CliError::InvalidParameter { parameter });
    }

    let (mut engine, warnings) = ExecutionEngine::from_file(program_file, execution_parameters)
        .await?
        .into_pair();
    warnings.eprint(cli.disable_warnings)?;

    log::info!("Rules parsed");

    // Resolved before reasoning but after parsing: the key needs the normalized
    // program, and building the engine does not read any imported data -- sources
    // stay lazy until a table is first touched -- so a hit still skips the cost.
    let cache_plan = resolve_cache(&cli, &engine, parameter_bindings)?;

    for (predicate, handler) in engine.exports() {
        export_manager.validate(&predicate, &handler)?;
    }

    TimedCode::instance().sub("Reading & Preprocessing").stop();

    let mut cache_target = None;

    match cache_plan {
        Some(CachePlan::Hit { store }) => {
            TimedCode::instance().sub("Reasoning").start();
            log::info!("Restoring model from cache ...");
            let import_manager = ImportManager::new(ResourceProviders::with_base_path(
                cli.import_directory.clone(),
            ));
            engine = ExecutionEngine::from_model_store(
                engine.current_program_handle(),
                import_manager,
                &store,
            )
            .await?;
            log::info!("Model restored");
            TimedCode::instance().sub("Reasoning").stop();
        }
        plan => {
            if let Some(CachePlan::Miss { target, key }) = plan {
                cache_target = Some((target, key));
            }

            TimedCode::instance().sub("Reasoning").start();
            log::info!("Reasoning ... ");
            engine.execute().await?;
            log::info!("Reasoning done");
            TimedCode::instance().sub("Reasoning").stop();
        }
    }

    if let Some((target, key)) = cache_target {
        // After reasoning and before export, so an export failure does not
        // discard a model that was expensive to compute. A failure to store is
        // reported but not fatal: the results are already correct.
        log::info!("Storing model in cache ...");
        match engine.write_model_store_with_key(&target, key).await {
            Ok(path) => log::info!("Model stored at {}", path.display()),
            Err(error) => eprintln!("warning: could not store the model: {error}"),
        }
    }

    let mut stdout_used = false;

    if !export_manager.write_disabled() {
        TimedCode::instance()
            .sub("Output & Final Materialization")
            .start();
        log::info!("writing output");

        for (predicate, handler) in engine.exports() {
            stdout_used |= export_manager.export_table(
                &predicate,
                &handler,
                engine.predicate_rows(&predicate).await?,
            )?;
        }

        TimedCode::instance()
            .sub("Output & Final Materialization")
            .stop();
    }

    if cli.output.print_facts_setting.is_enabled() {
        TimedCode::instance().sub("Printing Facts").start();
        log::info!("Printing facts");

        let mut stdout = Box::new(stdout().lock());

        for predicate in
            predicates_to_print_facts_for(cli.output.print_facts_setting, engine.chase_program())
        {
            if let Some(table) = engine.predicate_rows(&predicate).await? {
                print_facts_for_table(&mut stdout, table, predicate)?;
            }
        }

        TimedCode::instance().sub("Printing Facts").stop();
    }

    // Tracing runs inside the timed region, before the summary is printed.
    //
    // It used to run after TimedCode::stop() and after the report, so
    // "Reasoning completed in Xms" excluded it entirely and no timed block
    // covered it -- even though explaining a materialized model can dominate
    // total runtime. Measured on 13,123 SSSOM facts: 52.5 s of tracing behind a
    // reported 145 ms.
    let tracing_requested = tracing_requested(&cli);
    if tracing_requested {
        TimedCode::instance().sub("Tracing").start();
        handle_tracing(&cli, &mut engine).await?;
        TimedCode::instance().sub("Tracing").stop();
    }

    TimedCode::instance().stop();

    let (print_summary, print_times, print_memory) = match cli.reporting {
        Reporting::All => (true, true, true),
        Reporting::Short => (true, false, false),
        Reporting::Time => (true, true, false),
        Reporting::Mem => (true, false, true),
        Reporting::None => (false, false, false),
        Reporting::Auto => (!stdout_used, false, false),
    };

    if print_summary {
        print_finished_message(
            engine.count_facts_in_memory_for_derived_predicates(),
            !export_manager.write_disabled(),
            tracing_requested,
        );
    }
    if print_times {
        print_timing_details();
    }
    if print_memory {
        print_memory_details(&engine);
    }

    Ok(())
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let cli = CliApp::parse();

    let disable_warnings = cli.disable_warnings;

    cli.logging.initialize_logging();
    log::info!("Version: {}", clap::crate_version!());
    log::debug!("Rule files: {:?}", cli.rules);

    if let Err(error) = run(cli).await {
        if let CliError::NemoError(Error::ProgramReport(report)) = error {
            let _ = report.eprint(disable_warnings);

            if report.contains_errors() {
                std::process::exit(1);
            }
        } else {
            log::error!("{} {error}", "error:".red().bold());
            std::process::exit(1);
        }
    }
}
