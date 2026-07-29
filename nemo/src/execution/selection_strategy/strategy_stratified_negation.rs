//! Defines the execution strategy by which each rule is applied in the order it appears.

use std::collections::HashMap;

use petgraph::Directed;

use crate::{
    execution::planning::normalization::rule::NormalizedRule, rule_model::components::tag::Tag,
    util::labeled_graph::LabeledGraph,
};

use super::strategy::{RuleSelectionStrategy, SelectionStrategyError};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
enum EdgeLabel {
    Positive,
    Negative,
}

type NegationGraph = LabeledGraph<usize, EdgeLabel, Directed>;

/// Defines a strategy where rule are divided into different strata
/// which are executed in succession.
/// Entering a new statum implies that the table for every negated atom
/// will not get any new elements.
#[derive(Debug)]
pub struct StrategyStratifiedNegation<SubStrategy: RuleSelectionStrategy> {
    ordered_strata: Vec<Vec<usize>>,
    substrategies: Vec<SubStrategy>,

    current_stratum: usize,
}

impl<SubStrategy: RuleSelectionStrategy> StrategyStratifiedNegation<SubStrategy> {
    fn build_graph(rules: &Vec<&NormalizedRule>) -> NegationGraph {
        let mut predicate_to_rules_body_positive = HashMap::<Tag, Vec<usize>>::new();
        let mut predicate_to_rules_body_negative = HashMap::<Tag, Vec<usize>>::new();
        let mut predicate_to_rules_head = HashMap::<Tag, Vec<usize>>::new();

        let rule_count = rules.len();

        for (rule_index, rule) in rules.iter().enumerate() {
            for (body_predicate, _) in rule.predicates_positive() {
                let indices = if rule.contains_aggregates() {
                    // An aggregate in a head means that the head predicates need to be in a higher stratum than the body predicates
                    // This is the same as when all body literals are negative
                    // Therefore, we can easily compute strata for aggregates by acting if all body atoms in the rule were negated
                    predicate_to_rules_body_negative
                        .entry(body_predicate)
                        .or_default()
                } else {
                    // No aggregates in the rule
                    predicate_to_rules_body_positive
                        .entry(body_predicate)
                        .or_default()
                };

                indices.push(rule_index);
            }

            for (body_predicate, _) in rule.predicates_negative() {
                let indices = predicate_to_rules_body_negative
                    .entry(body_predicate)
                    .or_default();

                indices.push(rule_index);
            }

            for (head_predicate, _) in rule.predicates_head() {
                let indices = predicate_to_rules_head.entry(head_predicate).or_default();

                indices.push(rule_index);
            }
        }

        let mut graph = NegationGraph::default();

        for rule_index in 0..rule_count {
            graph.add_node(rule_index);
        }

        for (head_predicate, head_rules) in predicate_to_rules_head {
            if let Some(body_rules) = predicate_to_rules_body_positive.get(&head_predicate) {
                for &head_index in &head_rules {
                    for &body_index in body_rules {
                        graph.add_edge(head_index, body_index, EdgeLabel::Positive);
                    }
                }
            }

            if let Some(body_rules) = predicate_to_rules_body_negative.get(&head_predicate) {
                for &head_index in &head_rules {
                    for &body_index in body_rules {
                        graph.add_edge(head_index, body_index, EdgeLabel::Negative);
                    }
                }
            }
        }

        graph
    }
}

impl<SubStrategy: RuleSelectionStrategy> RuleSelectionStrategy
    for StrategyStratifiedNegation<SubStrategy>
{
    /// Create new [StrategyStratifiedNegation].
    fn new(rules: Vec<&NormalizedRule>) -> Result<Self, SelectionStrategyError> {
        let graph = Self::build_graph(&rules);

        if let Some(mut strata) = graph.stratify(&[EdgeLabel::Negative]) {
            let mut substrategies = Vec::new();

            for stratum in &mut strata {
                stratum.sort();

                let sub_rules = stratum.iter().map(|i| rules[*i]).collect::<Vec<_>>();

                substrategies.push(SubStrategy::new(sub_rules)?);
            }

            for stratum in &mut strata {
                stratum.sort();
            }

            if strata.len() > 1 {
                log::info!("Stratified program: {strata:?}")
            }

            Ok(Self {
                ordered_strata: strata,
                substrategies,
                current_stratum: 0,
            })
        } else {
            Err(SelectionStrategyError::NonStratifiedProgram)
        }
    }

    fn next_rule(&mut self, mut new_derivations: Option<bool>) -> Option<usize> {
        while self.current_stratum < self.ordered_strata.len() {
            if let Some(substrategy_next_rule) =
                self.substrategies[self.current_stratum].next_rule(new_derivations)
            {
                return Some(self.ordered_strata[self.current_stratum][substrategy_next_rule]);
            } else {
                self.current_stratum += 1;
                new_derivations = None;
            }
        }

        None
    }
}

#[cfg(test)]
mod test {
    use std::collections::BTreeSet;

    use crate::{
        execution::{
            execution_parameters::ExecutionParameters,
            planning::normalization::program::NormalizedProgram,
            selection_strategy::{
                strategy::{RuleSelectionStrategy, SelectionStrategyError},
                strategy_round_robin::StrategyRoundRobin,
            },
        },
        rule_file::RuleFile,
        rule_model::{
            pipeline::transformations::default::TransformationDefault,
            programs::handle::ProgramHandle,
        },
    };

    use super::{EdgeLabel, StrategyStratifiedNegation};

    type Strategy = StrategyStratifiedNegation<StrategyRoundRobin>;

    /// Parse and normalize a program
    fn normalize(program: &str) -> NormalizedProgram {
        let handle =
            ProgramHandle::from_file(&RuleFile::new(program.to_string(), String::default()))
                .expect("program parses")
                .into_object();
        let parameters = ExecutionParameters::default();
        let handle = handle
            .transform(TransformationDefault::new(&parameters))
            .expect("program is valid");

        NormalizedProgram::normalize_program(&handle)
    }

    /// Return the dependency graph built for `program`
    /// as a set of `(from_rule, to_rule, label)` triples.
    fn edges(program: &str) -> BTreeSet<(usize, usize, EdgeLabel)> {
        let normalized = normalize(program);
        let rules = normalized.rules().iter().collect::<Vec<_>>();
        let graph = Strategy::build_graph(&rules);
        let graph = graph.graph();

        graph
            .edge_indices()
            .map(|edge| {
                let (from, to) = graph
                    .edge_endpoints(edge)
                    .expect("edge index comes from the graph");

                (
                    *graph.node_weight(from).expect("node index is valid"),
                    *graph.node_weight(to).expect("node index is valid"),
                    *graph.edge_weight(edge).expect("edge index is valid"),
                )
            })
            .collect()
    }

    /// Return the strata computed for `program`, or `None` if it is not stratifiable.
    fn strata(program: &str) -> Option<Vec<Vec<usize>>> {
        let normalized = normalize(program);
        let rules = normalized.rules().iter().collect::<Vec<_>>();

        match Strategy::new(rules) {
            Ok(strategy) => Some(strategy.ordered_strata),
            Err(SelectionStrategyError::NonStratifiedProgram) => None,
        }
    }

    /// Drive the strategy until it reports that no rule is left,
    /// always claiming that the last rule application produced no new derivations.
    fn rule_order(program: &str) -> Vec<usize> {
        let normalized = normalize(program);
        let rules = normalized.rules().iter().collect::<Vec<_>>();
        let mut strategy = Strategy::new(rules).expect("program is stratifiable");

        let mut result = Vec::new();
        let mut derivations = None;

        while let Some(rule) = strategy.next_rule(derivations) {
            result.push(rule);
            derivations = Some(false);
        }

        result
    }

    const PROGRAM: &str = "
        edge(1, 2) . edge(2, 3) . edge(3, 1) .

        a(?x) :- total(?x) .
        total(#count(?x)) :- acyclic(?x) .

        b(?x) :- a(?x) .
        a(?x) :- b(?x) .

        acyclic(?x) :- source(?x), ~cyclic(?x) .

        source(?x), sink(?y) :- reachable(?x, ?y) .

        cyclic(?x) :- reachable(?x, ?y), reachable(?y, ?x) .

        reachable(?x, ?z) :- reachable(?x, ?y), edge(?y, ?z) .

        reachable(?x, ?y) :- edge(?x, ?y) .
    ";

    #[test]
    fn dependency_graph() {
        assert_eq!(
            edges(PROGRAM),
            BTreeSet::from([
                // rule 1 derives `total`, which rule 0 consumes
                (1, 0, EdgeLabel::Positive),
                // the aggregate of rule 1 turns its body into a negative dependency
                (4, 1, EdgeLabel::Negative),
                // rules 0 and 3 derive `a`, which rule 2 consumes
                (0, 2, EdgeLabel::Positive),
                (3, 2, EdgeLabel::Positive),
                // rule 2 derives `b`, which rule 3 consumes
                (2, 3, EdgeLabel::Positive),
                // rule 5 derives `source`, rule 6 derives the negated `cyclic`
                (5, 4, EdgeLabel::Positive),
                (6, 4, EdgeLabel::Negative),
                // rules 7 and 8 derive `reachable`, which rules 5, 6 and 7 consume;
                // the two body atoms of rule 6 do not make its dependencies count twice
                (7, 5, EdgeLabel::Positive),
                (7, 6, EdgeLabel::Positive),
                (7, 7, EdgeLabel::Positive),
                (8, 5, EdgeLabel::Positive),
                (8, 6, EdgeLabel::Positive),
                (8, 7, EdgeLabel::Positive),
            ])
        );
    }

    #[test]
    fn stratification() {
        assert_eq!(
            strata(PROGRAM),
            Some(vec![vec![5, 6, 7, 8], vec![4], vec![0, 1, 2, 3]])
        );
    }

    #[test]
    fn rule_selection() {
        assert_eq!(rule_order(PROGRAM), vec![5, 6, 7, 8, 4, 0, 1, 2, 3]);
    }

    #[test]
    fn not_stratified() {
        // `c` is derived from `a` but also negated in the derivation of `a`.
        assert_eq!(
            strata(
                "a(?x) :- b(?x), ~c(?x) .
                 c(?x) :- a(?x) ."
            ),
            None
        );

        // A rule negating its own head predicate.
        assert_eq!(strata("a(?x) :- b(?x), ~a(?x) ."), None);
    }
}
