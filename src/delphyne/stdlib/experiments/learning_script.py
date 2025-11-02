"""
Script for running learning experiments with Delphyne.

!!! warning "Experimental"
    This is an experimental module that should evolve rapidly.

# Structure of learning experiment folders

```
<learning_experiment_name>/
  - embeddings.cache.h5
  - iterations/<iteration_i>
    - feedback.yaml:
    - train
    - test
    - analyze
    - summarize
    - data
      - <SketchProof>.tips.data.yaml
    - demos
      - <SketchProof>.demo.yaml
```
"""

import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, TypedDict

import delphyne.analysis as an
import delphyne.core_and_base as dp
import delphyne.stdlib.answer_loaders as al
import delphyne.stdlib.commands as cmd
import delphyne.stdlib.execution_contexts as ec
import delphyne.stdlib.experiments.experiment_launcher as el
import delphyne.stdlib.feedback_processing as fp
import delphyne.stdlib.hindsight_feedback as hf

GLOBAL_EMBEDDINGS_CACHE_FILE = ec.DEFAULT_GLOBAL_EMBEDDINGS_CACHE_FILE
EXPERIMENTS_RESULTS_SUMMARY_FILE = el.RESULTS_SUMMARY
ITERATIONS_DIR = "iterations"
FEEDBACK_FILE = "feedback.yaml"
TRAINING_EXP_DIR = "train"
TEST_EXP_DIR = "eval"
ANALYZE_EXP_DIR = "analyze"
SUMMARIZE_EXP_DIR = "summarize"


type ProblemKind = Literal["train", "test"]


class SolveProblemFn(Protocol):
    def __call__(
        self, problem_kind: ProblemKind, problem_name: str
    ) -> cmd.RunStrategyArgs: ...


class GenerateTipsFn(Protocol):
    def __call__(
        self, feedback: "SerializedQueryFeedback"
    ) -> cmd.RunStrategyArgs: ...


class SummarizeTipsFn(Protocol):
    def __call__(self, tips: "Sequence[Tip]") -> cmd.RunStrategyArgs: ...


@dataclass
class LearningExperiment:
    context: ec.ExecutionContext
    problems: Sequence[str]
    directory: Path
    solve_problem: SolveProblemFn
    generate_tips: GenerateTipsFn
    summarize_tips: SummarizeTipsFn
    workers_setup: el.WorkersSetup[Any] | None = None

    @property
    def configs_context(self) -> "LearningExperimentContext":
        return LearningExperimentContext(
            directory=self.directory,
            solve_problem=self.solve_problem,
            generate_tips=self.generate_tips,
            summarize_tips=self.summarize_tips,
        )

    ##### Obtaining Experiments #####

    pass


@dataclass
class LearningExperimentContext:
    directory: Path
    solve_problem: SolveProblemFn
    generate_tips: GenerateTipsFn
    summarize_tips: SummarizeTipsFn


@dataclass
class SolveProblemConfig(el.ExperimentConfig):
    """
    Configuration for solving a train or test problem.
    """

    kind: Literal["train", "eval"]
    problem: str
    iteration: int


@dataclass
class GenerateTipsConfig(el.ExperimentConfig):
    """
    Configuration for generating tips from feedback.
    """

    iteration: int
    index: int


@dataclass
class SummarizeTipsConfig(el.ExperimentConfig):
    """
    Configuration for summarizing tips.
    """

    iteration: int


#####
##### Tips
#####


class Tip(TypedDict):
    name: str
    content: str


#####
##### Gathering Feedback
#####


@dataclass
class QueryFeedback:
    """
    Aggregated feedback for a specific query.
    """

    problem_name: str
    query: dp.SerializedQuery
    good_answers: list[dp.Answer]
    bad_answers: list[tuple[dp.Answer, dp.Error]]


type SerializedQueryFeedback = Any
"""
JSON representation of a `QueryFeedback` object.
"""


def gather_feedback_per_query(
    resolver: an.IRefResolver,
    *,
    problem_name: str,
    roots: Sequence[fp.NodeId],
    filter_sources: fp.FeedbackFilter | None = None,
    filter_backprop_handlers: fp.FeedbackFilter | None = None,
) -> Sequence[QueryFeedback]:
    """
    Extract aggregated feedback per query from the given roots and
    sources (see `process_feedback`).
    """
    feedback: dict[fp.AnswerId, QueryFeedback] = {}
    for f in fp.process_feedback(
        resolver,
        roots=roots,
        filter_sources=filter_sources,
        filter_backprop_handlers=filter_backprop_handlers,
    ):
        if f.answer_id not in feedback:
            serialized = dp.SerializedQuery.make(f.query)
            feedback[f.answer_id] = QueryFeedback(
                problem_name=problem_name,
                query=serialized,
                good_answers=[],
                bad_answers=[],
            )
        if isinstance(f.feedback, hf.GoodValue):
            feedback[f.answer_id].good_answers.append(f.answer)
        elif isinstance(f.feedback, hf.BadValue):
            feedback[f.answer_id].bad_answers.append(
                (f.answer, f.feedback.error)
            )
        elif isinstance(f.feedback, hf.BetterValue):
            feedback[f.answer_id].good_answers.append(f.feedback.value)
        elif isinstance(f.feedback, hf.BadValueAlso):
            feedback[f.answer_id].bad_answers.append(
                (f.feedback.value, f.feedback.error)
            )
    return list(feedback.values())


def feedback_from_command_file(
    command_file: Path,
    *,
    object_loader: dp.ObjectLoader,
    problem_name: str,
    enabled_feedback_nodes: Sequence[str],
) -> Sequence[QueryFeedback]:
    """
    Load query feedback from a given command file.
    """
    trace_data = al.load_trace_data_from_command_file(command_file)
    resolver = trace_data.resolver(object_loader)
    filter: fp.FeedbackFilter = (
        lambda label, node_id: label in enabled_feedback_nodes
    )
    return list(
        gather_feedback_per_query(
            resolver,
            problem_name=problem_name,
            roots=[fp.NodeId(i) for i in trace_data.success_nodes[:1]],
            filter_sources=filter,
            filter_backprop_handlers=filter,
        )
    )


type RandomSelection = tuple[int, int]
"""
A (num_selected, among) pair where `num_selected` items are to be
randomly selected among `among` total items that maximize some
criterion.
"""


@dataclass
class FeedbackFilteringSettings:
    """
    Settings for filtering feedback.

    Attributes:
        max_per_problem: The number of feedback items to collect per
            problem, randomly selected among candidates with the maximal
            number of wrong answers (before the `max_wrong_answers`
            filter is applied).
        max_total: The total number of feedback items to collect,
            randomly selected among candidates with the maximal number
            after the `max_per_problem` filter is applied.
        max_wrong_answers: Maximum number of wrong answers per feedback
            item. If there are more, a random subset must be selected.
        ensure_one_good_answer_exactly: Whether to ensure that exactly one
            good answer is present for each feedback item. (A runtime
            error must be raised if this condition is violated.)
    """

    max_per_problem: RandomSelection
    max_total: RandomSelection
    max_wrong_answers: int
    ensure_one_good_answer_exactly: bool


type FeedbackFilteringSettingsDict = dict[str, FeedbackFilteringSettings]
"""
A mapping from query types to their feedback filtering settings.
"""


def filter_feedback(
    feedback: Sequence[QueryFeedback],
    *,
    rng: random.Random,
    settings: FeedbackFilteringSettingsDict,
) -> Sequence[QueryFeedback]:
    """
    Filter feedback according to the given settings.

    See `FeedbackFilteringSettings` for details.
    """

    # Group feedback items by query type (name)
    by_type: dict[str, list[QueryFeedback]] = {}
    for fb in feedback:
        qtype = fb.query.name
        if qtype not in settings:
            raise ValueError(f"No filter settings for query type '{qtype}'")
        by_type.setdefault(qtype, []).append(fb)

    def rank_wrong_count(item: QueryFeedback) -> int:
        # Ranking metric: number of wrong answers BEFORE any trimming
        return len(item.bad_answers)

    def select_top_then_sample(
        items: Sequence[QueryFeedback], sel: RandomSelection
    ) -> list[QueryFeedback]:
        num_selected, among = sel
        assert 0 <= num_selected <= among
        if num_selected == 0:
            return []
        if len(items) <= num_selected:
            return list(items)
        sorted_items = sorted(items, key=rank_wrong_count, reverse=True)
        pool = sorted_items[:among]
        return rng.sample(pool, num_selected)

    result: list[QueryFeedback] = []

    for qtype, items in by_type.items():
        cfg = settings[qtype]

        # 1) Per-problem selection
        by_problem: dict[str, list[QueryFeedback]] = {}
        for it in items:
            by_problem.setdefault(it.problem_name, []).append(it)
        per_problem_selected: list[QueryFeedback] = []
        for _, prob_items in by_problem.items():
            per_problem_selected.extend(
                select_top_then_sample(prob_items, cfg.max_per_problem)
            )
        if not per_problem_selected:
            continue

        # 2) Global selection across all problems for this query type
        globally_selected = select_top_then_sample(
            per_problem_selected, cfg.max_total
        )

        # 3) Apply per-item constraints and produce copies
        for it in globally_selected:
            # Enforce max_wrong_answers by subsampling wrong answers
            bad = it.bad_answers
            if len(bad) > cfg.max_wrong_answers:
                bad = rng.sample(bad, cfg.max_wrong_answers)
            # Optionally ensure exactly one good answer
            if (
                cfg.ensure_one_good_answer_exactly
                and len(it.good_answers) != 1
            ):
                raise RuntimeError(
                    "Expected exactly one good answer for query "
                    f"{it.query.name} in problem '{it.problem_name}', "
                    f"found {len(it.good_answers)}"
                )
            # Append a fresh QueryFeedback with trimmed bad answers
            result.append(
                QueryFeedback(
                    problem_name=it.problem_name,
                    query=it.query,
                    good_answers=it.good_answers,
                    bad_answers=bad,
                )
            )
    return result


#####
##### Meta strategies and prompts
#####


#####
##### Script
#####


#####
##### CLI
#####


class LearningExperimentCLI:
    pass
