from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import lean_interact as li
import minif2f

import delphyne as dp
import delphyne.stdlib.experiments.learning_script as dl

TRAINING_THEOREMS = minif2f.load_minif2f("valid")
TESTING_THEOREMS = minif2f.load_minif2f("test")
INIT_COMMANDS = ["import MiniF2F.Minif2fImport"]


@dataclass(kw_only=True)
class ExperimentSettings:
    training_problems: list[str]
    testing_problems: list[str]
    output: str
    max_dollars_per_training_problem: float
    max_requests_per_training_problem: int
    max_dollars_per_testing_problem: float
    max_requests_per_testing_problem: int


def load_problem(problem_kind: dl.ProblemKind, problem_name: str):
    if problem_kind == "test":
        return TESTING_THEOREMS[problem_name]
    else:
        return TRAINING_THEOREMS[problem_name]


def solve_problem(
    settings: ExperimentSettings,
    problem_kind: dl.ProblemKind,
    problem_name: str,
):
    if problem_kind == "test":
        max_dollars_per_problem = settings.max_dollars_per_testing_problem
        max_requests_per_problem = settings.max_requests_per_testing_problem
    else:
        max_dollars_per_problem = settings.max_dollars_per_training_problem
        max_requests_per_problem = settings.max_requests_per_training_problem
    return dp.RunStrategyArgs(
        strategy="prove_theorem",
        args={"theorem": load_problem(problem_kind, problem_name)},
        policy="ProveTheoremPolicy",
        policy_args={},  # We use default arguments
        budget={
            dp.DOLLAR_PRICE: max_dollars_per_problem,
            dp.NUM_REQUESTS: max_requests_per_problem,
        },
    )


def generate_tips(feedback: dl.SerializedQueryFeedback) -> dp.RunStrategyArgs:
    return dp.RunStrategyArgs(
        strategy="generate_tips",
        args={"feedback": feedback},
        policy="generate_tips_policy",
        policy_args={"model_name": "gpt-5", "effort": "low"},
        budget={dp.NUM_REQUESTS: 2},  # 1 should be enough in theory
    )


def summarize_tips(query_type: str, tips: Sequence[dl.Tip]):
    return dp.RunStrategyArgs(
        strategy="summarize_tips",
        args={"query_type": query_type, "tips": tips},
        policy="generate_tips_policy",
        policy_args={"model_name": "gpt-5", "effort": "medium"},
        budget={dp.NUM_REQUESTS: 2},  # 1 should be enough in theory
    )


def global_init() -> li.LeanREPLConfig:
    from leandra.tools import lean_server_config

    repo_path = minif2f.repo_path()
    # We setup a memory limit so that 64 workers can work together on a
    # 200GB machine.
    # mem = int(200_000 / 64)
    # Update: Mathlib is memory hungry so having 8GB is good.
    mem = 8000
    return lean_server_config(repo_path=repo_path, memory_hard_limit_mb=mem)


def worker_init(config: li.LeanREPLConfig):
    from leandra.tools import init_global_lean_server_with_config

    init_global_lean_server_with_config(config, INIT_COMMANDS)


workers_setup = dp.WorkersSetup(common=global_init, per_worker=worker_init)


def make_experiment(settings: ExperimentSettings):
    context = dp.workspace_execution_context(__file__)
    # We want custom initialization compatible with multiprocessing
    context = replace(context, init=())
    output_directory = Path(__file__).absolute().parent / settings.output
    return dl.LearningExperiment(
        context=context,
        training_problems=settings.training_problems,
        testing_problems=settings.testing_problems,
        directory=output_directory,
        solve_problem=partial(solve_problem, settings),
        generate_tips=generate_tips,
        summarize_tips=summarize_tips,
        # We generate 100 tips of each kind at much at each iteration
        feedback_filters={
            "SketchProof": dl.FeedbackFilteringSettings(
                max_per_problem=(1, 1),
                max_total=(100, 200),
                max_wrong_answers=8,
                ensure_one_good_answer_exactly=True,
            ),
            "ProveGoal": dl.FeedbackFilteringSettings(
                max_per_problem=(10, 10),
                max_total=(100, 200),
                max_wrong_answers=8,
                ensure_one_good_answer_exactly=True,
            ),
        },
        enabled_feedback_nodes=["first", "subproof"],
        workers_setup=workers_setup,
    )
