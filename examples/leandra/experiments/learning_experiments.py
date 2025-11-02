from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import delphyne as dp
import delphyne.stdlib.commands.run_strategy as cmd
import delphyne.stdlib.experiments.learning_script as dl
import lean_interact as li
import minif2f

TRAINING_THEOREMS = minif2f.load_minif2f("valid")
TESTING_THEOREMS = minif2f.load_minif2f("test")
INIT_COMMANDS = ["import MiniF2F.Minif2fImport"]
MAX_DOLLARS_PER_PROBLEM = 2
MAX_REQUESTS_PER_PROBLEM = 100


def load_problem(problem_kind: dl.ProblemKind, problem_name: str):
    if problem_kind == "test":
        return TESTING_THEOREMS[problem_name]
    else:
        return TRAINING_THEOREMS[problem_name]


def solve_problem(problem_kind: dl.ProblemKind, problem_name: str):
    return cmd.RunStrategyArgs(
        strategy="prove_theorem",
        args={"theorem": load_problem(problem_kind, problem_name)},
        policy="ProveTheoremPolicy",
        policy_args={},  # We use default arguments
        budget={
            dp.DOLLAR_PRICE: MAX_DOLLARS_PER_PROBLEM,
            dp.NUM_REQUESTS: MAX_REQUESTS_PER_PROBLEM,
        },
    )


def generate_tips(feedback: dl.SerializedQueryFeedback) -> cmd.RunStrategyArgs:
    assert False


def summarize_tips(tips: Sequence[dl.Tip]):
    assert False


def global_init() -> li.LeanREPLConfig:
    from leandra.tools import lean_server_config

    repo_path = minif2f.repo_path()
    # We setup a memory limit so that 64 workers can work together on a
    # 200GB machine.
    mem = int(200_000 / 64)
    return lean_server_config(repo_path=repo_path, memory_hard_limit_mb=mem)


def worker_init(config: li.LeanREPLConfig):
    from leandra.tools import init_global_lean_server_with_config

    init_global_lean_server_with_config(config, INIT_COMMANDS)


def make_experiment(
    training_problems: list[str], testing_problems: list[str], output: str
):
    context = dp.workspace_execution_context(__file__)
    # We want custom initialization compatible with multiprocessing
    context = replace(context, init=())
    output_directory = Path(__file__).absolute().parent / output
    return dl.LearningExperiment(
        context=context,
        training_problems=training_problems,
        testing_problems=testing_problems,
        directory=output_directory,
        solve_problem=solve_problem,
        generate_tips=generate_tips,
        summarize_tips=summarize_tips,
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
        workers_setup=dp.WorkersSetup(
            common=global_init, per_worker=worker_init
        ),
    )
