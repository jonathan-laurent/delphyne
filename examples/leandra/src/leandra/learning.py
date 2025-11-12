"""
Strategies for generating tips
"""

from collections.abc import Sequence
from dataclasses import dataclass

import delphyne as dp
import delphyne.stdlib.experiments.learning_script as dl
from delphyne import Branch, Strategy, strategy

from leandra.prove_theorems import ProveGoal, SketchProof


@strategy
def generate_tips(
    feedback: dl.QueryFeedback,
) -> Strategy[Branch, dp.PromptingPolicy, Sequence[dl.Tip]]:
    if feedback.query.name == ProveGoal.__name__:
        query = feedback.query.parse(ProveGoal)
        tips = yield from dp.branch(
            GenerateProofTip(
                query.goal,
                good_answer=_answer_text(feedback.good_answers[0]),
                bad_answers=[
                    (_answer_text(ans), err)
                    for ans, err in feedback.bad_answers
                ],
            ).using(dp.ambient_pp),
        )
        return tips.tips
    elif feedback.query.name == SketchProof.__name__:
        query = feedback.query.parse(SketchProof)
        tips = yield from dp.branch(
            GenerateSketchTip(
                theorem=query.theorem,
                good_answer=_answer_json(feedback.good_answers[0]),
                bad_answers=[
                    (_answer_json(ans), err)
                    for ans, err in feedback.bad_answers
                ],
            ).using(dp.ambient_pp),
        )
        return tips.tips
    else:
        assert False, f"Unknown query: {feedback.query.name}"


def _answer_text(ans: dp.Answer) -> str:
    assert isinstance(ans.content, str)
    return ans.content


def _answer_json(ans: dp.Answer) -> object:
    assert isinstance(ans.content, dp.Structured)
    return ans.content.structured


@dataclass
class Tips:
    # Wrapper around tips so that we can use structured output
    tips: Sequence[dl.Tip]


@dataclass
class GenerateSketchTip(dp.Query[Tips]):
    theorem: str
    good_answer: object  # in JSON format
    bad_answers: Sequence[tuple[object, dp.Error]]
    __parser__ = dp.structured


@dataclass
class GenerateProofTip(dp.Query[Tips]):
    goal: str
    good_answer: str
    bad_answers: Sequence[tuple[str, dp.Error]]
    __parser__ = dp.structured


@strategy
def summarize_tips(
    query_type: str,
    tips: Sequence[dl.Tip],
) -> Strategy[Branch, dp.PromptingPolicy, Sequence[dl.Tip]]:
    summarized = yield from dp.branch(
        SummarizeTips(query_type, tips).using(dp.ambient_pp)
    )
    return summarized.tips


@dataclass
class SummarizeTips(dp.Query[Tips]):
    query_type: str
    tips: Sequence[dl.Tip]
    __parser__ = dp.structured


def generate_tips_policy(
    model_name: str, effort: dp.ReasoningEffort
) -> dp.Policy[Branch, dp.PromptingPolicy]:
    model = dp.standard_model(model_name, {"reasoning_effort": effort})
    return dp.dfs() & dp.few_shot(model)
