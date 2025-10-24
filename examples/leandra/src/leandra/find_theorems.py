"""
Strategy for finding Lean theorems
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import assert_never

import delphyne as dp
from delphyne import Branch, Compute, Fail, Strategy, strategy

from leandra import loogle_utils as loogle

# fmt: off


MAX_LOOGLE_HITS = 8
"""
Number of hits showed in the output of the Loogle tool.
"""


#####
##### Strategies
#####


@dataclass
class TheoremRequest:
    """
    Request for finding a particular theorem.

    Serves as the input to the `find_theorem` strategy.

    Attributes:
        name_guess: A guess of the name of the theorem to find.
            If the guess is correct, one can return its description
            straight away.
        description: An optional natural language description of the
            theorem being seeked.
        blacklist: A list of theorems that were proposed before but do
            not work for some explained reason.
    """

    name_guess: str
    description: str | None = None
    blacklist: Sequence["BlacklistedTheorem"] = ()


@dataclass
class BlacklistedTheorem:
    theorem: str
    reason: str


@strategy
def find_theorem(
    request: TheoremRequest,
) -> Strategy[Compute | Branch, "FindTheoremIP", loogle.LoogleHit]:
    """
    Strategy for implementing theorem finding requests.
    """
    IP = FindTheoremIP
    res = yield from unique_loogle_hit(request.name_guess)
    if res is not None:
        return res
    loogle_hit = yield from dp.interact(
        step=lambda prefix, _:
            FindTheorem(request, prefix).using(lambda p: p.step, IP),
        process=lambda ans, _:
            unique_loogle_hit_or_fail(ans).using(lambda p: p.process, IP),
        tools={LoogleTool: lambda call:
            call_loogle(call.loogle_request).using(lambda p: p.loogle, IP)})
    return loogle_hit


@dataclass
class LoogleTool(dp.AbstractTool[loogle.LoogleResults]):
    loogle_request: str


@dataclass
class FindTheorem(dp.Query[dp.Response[str, LoogleTool]]):
    request: TheoremRequest
    prefix: dp.AnswerPrefix
    __parser__ = dp.get_text.trim


@strategy
def call_loogle(
    request: str,
) -> Strategy[dp.Compute, object, loogle.LoogleResults]:
    res = yield from dp.compute(loogle.query_loogle)(request)
    if isinstance(res, loogle.LoogleHits):
        res = replace(res, hits=res.hits[:MAX_LOOGLE_HITS])
    return res


def unique_loogle_hit(
    theorem_name: str,
) -> Strategy[dp.Compute, object, loogle.LoogleHit | None]:
    res = yield from call_loogle(theorem_name).inline()
    if isinstance(res, loogle.LoogleHits) and len(res.hits) == 1:
        return res.hits[0]
    return None


@strategy
def unique_loogle_hit_or_fail(
    theorem_name: str,
) -> Strategy[dp.Compute | Fail, object, loogle.LoogleHit]:
    res = yield from unique_loogle_hit(theorem_name)
    if res is None:
        assert_never((yield from dp.fail("selected_absent_theorem")))
    return res


#####
##### Policies
#####

# fmt: on


@dataclass
class FindTheoremIP:
    step: dp.PromptingPolicy
    process: dp.Policy[Compute | Fail, None]
    loogle: dp.Policy[Compute, None]


@dp.ensure_compatible(find_theorem)
def find_theorem_policy(
    model_name: str = "gpt-5-mini",
    effort: dp.ReasoningEffort = "low",
    num_rounds: int = 3,
):
    model = dp.standard_model(model_name, {"reasoning_effort": effort})
    ip = FindTheoremIP(
        step=dp.few_shot(model),
        process=dp.exec @ dp.elim_compute() & None,
        loogle=dp.exec @ dp.elim_compute() & None,
    )
    return (dp.dfs() @ dp.elim_compute()) & ip
