"""
Strategy for finding Lean theorems.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace

import delphyne as dp
from delphyne import Branch, Compute, Strategy, strategy

from leandra import loogle_utils as loogle

# fmt: off


MAX_LOOGLE_HITS = 8
"""
Number of hits showed in the output of the Loogle tool.
"""

FIND_THEOREM_MODEL_CLASS = "find_theorem"


#####
##### Strategies
#####


@dataclass
class TheoremRequest(dp.AbstractTool[loogle.LoogleHit | None]):
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
    """
    Theorem blacklisted from consideration in a `TheoremRequest`.
    """

    theorem: str
    reason: str


@strategy
def find_theorem(
    request: TheoremRequest,
) -> Strategy[Compute | Branch, "FindTheoremIP", loogle.LoogleHit]:
    """
    Strategy for finding a Lean theorem.

    This strategy first tries to find a unique hit on Loogle using the
    guessed name. If successful, it returns immediately. Otherwise, a
    conversation is started where the LLM can make arbitrary Loogle
    requests and must eventually return the name of a valid theorem
    (which itself results in a unique Loogle hit).
    """
    IP = FindTheoremIP
    res = yield from unique_loogle_hit(request.name_guess)
    if res is not None:
        return res
    loogle_hit = yield from dp.interact(
        step=lambda prefix, _:
            FindTheorem(request, prefix).using(lambda p: p.step, IP),
        process=lambda ans, _:
            unique_loogle_hit_or_error(ans).using(lambda p: p.process, IP),
        tools={LoogleCall: lambda call:
            call_loogle(call.loogle_request).using(lambda p: p.loogle, IP)})
    return loogle_hit


@dataclass
class LoogleCall(dp.AbstractTool[loogle.LoogleResults]):
    """
    Loogle tool call.
    """

    loogle_request: str


@dataclass
class FindTheorem(dp.Query[dp.Response[str, LoogleCall]]):
    """
    Main query underlying the `find_theorem` strategy.

    Loogle tool calls can return error or success values. Returning a
    theorem name that does not exist (i.e. does not result in a unique
    Loogle hit) results in an error with label `not_unique_loogle_hit`.
    """

    request: TheoremRequest
    prefix: dp.AnswerPrefix
    __parser__ = dp.get_text.trim.response


@strategy
def call_loogle(
    request: str,
) -> Strategy[dp.Compute, object, loogle.LoogleResults]:
    """
    Strategy for calling Loogle. Only the first few results are
    returned (as does the #loogle command in Lean).
    """
    res = yield from dp.compute(loogle.query_loogle)(request)
    if isinstance(res, loogle.LoogleHits):
        res = replace(res, hits=res.hits[:MAX_LOOGLE_HITS])
    return res


def unique_loogle_hit(
    theorem_name: str,
) -> Strategy[dp.Compute, object, loogle.LoogleHit | None]:
    """
    Wrapper around `call_loogle` that returns a unique hit if there is one.

    !!! note
        This strategy is meant to be inlined, which is why it is not
        decorated with `@strategy` and why it is compatible with
        arbitrary inner policy types (`object`).
    """
    res = yield from call_loogle(theorem_name).inline()
    if isinstance(res, loogle.LoogleHits) and len(res.hits) == 1:
        return res.hits[0]
    return None


@strategy
def unique_loogle_hit_or_error(
    theorem_name: str,
) -> Strategy[dp.Compute, None, loogle.LoogleHit | dp.Error]:
    """
    Wrapper around `call_loogle` that returns an error unless there is a
    unique hit.
    """
    res = yield from unique_loogle_hit(theorem_name)
    if res is None:
        return dp.Error(label="not_unique_loogle_hit")
    return res


#####
##### Policies
#####


@dataclass
class FindTheoremIP:
    step: dp.PromptingPolicy
    process: dp.Policy[Compute, None]
    loogle: dp.Policy[Compute, None]


@dataclass
class FindTheoremPolicy(dp.PolicyRecord[Compute | Branch, FindTheoremIP]):
    """
    Standard policy for the `find_theorem` strategy.

    Attributes:
        model_name: Model used for crafting Loogle queries.
        effort: Reasoning effort for the model.
        max_requests: Maximum number of LLM requests allowed to find a
            theorem (each new tool call or round of feedback requires an
            additional request).
    """

    model_name: str = "gpt-5-mini"
    effort: dp.ReasoningEffort = "low"
    max_requests: int = 4

    def instantiate(self):
        model = dp.standard_model(
            self.model_name, {"reasoning_effort": self.effort},
            model_class=FIND_THEOREM_MODEL_CLASS)
        ip = FindTheoremIP(
            step=dp.few_shot(model),
            process=dp.exec @ dp.elim_compute() & None,
            loogle=dp.exec @ dp.elim_compute() & None)
        requests_limit = dp.BudgetLimit({dp.NUM_REQUESTS: self.max_requests})
        sp = dp.with_budget(requests_limit) @ dp.dfs() @ dp.elim_compute()
        return sp & ip
