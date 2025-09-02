"""
Special prompting policies that rely on mock oracles, for debugging.
"""

import json
from collections.abc import Callable, Iterable
from typing import Any

import delphyne as dp
import delphyne.core.demos as dm
import delphyne.core.refs as refs
from delphyne.stdlib.environments import PolicyEnv
from delphyne.stdlib.models import NUM_REQUESTS
from delphyne.stdlib.policies import PromptingPolicy, prompting_policy
from delphyne.stdlib.streams import SpendingDeclined, spend_on


def demo_mock_oracle(
    demo: dm.StrategyDemo,
    rev_search: bool = False,
    fail_on_missing: bool = True,
) -> PromptingPolicy:
    """
    Produce a prompting policy from a demonstration by having an oracle
    cycle through the proposed answers on each query.

    Arguments:
        demo: the demonstration to use as an oracle.
        rev_search: if True, answers are cycled through in reverse order,
            starting with the last one.
        fail_on_missing: if True, raises an exception is a query is
            encountered that is not answered in the demonstration.

    Raises:
        ValueError: if no answers are found for a query and
            `fail_on_missing` is True.
    """

    def find_answers(query: dp.AbstractQuery[Any]) -> Iterable[refs.Answer]:
        for q in demo.queries:
            if q.query != query.query_name():
                continue
            if json.dumps(q.args, sort_keys=True) != json.dumps(
                query.serialize_args(), sort_keys=True
            ):
                continue
            for a in q.answers:
                yield dm.translate_answer(a)

    def policy[T](
        query: dp.AttachedQuery[T],
        env: PolicyEnv,
    ) -> dp.StreamGen[T]:
        answers = list(find_answers(query.query))
        i, n = 0, len(answers)
        if n == 0:
            if fail_on_missing:
                raise ValueError(f"Missing answer for {query.query}")
            return
        while True:
            answer = answers[n - 1 - i if rev_search else i]
            i = (i + 1) % n
            budget = dp.Budget({NUM_REQUESTS: 1})
            res = yield from spend_on(lambda: (None, budget), budget)
            if isinstance(res, SpendingDeclined):
                return
            parsed = query.parse_answer(answer)
            if not isinstance(parsed, dp.ParseError):
                yield dp.Solution(parsed)

    return PromptingPolicy(policy)


@prompting_policy
def fixed_oracle[T](
    query: dp.AttachedQuery[T],
    env: PolicyEnv,
    oracle: Callable[[dp.AbstractQuery[Any]], Iterable[dp.Answer]],
) -> dp.StreamGen[T]:
    """
    A minimal prompting policy that relies on a simple oracle function,
    for debugging and testing purposes.

    Arguments:
        oracle: a function that maps a query to an iterable collection
            of answers. The resulting prompting policy yields these
            answers in order.
    """
    for answer in oracle(query.query):
        budget = dp.Budget({NUM_REQUESTS: 1})
        res = yield from spend_on(lambda: (None, budget), budget)
        if isinstance(res, SpendingDeclined):
            return
        parsed = query.parse_answer(answer)
        if not isinstance(parsed, dp.ParseError):
            yield dp.Solution(parsed)
