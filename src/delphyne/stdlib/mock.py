"""
Special prompting policies that rely on mock oracles, for debugging.
"""

import json
from collections.abc import Callable, Iterable
from typing import Any

import delphyne as dp
import delphyne.core.demos as dm
import delphyne.core.refs as refs
from delphyne.stdlib.models import NUM_REQUESTS
from delphyne.stdlib.policies import PromptingPolicy, prompting_policy
from delphyne.stdlib.streams import SpendingDeclined, spend_on


def demo_mock_oracle(
    demo: dm.StrategyDemo,
    rev_search: bool = False,
    fail_on_missing: bool = True,
) -> PromptingPolicy:
    def find_answers(query: dp.AbstractQuery[Any]) -> Iterable[refs.Answer]:
        for q in demo.queries:
            if q.query != query.query_name():
                continue
            if json.dumps(q.args) != json.dumps(query.serialize_args()):
                continue
            for a in q.answers:
                yield dm.translate_answer(a)

    def policy[T](
        query: dp.AttachedQuery[T],
        env: dp.PolicyEnv,
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
    env: dp.PolicyEnv,
    oracle: Callable[[dp.AbstractQuery[Any]], Iterable[dp.Answer]],
) -> dp.StreamGen[T]:
    for answer in oracle(query.query):
        budget = dp.Budget({NUM_REQUESTS: 1})
        res = yield from spend_on(lambda: (None, budget), budget)
        if isinstance(res, SpendingDeclined):
            return
        parsed = query.parse_answer(answer)
        if not isinstance(parsed, dp.ParseError):
            yield dp.Solution(parsed)
