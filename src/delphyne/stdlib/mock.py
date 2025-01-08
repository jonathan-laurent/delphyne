"""
Special prompting policies that rely on mock oracles, for debugging.
"""

import json
from collections.abc import Iterable
from typing import Any

import delphyne as dp
import delphyne.core.demos as dm
import delphyne.core.refs as refs
from delphyne.stdlib.models import NUM_REQUESTS_BUDGET
from delphyne.stdlib.policies import PromptingPolicy


def demo_mock_oracle(
    demo: dm.Demonstration, rev_search: bool = False
) -> PromptingPolicy:
    def find_answers(query: dp.AbstractQuery[Any]) -> Iterable[refs.Answer]:
        for q in demo.queries:
            if q.query != query.name():
                continue
            if json.dumps(q.args) != json.dumps(query.serialize_args()):
                continue
            for a in q.answers:
                yield refs.Answer(a.mode, a.answer)

    async def policy[T](
        query: dp.AttachedQuery[T],
        env: dp.PolicyEnv,
    ) -> dp.Stream[T]:
        answers = list(find_answers(query.query))
        i, n = 0, len(answers)
        while True:
            answer = answers[n - 1 - i if rev_search else i]
            i = (i + 1) % n
            budget = dp.Budget({NUM_REQUESTS_BUDGET: 1})
            yield dp.Barrier(budget)
            yield dp.Spent(budget)
            parsed = query.answer(answer.mode, answer.text)
            if not isinstance(parsed, dp.ParseError):
                yield dp.Yield(parsed)

    return PromptingPolicy(policy)
