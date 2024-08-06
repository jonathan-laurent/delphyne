"""
Using demonstrations to create mock oracles.
"""

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass

from delphyne.core.demos import Demonstration
from delphyne.core.queries import AnyQuery, Prompt
from delphyne.server.evaluate_demo import ExecutionContext, SerializedQuery
from delphyne.stdlib.generators import Budget
from delphyne.stdlib.search_envs import SearchEnv
from delphyne.utils.typing import pydantic_load


@dataclass
class DemoMockedSearchEnv(SearchEnv):
    """
    Wraps a search env, replacing LLM calls by access to a
    demonstration's data.
    """

    demo: Demonstration
    query_classes: Sequence[type[AnyQuery]]
    rev_search: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.queries: list[tuple[SerializedQuery, list[str]]] = []
        for q in self.demo.queries:
            query_class = self.load_query(q.query)
            query = pydantic_load(query_class, q.args)
            answers = [a.answer for a in q.answers]
            serialized = SerializedQuery.make(query)
            self.queries.append((serialized, answers))
        self.state: list[int] = [0] * len(self.queries)

    def load_query(self, name: str) -> type[AnyQuery]:
        for cls in self.query_classes:
            if cls.__name__ == name:
                return cls
        assert False

    def query_index(self, query: AnyQuery) -> int:
        serialized = SerializedQuery.make(query)
        for i, (q, _) in enumerate(self.queries):
            if q == serialized:
                return i
        assert False

    async def execute_prompt(
        self, query: AnyQuery, prompt: Prompt, n: int
    ) -> tuple[Sequence[str | None], Budget]:
        i = self.query_index(query)
        j = self.state[i]
        answers = self.queries[i][1]
        n = len(answers)
        answer = answers[n - 1 - j if self.rev_search else j]
        self.state[i] = (j + 1) % n
        return [answer], Budget.zero()


class MockedSearchEnv(SearchEnv):

    def __init__(self, oracle: Callable[[AnyQuery, Prompt], Iterable[str]]):
        # Since we'll never have to load demonstrations, we do not need
        # an execution context.
        super().__init__([], ExecutionContext([], []))
        self.oracle = oracle
        # queries are indexed by their string representation
        self.state: dict[str, Iterator[str]] = {}

    async def execute_prompt(
        self, query: AnyQuery, prompt: Prompt, n: int
    ) -> tuple[Sequence[str | None], Budget]:
        key = str(query)
        if key not in self.state:
            self.state[key] = iter(self.oracle(query, prompt))
        it = self.state[key]
        try:
            return [next(it) for _ in range(n)], Budget.spent(num_requests=n)
        except StopIteration:
            assert False, "Mock oracles must produce infinite streams."

    def collect_examples(
        self, query: AnyQuery
    ) -> Sequence[tuple[AnyQuery, str]]:
        return []

    def estimate_cost(self, prompt: Prompt) -> Budget:
        return Budget.spent(num_requests=1)
