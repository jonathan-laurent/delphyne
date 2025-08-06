import time
from collections.abc import Sequence
from dataclasses import dataclass

import pytest

import delphyne as dp


def _spend(
    *, estimate: float, cost: float, duration: float | None = None
) -> dp.StreamGen[bool]:
    metric = dp.DOLLAR_PRICE
    budget_estimate = dp.Budget({metric: estimate})
    budget_cost = dp.Budget({metric: cost})

    def task():
        if duration is not None:
            time.sleep(duration)
        return (None, budget_cost)

    ret = yield from dp.spend_on(task, estimate=budget_estimate)
    return not isinstance(ret, dp.SpendingDeclined)


def _dummy_tracked[T](obj: T) -> dp.Tracked[T]:
    return dp.Tracked(obj, None, None, None)  # type: ignore


def _solution[T](obj: T) -> dp.Stream[T]:
    yield dp.Solution(_dummy_tracked(obj))


def _budget_limit(float: float) -> dp.BudgetLimit:
    return dp.BudgetLimit({dp.DOLLAR_PRICE: float})


@pytest.mark.parametrize(
    "limit,spent,num_gen", [(0, 0, 0), (1, 0, 0), (2, 3, 2)]
)
def test_with_budget(limit: float, spent: float, num_gen: int):
    @dp.SearchStream
    def gen() -> dp.Stream[str]:
        if not (yield from _spend(estimate=2, cost=1)):
            return
        yield from _solution("A")
        if not (yield from _spend(estimate=1, cost=2)):
            return
        yield from _solution("B")

    res, spent_actual = gen.with_budget(_budget_limit(limit)).collect()
    assert len(res) == num_gen
    assert spent_actual.values.get(dp.DOLLAR_PRICE, 0) == spent


# What behaviors do I want to check?
# - If the real cost is less than the estimates, I complete more serially.
# - For many random schedules, I terminate and generate all elements.


@dataclass
class _Step:
    estimate: float
    cost: float
    duration: float
    exn: bool = False


@dataclass
class _ParTest:
    plan: Sequence[Sequence[_Step]]
    budget: float | None = None
    spent: float | None = None
    num_gen: int | None = None
    duration: float | None = None
    exn: bool = False

    def max_overspend(self):
        delta = max(s.cost - s.estimate for p in self.plan for s in p)
        delta = max(delta, 0)
        return len(self.plan) * delta

    def total_estimated_cost(self):
        return sum(s.estimate for p in self.plan for s in p)

    def total_cost(self):
        return sum(s.cost for p in self.plan for s in p)

    def num_workers(self):
        return len(self.plan)


DURATION_MULTIPLIER = 0.05
DURATION_PRECISION = 0.5 * DURATION_MULTIPLIER


def run_parallel_test(test: _ParTest):
    def worker(i: int, steps: Sequence[_Step]) -> dp.Stream[str]:
        for j, step in enumerate(steps):
            if step.exn:
                assert False
            print(f"? {i}.{j}")
            if not (
                yield from _spend(
                    estimate=step.estimate,
                    cost=step.cost,
                    duration=DURATION_MULTIPLIER * step.duration,
                )
            ):
                return
            print(f"! {i}.{j}")
        print(f"{{ {i} }}")
        yield from _solution("done")

    start = time.time()
    workers = [
        dp.SearchStream(lambda i=i, s=s: worker(i, s))
        for i, s in enumerate(test.plan)
    ]
    stream = dp.SearchStream.parallel(workers)
    if test.budget is not None:
        stream = stream.with_budget(_budget_limit(test.budget))
    try:
        res, spent_actual = stream.collect()
    except Exception as e:
        print(e)
        assert test.exn
        return
    elapsed = time.time() - start
    spent = spent_actual.values.get(dp.DOLLAR_PRICE, 0)
    if test.num_gen is not None:
        assert len(res) == test.num_gen
    if test.budget:
        assert spent <= test.budget + test.max_overspend()
    if (
        test.budget is None
        or test.budget >= test.total_cost() + test.max_overspend()
    ):
        assert len(res) == test.num_workers()
    if test.spent is not None:
        assert spent == test.spent
    if test.duration is not None:
        duration = test.duration * DURATION_MULTIPLIER
        assert abs(duration - elapsed) < DURATION_PRECISION
    print(f"{spent=} {len(res)=} {elapsed=:.2g}s")


@pytest.mark.parametrize(
    "test",
    [
        # No worker
        _ParTest([], spent=0, num_gen=0, duration=0),
        # One worker
        _ParTest(
            [[_Step(2, 1, 1), _Step(1, 2, 1)]],
            spent=3,
            num_gen=1,
            duration=2,
        ),
        # With exception
        _ParTest(
            [[_Step(2, 1, 1), _Step(1, 2, 1, exn=True)]],
            spent=3,
            num_gen=1,
            duration=2,
            exn=True,
        ),
        # Parallel operations running in parallel
        _ParTest(
            [[_Step(1, 1, 1)], [_Step(1, 1, 1)]],
            spent=2,
            num_gen=2,
            duration=1,
        ),
        # Not enough budget
        _ParTest(
            [[_Step(1, 1, 1), _Step(1, 1, 1)]],
            budget=1,
            spent=1,
            num_gen=0,
            duration=1,
        ),
        # If the estimates are wrong, I can overspend a little.
        _ParTest(
            [[_Step(2, 2, 2)], [_Step(1, 1, 1), _Step(1, 2, 1)]],
            budget=4,
            spent=5,
            num_gen=2,
            duration=2,
        ),
        _ParTest(
            [
                [_Step(2, 2, 2)],
                [_Step(0, 1, 1)],
                [_Step(0, 1, 1), _Step(1, 1, 1)],
            ],
            budget=2,
            spent=4,
            num_gen=2,
            duration=2,
        ),
        # If the real cost is less than the estimates, I complete more
        # serially.
        _ParTest(
            [
                [_Step(2, 1, 1)],
                [_Step(2, 1, 1)],
                [_Step(2, 1, 1)],
            ],
            budget=4,
            spent=3,
            num_gen=3,
            duration=2,
        ),
    ],
)
def test_parallel(test: _ParTest):
    run_parallel_test(test)


def test_parallel_random():
    import random

    random.seed(42)

    for _ in range(50):
        plan: list[_Step] = []
        for _ in range(random.randint(1, 5)):
            estimate = random.randint(1, 5)
            cost = random.randint(1, 5)
            duration = random.randint(1, 5) / 50
            plan.append(_Step(estimate, cost, duration))
        test = _ParTest([plan], budget=len(plan) * random.randint(1, 10))
        if random.random() > 0.5:
            test.budget = test.total_estimated_cost()
        run_parallel_test(test)


if __name__ == "__main__":
    # DURATION_MULTIPLIER = 1
    test_parallel(
        _ParTest(
            [[_Step(1, 1, 1)]],
            budget=0,
            spent=1,
            num_gen=0,
            duration=1,
        ),
    )
