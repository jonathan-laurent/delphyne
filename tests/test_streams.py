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
# - Parallel operations are actually running in parallel.
# - If the estimates are wrong, I can overspend a little.
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
    spent: float
    num_gen: int
    duration: float
    exn: bool = False


DURATION_MULTIPLIER = 0.05
DURATION_PRECISION = 0.5 * DURATION_MULTIPLIER


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
    ],
)
def test_parallel(test: _ParTest):
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
        dp.SearchStream(lambda: worker(i, s)) for i, s in enumerate(test.plan)
    ]
    try:
        res, spent_actual = (
            dp.SearchStream.parallel(workers)
            .with_budget(_budget_limit(test.spent))
            .collect()
        )
    except Exception as e:
        print(e)
        assert test.exn
        return
    elapsed = time.time() - start
    assert len(res) == test.num_gen
    assert spent_actual.values.get(dp.DOLLAR_PRICE, 0) == test.spent
    duration = test.duration * DURATION_MULTIPLIER
    assert abs(duration - elapsed) < DURATION_PRECISION
    print(f"{spent_actual=} {len(res)=} {elapsed=:.2g}s")


if __name__ == "__main__":
    test_parallel(_ParTest([[_Step(2, 1, 1), _Step(1, 2, 1)]], 3, 1, 2))
