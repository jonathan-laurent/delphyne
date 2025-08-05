import time

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
