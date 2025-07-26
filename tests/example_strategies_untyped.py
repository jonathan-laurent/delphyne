"""
Simple example strategies, without using type annotations

Elements from this file are reexported in `example_strategies.py`.
"""

# pyright: basic

import delphyne as dp


@dp.strategy
def trivial_untyped_strategy(string, integer):
    return {"integer": integer, "string": string}
    yield
