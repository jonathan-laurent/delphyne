# pyright: basic

import delphyne.core.inspect as insp


def test_function_args_dict():
    def f(x, y, z=0):
        pass

    assert insp.function_args_dict(f, (1, 2), {}) == {"x": 1, "y": 2}
