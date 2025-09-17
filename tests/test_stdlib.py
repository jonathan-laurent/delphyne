import pytest
from example_strategies import Conjecture

import delphyne as dp
from delphyne.core import node_fields as nf


def test_nodes():
    assert dp.Branch.fields() == {
        "cands": nf.SpaceF(),
        "meta": nf.DataF(),
    }
    assert Conjecture.fields() == {
        "cands": nf.SpaceF(),
        "disprove": nf.ParametricF(nf.SpaceF()),
        "aggregate": nf.ParametricF(nf.SpaceF()),
    }
    assert dp.Join.fields() == {
        "subs": nf.SequenceF(nf.EmbeddedF()),
        "meta": nf.DataF(),
    }


#####
##### Testing Parsers
#####

UNBALANCED = """
```
d
```
hello
```
"""

BALANCED = """

```
foo
```

Hello.

```python
bar
```

I am here.
"""


def test_yaml_block():
    from delphyne.stdlib.queries import extract_final_block

    assert extract_final_block(UNBALANCED) == "hello\n"
    assert extract_final_block(BALANCED) == "bar\n"


#####
##### Testing standard models
#####


def test_pricing_dict_exhaustiveness():
    import delphyne.stdlib.standard_models as sm

    sm.test_pricing_dict_exhaustiveness()


def test_unknown_model():
    with pytest.raises(ValueError, match="provider"):
        dp.standard_model("unknown-model-123")

    with pytest.raises(ValueError, match="Pricing"):
        dp.openai_model("unknown-model-123")


def test_standard_model_with_suffix():
    model = dp.standard_model("gpt-4o-2024-08-06")
    assert model.pricing is not None

    # We explicitly choose not to infer pricing
    model = dp.openai_model("unknown_openai_model", pricing=None)
    assert model.pricing is None


#####
##### Testing loaders
#####


@pytest.mark.parametrize(
    "input,expected",
    [
        (
            "nested(23, compare([cands{%2}, cands{%11}]))",
            (set([23, 2, 11]), set[int]()),
        ),
        (
            "child(2, cands{@3})",
            (set([2]), set([3])),
        ),
    ],
)
def test_node_and_answer_ids_in_node_origin_string(
    input: str, expected: tuple[set[int], set[int]]
):
    import delphyne.stdlib.answer_loaders as al

    assert al.node_and_answer_ids_in_node_origin_string(input) == expected
