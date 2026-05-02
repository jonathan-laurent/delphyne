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


def test_standard_model_with_chat_completions_api():
    model = dp.standard_model("gpt-5-nano")
    assert isinstance(model, dp.OpenAICompatibleModel)


def test_standard_model_with_responses_api():
    model = dp.standard_model("gpt-5-nano", api_type="responses")
    assert isinstance(model, dp.OpenAIResponsesModel)
    assert model.use_reasoning_cache is True
    assert model.reasoning_cache is not None

    model_no_cache = dp.standard_model(
        "gpt-5-nano", api_type="responses", use_reasoning_cache=False
    )
    assert isinstance(model_no_cache, dp.OpenAIResponsesModel)
    assert model_no_cache.reasoning_cache is None


def test_standard_model_with_chat_completions_api_and_reasoning_cache():
    with pytest.raises(ValueError, match="Reasoning"):
        dp.standard_model("gpt-5-nano", use_reasoning_cache=True)


def test_standard_model_with_non_openai_model_and_responses_api():
    with pytest.raises(ValueError, match="OpenAI models"):
        dp.standard_model("gemini-2.5-pro", api_type="responses")
