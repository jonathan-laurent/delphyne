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
