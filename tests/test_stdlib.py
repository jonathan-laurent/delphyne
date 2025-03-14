from example_strategies import Conjecture

import delphyne as dp
from delphyne.core import node_fields as nf


def test_nodes():
    assert dp.Branch.fields() == {
        "cands": nf.SpaceF(),
        "extra_tags": nf.DataF(),
    }
    assert Conjecture.fields() == {
        "cands": nf.SpaceF(),
        "disprove": nf.ParametricF(nf.SpaceF()),
        "aggregate": nf.ParametricF(nf.SpaceF()),
    }
