"""
Testing reification and the `Tree` datastructure
"""

import textwrap

from example_strategies import make_sum

import delphyne as dp
from delphyne.utils.yaml import dump_yaml


def test_make_sum():
    # Reifying the strategy and inspecting the root
    tracer = dp.Tracer()
    root = dp.reify(make_sum([4, 6, 2, 9], 11), dp.tracer_hook(tracer))
    assert isinstance(root.node, dp.Branch)
    root_space = root.node.cands.source()
    assert isinstance(root_space, dp.AttachedQuery)
    # Testing to answer with the wrong sum
    wrong_sum_ans = root_space.answer(None, "[4, 6]")
    assert not isinstance(wrong_sum_ans, dp.ParseError)
    wrong_sum = root.child(wrong_sum_ans)
    assert isinstance(wrong_sum.node, dp.Failure)
    assert wrong_sum.node.message == "wrong-sum"
    # Trying to answer with a forbidden number
    forbidden_ans = root_space.answer(None, "[4, 8]")
    assert not isinstance(forbidden_ans, dp.ParseError)
    forbidden = root.child(forbidden_ans)
    assert isinstance(forbidden.node, dp.Failure)
    assert forbidden.node.message == "forbidden-num"
    # Making a parse error
    parse_error = root_space.answer(None, "4, 8")
    assert isinstance(parse_error, dp.ParseError)
    # Correct answer
    success_ans = root_space.answer(None, "[9, 2]")
    assert not isinstance(success_ans, dp.ParseError)
    success = root.child(success_ans)
    assert isinstance(success.node, dp.Success)
    assert success.node.success.value == [9, 2]
    pretty_trace = dump_yaml(dp.ExportableTrace, tracer.trace.export())
    expected = textwrap.dedent(
        """
        nodes:
          1: nested(0, $main)
          2: child(1, cands{@1})
          3: child(1, cands{@2})
          4: child(1, cands{@3})
        queries:
          - node: 1
            space: cands
            answers:
              1:
                mode:
                text: '[4, 6]'
              2:
                mode:
                text: '[4, 8]'
              3:
                mode:
                text: '[9, 2]'
        """
    )
    print(pretty_trace)
    assert pretty_trace.strip() == expected.strip()
