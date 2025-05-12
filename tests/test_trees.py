"""
Testing reification and the `Tree` datastructure
"""

# pyright: basic

import textwrap

import example_strategies as ex
from example_strategies import make_sum, synthetize_fun

import delphyne as dp
from delphyne.core import refs
from delphyne.utils.yaml import dump_yaml


def test_make_sum():
    # Reifying the strategy and inspecting the root
    tracer = dp.Tracer()
    cache: dp.TreeCache = {}
    monitor = dp.TreeMonitor(cache=cache, hooks=[dp.tracer_hook(tracer)])
    root = dp.reify(make_sum([4, 6, 2, 9], 11), monitor)
    assert root.ref == refs.MAIN_ROOT
    assert isinstance(root.node, dp.Branch)
    root_space = root.node.cands.source()
    assert isinstance(root_space, dp.AttachedQuery)
    # Testing `nested_space`
    cands_space = root.node.nested_space(refs.SpaceName("cands", ()), ())
    assert isinstance(cands_space, dp.OpaqueSpace)
    # Testing to answer with the wrong sum
    wrong_sum_ans = root_space.parse_answer(dp.Answer(None, "[4, 6]"))
    assert not isinstance(wrong_sum_ans, dp.ParseError)
    wrong_sum = root.child(wrong_sum_ans)
    assert isinstance(wrong_sum.node, dp.Failure)
    assert wrong_sum.node.message == "wrong-sum"
    # Trying to answer with a forbidden number
    forbidden_ans = root_space.parse_answer(dp.Answer(None, "[4, 8]"))
    assert not isinstance(forbidden_ans, dp.ParseError)
    forbidden = root.child(forbidden_ans)
    assert isinstance(forbidden.node, dp.Failure)
    assert forbidden.node.message == "forbidden-num"
    # Making a parse error
    parse_error = root_space.parse_answer(dp.Answer(None, "4, 8"))
    assert isinstance(parse_error, dp.ParseError)
    # Correct answer
    success_ans = root_space.parse_answer(dp.Answer(None, "[9, 2]"))
    assert not isinstance(success_ans, dp.ParseError)
    success = root.child(success_ans)
    assert isinstance(success.node, dp.Success)
    assert success.node.success.value == [9, 2]
    tracer.trace.check_consistency()
    opts = {"exclude_defaults": True}
    pretty_trace = dump_yaml(dp.ExportableTrace, tracer.trace.export(), **opts)
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
                content: '[4, 6]'
              2:
                mode:
                content: '[4, 8]'
              3:
                mode:
                content: '[9, 2]'
        """
    )
    print(pretty_trace)
    assert pretty_trace.strip() == expected.strip()


# Useful for nested strategies.
def test_synthetize_fun():
    tracer = dp.Tracer()
    cache: dp.TreeCache = {}
    monitor = dp.TreeMonitor(cache=cache, hooks=[dp.tracer_hook(tracer)])
    vars = ["x", "y"]
    prop = (["a", "b"], "F(a, b) == F(b, a) and F(0, 1) == 2")
    root = dp.reify(synthetize_fun(vars, prop), monitor)
    assert isinstance(root.node, ex.Conjecture)
    root_cands = root.node.cands.source()
    assert isinstance(root_cands, dp.NestedTree)
    conjecture_root = root_cands.spawn_tree()
    assert isinstance(conjecture_root.node, dp.Branch)
    query = conjecture_root.node.cands.source()
    assert isinstance(query, dp.AttachedQuery)
    answer = query.parse_answer(dp.Answer(None, "2*(x + y)"))
    assert not isinstance(answer, dp.ParseError)
    inner_succ = conjecture_root.child(answer)
    assert isinstance(inner_succ.node, dp.Success)
    success = root.child(inner_succ.node.success)
    assert isinstance(success.node, dp.Success)
    tracer.trace.check_consistency()
    pretty_trace = dump_yaml(dp.ExportableTrace, tracer.trace.export())
    print(pretty_trace)
