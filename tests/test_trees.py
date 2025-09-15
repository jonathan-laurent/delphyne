"""
Testing reification and the `Tree` datastructure
"""

# pyright: basic

import textwrap

import example_strategies as ex
import pytest
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
    # To add serialized queries to the trace, we can register them manually.
    # tracer.trace_query(root_space)
    # Testing `nested_space`
    cands_space = root.node.nested_space(refs.SpaceName("cands", ()), ())
    assert isinstance(cands_space, dp.OpaqueSpace)
    # Testing to answer with the wrong sum
    wrong_sum_ans = root_space.parse_answer(dp.Answer(None, "[4, 6]"))
    assert not isinstance(wrong_sum_ans, dp.ParseError)
    wrong_sum = root.child(wrong_sum_ans)
    assert isinstance(wrong_sum.node, dp.Fail)
    assert wrong_sum.node.error.label == "wrong_sum"
    # Trying to answer with a forbidden number
    forbidden_ans = root_space.parse_answer(dp.Answer(None, "[4, 8]"))
    assert not isinstance(forbidden_ans, dp.ParseError)
    forbidden = root.child(forbidden_ans)
    assert isinstance(forbidden.node, dp.Fail)
    assert forbidden.node.error.label == "forbidden_num"
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
    pretty_trace = dump_yaml(
        dp.ExportableTrace, tracer.trace.export(), exclude_defaults=True
    )
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
                mode: null
                content: '[4, 6]'
              2:
                mode: null
                content: '[4, 8]'
              3:
                mode: null
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


def test_dual_number_generation_wrong_answer():
    tracer = dp.Tracer()
    cache: dp.TreeCache = {}
    monitor = dp.TreeMonitor(cache=cache, hooks=[dp.tracer_hook(tracer)])
    root = dp.reify(ex.dual_number_generation(), monitor)
    assert isinstance(root.node, dp.Branch)

    assert isinstance(root.node.cands, dp.OpaqueSpace)
    low_nested = root.node.cands.source()
    assert isinstance(low_nested, dp.NestedTree)
    low_node = low_nested.spawn_tree()
    assert isinstance(low_node.node, dp.Branch)
    low_query = low_node.node.cands.source()
    assert isinstance(low_query, dp.AttachedQuery)

    low_answer = low_query.parse_answer(dp.Answer(None, "25"))
    assert not isinstance(low_answer, dp.ParseError)
    with pytest.raises(refs.LocalityError):
        root.child(low_answer)
    low_success = low_node.child(low_answer)
    assert isinstance(low_success.node, dp.Success)
    low_answer_bis = low_success.node.success

    low_child = root.child(low_answer_bis)
    assert isinstance(low_child.node, dp.Branch)
    with pytest.raises(refs.LocalityError):
        low_child.child(low_answer)
    with pytest.raises(refs.LocalityError):
        low_child.child(low_answer_bis)


def test_imperative_strategy():
    tracer = dp.Tracer()
    cache: dp.TreeCache = {}
    monitor = dp.TreeMonitor(cache=cache, hooks=[dp.tracer_hook(tracer)])
    root = dp.reify(ex.imperative_strategy(), monitor)
    assert isinstance(root.node, dp.Branch)

    # Get the root query space for MakeSum
    root_space = root.node.cands.source()
    assert isinstance(root_space, dp.AttachedQuery)

    # Check that the original allowed list is [1, 2, 3]
    assert isinstance(root_space.query, ex.MakeSum)
    assert root_space.query.allowed == [1, 2, 3]

    # Take the [1, 2] branch for MakeSum
    makesum_answer = root_space.parse_answer(dp.Answer(None, "[1, 2]"))
    assert not isinstance(makesum_answer, dp.ParseError)
    makesum_child = root.child(makesum_answer)
    assert isinstance(makesum_child.node, dp.Branch)

    # Now we should be at the DummyChoice branch
    dummy_space = makesum_child.node.cands.source()
    assert isinstance(dummy_space, dp.AttachedQuery)
    assert isinstance(dummy_space.query, ex.DummyChoice)

    # Test first DummyChoice branch: answer with true
    dummy_answer1 = dummy_space.parse_answer(dp.Answer(None, "true"))
    assert not isinstance(dummy_answer1, dp.ParseError)
    success1 = makesum_child.child(dummy_answer1)
    assert isinstance(success1.node, dp.Success)

    # Test second DummyChoice branch: answer with false
    dummy_answer2 = dummy_space.parse_answer(dp.Answer(None, "false"))
    assert not isinstance(dummy_answer2, dp.ParseError)
    success2 = makesum_child.child(dummy_answer2)
    # NOTE: the test below would fail without performing the deepcopy in
    # `reification/_send_action`.
    assert isinstance(success2.node, dp.Success)

    # Verify that the original query's allowed field is still [1, 2, 3]
    # and has not been modified by the strategy execution
    assert root_space.query.allowed == [1, 2, 3]

    tracer.trace.check_consistency()
