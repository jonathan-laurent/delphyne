# """
# Testing reification and the `Tree` datastructure
# """

from example_strategies import make_sum

import delphyne as dp


def test_make_sum():
    # Reifying the strategy and inspecting the root
    root = dp.reify(make_sum([4, 6, 2, 9], 11))
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
