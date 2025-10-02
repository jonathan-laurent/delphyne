from collections.abc import Sequence

import pytest


@pytest.mark.parametrize(
    "input,expected",
    [
        (
            """
            theorem my_theorem: 1 + 1 = 2 := by sorry
            """,
            ["theorem my_theorem: 1 + 1 = 2 := by"],
        ),
        (
            """
            theorem my_theorem
                (h: 0 + 0 = 0) :
                1 + 1 = 2 := by
              have: 0 + 1 = 1 := by grind
              sorry
            """,
            [
                "theorem my_theorem",
                "    (h: 0 + 0 = 0) :",
                "    1 + 1 = 2 := by",
            ],
        ),
    ],
)
def test_extract_theorem_lines(input: str, expected: Sequence[str]):
    import textwrap

    from leandra.dsl import _extract_theorem_lines  # type: ignore

    input = textwrap.dedent(input).strip()
    print("!\n" + input)
    result = _extract_theorem_lines(input)
    assert list(result) == list(expected)


@pytest.mark.parametrize(
    "input,num_proofs,expected",
    [
        (
            """
            comment: "This is an example"
            steps:
              - suppose: [my_hyp, "x > 0"]
                do:
                  - prove: [aux, "z > 0"]
                  - define: [m, "1"]
                conclude: [concl, "y > 0"]
            """,
            3,
            """
            theorem my_theorem: 1 + 1 = 2 := by
              have concl: (x > 0) → y > 0 := by
                intro my_hyp
                have aux: z > 0 := by sorry
                let m := 1
                sorry
              sorry
            """,
        ),
    ],
)
def test_compile(input: str, num_proofs: int, expected: str | None):
    import textwrap

    from delphyne.utils.yaml import load_yaml

    from leandra.dsl import ProofSketch, compile_dsl

    input = textwrap.dedent(input).strip()
    thm = "theorem my_theorem: 1 + 1 = 2 := by sorry"
    sketch = load_yaml(ProofSketch, input)
    res = compile_dsl(thm, sketch, num_proofs * [None])
    res_with_proofs = compile_dsl(thm, sketch, num_proofs * ["sorry"])
    print()
    print(res)
    print()
    print(res_with_proofs)
    if expected is not None:
        expected = textwrap.dedent(expected).strip()
        assert res.strip() == expected.strip()
