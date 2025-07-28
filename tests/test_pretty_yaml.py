import pytest

from delphyne.utils.yaml import pretty_yaml

NO_MULTI = {"name": ("foo", "bar")}

WITH_MULTI = {
    "name": "Example",
    "descr": "This is a multiline\nstring that should be\nin block style.",
    "notes": ["Single line", "Another\nmultiline\nnote."],
}


@pytest.mark.parametrize(
    "obj,multi",
    [(NO_MULTI, False), (WITH_MULTI, True)],
)
def test_pretty_yaml(obj: object, multi: bool):
    res = pretty_yaml(obj, width=50)
    print(res)
    assert ("|" in res) == multi


def test_pretty_yaml_safe():
    class A:
        pass

    non_jsonv = {"non_serializable": A()}
    with pytest.raises(Exception):
        pretty_yaml(non_jsonv)
