"""
The `Data` effect.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Never, cast, overload, override

import delphyne.core_and_base as dp
from delphyne.utils.typing import NoTypeInfo, pydantic_load
from delphyne.utils.yaml import dump_yaml

type DataRef = str | tuple[str, str]
"""
Reference to some data, which either consists in:

- A file name (string), in which case the reference refers to the whole
  file's content.
- A (file name, key) pair, in which case the reference refers to the
  value associated with `key` in the dictionary contained in the file.
"""


def _pp_data_ref(ref: DataRef) -> str:
    if isinstance(ref, str):
        return ref
    else:
        return f"{ref[0]}[{ref[1]}]"


@dataclass
class DataNotFound(Exception):
    """
    Exception raised when some data is not found.
    """

    ref: DataRef

    def __str__(self):
        return f"Data not found: {_pp_data_ref(self.ref)}"


type _RawAnswer = (
    tuple[Literal["not_found"], DataRef]
    | tuple[Literal["found"], Sequence[Any]]
)
"""
The `__LoadData__` query expects structured answers with JSON data of
type `_RawAnswer`.
"""


def _produce_raw_answer(answer: Sequence[Any] | DataNotFound) -> _RawAnswer:
    if isinstance(answer, DataNotFound):
        return ("not_found", answer.ref)
    else:
        return ("found", answer)


def _parse_raw_answer(raw: _RawAnswer) -> Sequence[Any] | DataNotFound:
    tag, payload = raw
    if tag == "not_found":
        assert isinstance(payload, str) or (
            isinstance(payload, tuple)
            and len(payload) == 2
            and all(isinstance(x, str) for x in payload)
        )
        return DataNotFound(payload)
    else:
        assert tag == "found"
        assert isinstance(payload, Sequence)
        return payload


@dataclass
class __LoadData__(dp.AbstractQuery[Sequence[Any] | DataNotFound]):
    """
    A special query that represents the loading of data.

    Returns a sequence of parsed JSON values.

    Attributes:
        refs: Sequence of data references to load.
    """

    refs: Sequence[DataRef]

    @override
    def generate_prompt(
        self,
        *,
        kind: str,
        mode: dp.AnswerMode,
        params: dict[str, object],
        extra_args: dict[str, object] | None = None,
        env: dp.AbstractTemplatesManager | None = None,
    ) -> str:
        return dump_yaml(Any, self.__dict__)

    @override
    def query_modes(self):
        return [None]

    @override
    def answer_type(self):
        return object

    @override
    def parse_answer(self, answer: dp.Answer) -> Sequence[Any] | DataNotFound:
        assert isinstance(answer.content, dp.Structured)
        return _parse_raw_answer(answer.content.structured)


@dataclass
class Data(dp.Node):
    """
    The standard "Data" effect.

    This effect allows loading external data. Strategies must be
    monotonic with respect to data, meaning that adding a new data file
    or adding a new key into a data dictionary must not break an
    oracular program. Thus, a form of learning can be implemented by
    growing a database of learned facts.
    """

    query: dp.TransparentQuery[Any]

    def navigate(self) -> dp.Navigation:
        return (yield self.query)


@overload
def load_data[T](
    refs: Sequence[DataRef], type: type[T]
) -> dp.Strategy[Data, object, Sequence[T]]: ...


@overload
def load_data(
    refs: Sequence[DataRef], type: Any = NoTypeInfo
) -> dp.Strategy[Data, object, Sequence[Any]]: ...


def load_data(
    refs: Sequence[DataRef], type: Any = NoTypeInfo
) -> dp.Strategy[Data, object, Sequence[Any]]:
    """
    Load external data.

    An exception is raised if any piece of data is not found, thus
    enforcing monotonicity (assuming this exception is never caught in
    strategy code).

    Arguments:
        refs: References to the data to load. A list of identical length
            is returned.
        type: Optional type information for the elements of the returned
            list. If provided, Pydantic is used to load the data.

    Raises:
        DataNotFound: If any of the data references could not be found.
    """

    if not refs:
        return []

    query = dp.TransparentQuery.build(__LoadData__(refs))
    result = yield dp.spawn_node(Data, query=query)
    result = cast(Sequence[Any] | DataNotFound, result)
    if isinstance(result, DataNotFound):
        raise result
    if isinstance(type, NoTypeInfo):
        return result
    else:
        return [pydantic_load(type, item) for item in result]


def _load_data_ref(manager: dp.DataManager, ref: DataRef) -> Any:
    try:
        if isinstance(ref, str):
            return manager.data[ref]
        else:
            file, key = ref
            return manager.data[file][key]
    except Exception:
        raise DataNotFound(ref)


def _produce_answer(manager: dp.DataManager, query: __LoadData__) -> dp.Answer:
    try:
        res = [_load_data_ref(manager, r) for r in query.refs]
    except DataNotFound as e:
        res = e
    return dp.Answer(None, dp.Structured(_produce_raw_answer(res)))


@dp.contextual_tree_transformer
def elim_data(
    env: dp.PolicyEnv,
    policy: Any,
) -> dp.PureTreeTransformerFn[Data, Never]:
    def transform[N: dp.Node, P, T](
        tree: dp.Tree[Data | N, P, T],
    ) -> dp.Tree[N, P, T]:
        if isinstance(tree.node, Data):
            query = tree.node.query.attached.query
            assert isinstance(query, __LoadData__)
            answer = _produce_answer(env.data_manager, query)
            tracked = tree.node.query.attached.parse_answer(answer)
            assert not isinstance(tracked, dp.ParseError)
            return transform(tree.child(tracked))
        return tree.transform(tree.node, transform)

    return transform


def load_implicit_answer_generator(data_dirs: Sequence[Path]):
    manager = dp.DataManager(data_dirs)

    def generator(
        tree: dp.AnyTree, query: dp.AttachedQuery[Any]
    ) -> tuple[dp.ImplicitAnswerCategory, dp.Answer] | None:
        if isinstance(tree.node, Data):
            assert isinstance(query.query, __LoadData__)
            answer = _produce_answer(manager, query.query)
            return ("data", answer)

    return generator
