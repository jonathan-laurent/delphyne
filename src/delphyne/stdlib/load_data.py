"""
The `Data` effect.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast, override

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


def pp_data_ref(ref: DataRef) -> str:
    if isinstance(ref, str):
        return ref
    else:
        return f"{ref[0]}[{ref[1]}]"


@dataclass
class DataNotFound(Exception):
    ref: DataRef

    def __str__(self):
        return f"Data not found: {pp_data_ref(self.ref)}"


type _RawAnswer = (
    tuple[Literal["not_found"], DataRef]
    | tuple[Literal["found"], Sequence[Any]]
)
"""
The `__LoadData__` query expects structured answers with JSON data of
type `_RawAnswer`.
"""


def produce_raw_answer(answer: Sequence[Any] | DataNotFound) -> _RawAnswer:
    if isinstance(answer, DataNotFound):
        return ("not_found", answer.ref)
    else:
        return ("found", answer)


def parse_raw_answer(raw: _RawAnswer) -> Sequence[Any] | DataNotFound:
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
        return parse_raw_answer(answer.content.structured)


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


def load_data(
    refs: Sequence[DataRef], type: Any = NoTypeInfo
) -> dp.Strategy[Data, object, Sequence[Any]]:
    """
    Load external data.

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

    query = __LoadData__(refs)
    result = yield dp.spawn_node(Data, query=query)
    result = cast(Sequence[Any] | DataNotFound, result)
    if isinstance(result, DataNotFound):
        raise result
    if isinstance(type, NoTypeInfo):
        return result
    else:
        return [pydantic_load(type, item) for item in result]
