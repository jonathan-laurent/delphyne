"""
Concise references to values and choices.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class NodeId:
    id: int


@dataclass(frozen=True)
class AnswerId:
    id: int


type Assembly[T] = T | tuple[Assembly[T], ...]


type ChoiceRef = tuple[ChoiceLabel, tuple[ChoiceArgRef, ...]]


type ChoiceLabel = str


type ChoiceArgRef = int | ValueRef


type ValueRef = Assembly[ChoiceOutcomeRef]


type HintStr = str


type HintValue = str


@dataclass(frozen=True)
class Hint:
    """A hint for selecting a choice outcome.

    The hint can be conditioned to a specific subset of choices by
    specifying a selector.
    """

    query_name: str | None
    hint: HintValue


@dataclass(frozen=True)
class Hints:
    hints: tuple[Hint, ...]


@dataclass(frozen=True)
class ChoiceOutcomeRef:
    """A qualified reference to a choice outcome.

    - The value can be specified directly via an identifier or
      indirectly, via a sequence of hints or via a sequence of actions
      leading to a success node.
    - If `choice` is None, the choice must be inferred from context.
    """

    choice: ChoiceRef | None
    value: AnswerId | NodeId | Hints

    def __post_init__(self):
        assert self.choice is not None or isinstance(self.value, Hints)


type NodeOrigin = ChildOf | SubtreeOf


@dataclass(frozen=True)
class ChildOf:
    node: NodeId
    action: ValueRef


@dataclass(frozen=True)
class SubtreeOf:
    node: NodeId
    choice: ChoiceRef


"""
Utilities.

We start defining _basic_ references. Those cannot feature hints,
success paths or implicit choice references.
"""


def basic_choice_outcome_ref(cr: ChoiceOutcomeRef) -> bool:
    return isinstance(cr.value, (AnswerId, NodeId)) and cr.choice is not None


def basic_value_ref(vr: ValueRef) -> bool:
    if isinstance(vr, tuple):
        return all(basic_value_ref(v) for v in vr)
    return basic_choice_outcome_ref(vr)


def basic_choice_ref(cr: ChoiceRef) -> bool:
    return all(isinstance(a, int) or basic_value_ref(a) for a in cr[1])


def basic_node_origin(no: NodeOrigin) -> bool:
    match no:
        case ChildOf(_, action):
            return basic_value_ref(action)
        case SubtreeOf(_, choice):
            return basic_choice_ref(choice)
