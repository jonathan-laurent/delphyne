from dataclasses import dataclass


@dataclass
class Cell[T]:
    """
    A mutable reference to a value.

    This is useful for creating frozen dataclasses where only one field
    can be mutated.
    """

    content: T
