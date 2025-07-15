import pprint
from dataclasses import dataclass
from typing import Any


@dataclass
class Error:
    """
    Base class for rich error messages that can be raised within
    strategies, while providing a chance to generate meaningful
    feedback. Parser errors are a particular case.
    """

    label: str | None = None
    description: str | None = None
    meta: Any | None = None

    def __init__(
        self,
        *,
        label: str | None = None,
        description: str | None = None,
        meta: Any | None = None,
    ):
        if label is not None:
            assert label
            # Should we prevent some characters in labels?
            # assert not any(c in label for c in [" ", "\n", "\t"])
        # assert label or description
        self.label = label
        self.description = description
        self.meta = meta

    def __str__(self):
        elts: list[str] = []
        if self.label:
            elts.append(self.label)
        if self.description:
            elts.append(self.description)
        if self.meta:
            elts.append(pprint.pformat(self.meta))
        return "\n\n".join(elts)
