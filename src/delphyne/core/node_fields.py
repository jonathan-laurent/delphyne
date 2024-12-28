"""
Internal utilities to detect node fields.
"""

from dataclasses import dataclass


####
#### Fields
####


@dataclass
class SpaceF:
    """
    A space field that does not correspond to am embedded nested tree.
    """


@dataclass
class EmbeddedF:
    """
    A field corresponding to a nested embedded space.
    """


@dataclass
class DataF:
    """
    A field that corresponds to some other kind of data.
    """


@dataclass
class ParametricF:
    """
    A field corresponding to a parametric space.
    """

    res: SpaceF | EmbeddedF


@dataclass
class SequenceF:
    """
    A field corresponding to a sequence of spaces.
    """

    element: ParametricF | SpaceF | EmbeddedF


####
#### Inference Utilities
####
