"""
Standard Nodes
"""

from dataclasses import dataclass

from delphyne.core.trees import Node


@dataclass
class Branch(Node):
    pass
    # cands: OpaqueSpace[Any, Any]
