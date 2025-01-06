"""
Utilities for writing policies.
"""

from typing import Any

import delphyne.core as dp


def log(
    env: dp.PolicyEnv,
    message: str,
    metadata: dict[str, Any] | None = None,
    loc: dp.Tree[Any, Any, Any] | dp.AttachedQuery[Any] | None = None,
) -> None:
    match loc:
        case None:
            location = None
        case dp.Tree():
            location = dp.Location(loc.ref, None)
        case dp.AttachedQuery(_, ref):
            location = dp.Location(ref[0], ref[1])
    env.tracer.log(message, metadata, location)
