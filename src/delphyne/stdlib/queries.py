"""
Standard queries, backed by Jinja templates
"""

from collections.abc import Callable
from typing import Self, cast

from delphyne.core.environment import TemplatesManager
from delphyne.core.inspect import first_parameter_of_base_class
from delphyne.core.queries import AbstractQuery, AnswerModeName
from delphyne.core.trees import Builder, OpaqueSpace, PromptingPolicy
from delphyne.utils.typing import (
    TypeAnnot,
    pydantic_dump,
    pydantic_load,
)


class Query[T](AbstractQuery[T]):
    def system_prompt(
        self,
        env: TemplatesManager,
        mode: AnswerModeName,
        params: dict[str, object],
    ) -> str:
        """
        Raises `TemplateNotFound`.
        """
        args = {"query": self, "mode": mode, "params": params}
        return env.prompt("system", self.name(), args)

    def instance_prompt(
        self,
        env: TemplatesManager,
        mode: AnswerModeName,
        params: dict[str, object],
    ) -> str:
        """
        Raises `TemplateNotFound`.
        """
        args = {"query": self, "mode": mode, "params": params}
        return env.prompt("instance", self.name(), args)

    def serialize_args(self) -> dict[str, object]:
        return cast(dict[str, object], pydantic_dump(type(self), self))

    @classmethod
    def parse(cls, args: dict[str, object]) -> Self:
        return pydantic_load(cls, args)

    def answer_type(self) -> TypeAnnot[T]:
        return first_parameter_of_base_class(type(self))

    def __getitem__[P](
        self, get_policy: Callable[[P], PromptingPolicy]
    ) -> Builder[OpaqueSpace[P, T]]:
        return OpaqueSpace[P, T].from_query(self, get_policy)
