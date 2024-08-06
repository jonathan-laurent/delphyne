from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import jinja2


PROMPT_DIR = "prompts"
SYSTEM_MESSAGE_SUFFIX = ""
INSTANCE_MESSAGE_SUFFIX = ".instance"


class JinjaPromptManager:
    def __init__(self, strategy_dirs: Sequence[Path]):
        prompt_folders = [dir / PROMPT_DIR for dir in strategy_dirs]
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(prompt_folders),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def prompt(
        self, type: Literal["system", "instance"], query_name: str, query: Any
    ) -> str | None:
        if type == "system":
            suffix = SYSTEM_MESSAGE_SUFFIX
        elif type == "instance":
            suffix = INSTANCE_MESSAGE_SUFFIX
        template_name = f"{query_name}{suffix}.jinja"
        try:
            template = self.env.get_template(template_name)
        except jinja2.TemplateNotFound:
            return None
        return template.render({"query": query})
