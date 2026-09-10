"""The prompt library: the `.yml` files beside this module, and the loader.

Files are named by role, with the arm as a suffix — `answer.yml`,
`answer-naive.yml`, `split.yml`, `judge.yml`. Flat until a role grows several
arms.

⚠️ **Never name a model in a prompt file.** The arms exist to vary *one* thing
and read the delta; a prompt-plus-model file makes that delta unattributable.
They also change for different reasons — prompt text is content, model choice is
operational. `(prompt, model)` is an experiment configuration owned by the
caller, and which pair produced a result is recorded on the result.

⚠️ **Substitution is plain `str.replace` on `<%name%>` — do not "improve" it to
`str.format` or `string.Template`.** `{}` and `$` both occur in financial prose
("$29.0 billion"), so both of those would assign meaning to characters the
corpus contains. `str.replace` assigns meaning to nothing.

`render` raises on a leftover placeholder. A template rendered with a field
missing would otherwise reach the model with a literal `<%letters%>` in it,
which reads as a slightly odd prompt rather than an error — and produces a
plausible answer built on absent evidence.
"""

import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import yaml

PROMPTS_DIR = Path(__file__).parent

_PLACEHOLDER = re.compile(r"<%\s*(\w+)\s*%>")


@dataclass(frozen=True)
class Prompt:
    """One prompt file: a system message and a user-message template."""

    name: str
    system: str
    user: str

    def render(self, **fields: str) -> str:
        """Fill the user template. Raises if any placeholder is left unfilled."""
        rendered = self.user
        for key, value in fields.items():
            rendered = rendered.replace(f"<%{key}%>", value)

        leftover = sorted(set(_PLACEHOLDER.findall(rendered)))
        if leftover:
            raise ValueError(
                f"prompt {self.name!r} still has unfilled placeholders: "
                f"{', '.join(leftover)} (given: {', '.join(sorted(fields)) or 'nothing'})"
            )
        return rendered


@cache
def load(name: str) -> Prompt:
    """Load `prompts/<name>.yml`. Cached — prompt files don't change mid-run."""
    path = PROMPTS_DIR / f"{name}.yml"
    if not path.exists():
        available = ", ".join(sorted(p.stem for p in PROMPTS_DIR.glob("*.yml")))
        raise FileNotFoundError(
            f"no prompt {name!r} in {PROMPTS_DIR} (have: {available})"
        )

    document = yaml.safe_load(path.read_text())
    if not isinstance(document, dict):
        raise TypeError(f"{path} is not a YAML mapping")

    return Prompt(
        name=str(document.get("name", name)),
        system=_required_str(document, "system", path),
        user=_required_str(document, "user", path),
    )


def _required_str(document: dict[str, object], key: str, path: Path) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} is missing a non-empty {key!r} field")
    return value
