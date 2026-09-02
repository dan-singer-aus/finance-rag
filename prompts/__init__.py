"""The prompt library: the `.yml` files beside this module, and the loader.

Ancillary scaffolding — not learning-target code. Same shape and same reason as
`embedding.py`: a small typed boundary over something vendor- or file-shaped, so
the rest of the system deals in a record rather than in a dict of `Any`.

Prompts live here rather than inside the package that uses them because prompt
iteration is cross-cutting: the generator, the L4 splitter and the L4 judge each
need one, and the comparison harness wants to swap between arms (`answer.yml` vs
`answer-naive.yml`) without reaching into three packages. They are also content
— reviewed and diffed as text, not as code.

**Loader and content share one package deliberately.** They started as a root
`prompts.py` module beside a `prompts/` directory, which works only by accident:
Python resolves a module ahead of a namespace package, so adding an `__init__.py`
to the directory would have shadowed the module and taken `load` with it. One
name, one package, no collision — and once something has to interpret the files,
a package that carries its own loader is the honest shape.

**Nothing here names a model.** A model in a prompt file couples two things that
must vary independently: the answer arms exist to change *one* variable and read
the delta, and a prompt-plus-model file makes that delta unattributable. They
also change for different reasons — prompt text is content, model choice is
operational (cost, latency, deprecation). `(prompt, model)` is an experiment
configuration owned by the caller; which pair produced a given answer is recorded
on the result, not on the input.

Files are named by role, with the arm as a suffix: `answer.yml`,
`answer-naive.yml`, later `split.yml`, `judge.yml`, `rewrite.yml`. Flat until a
role grows several arms.

**Substitution is plain string replacement on `<%name%>`.** Deliberately not
`str.format` and not `string.Template`: `{}` and `$` both appear in financial
prose ("$29.0 billion"), and both of those mechanisms would assign meaning to a
character the corpus contains. `str.replace` assigns meaning to nothing.

The one non-obvious behaviour is `render`'s check for leftover placeholders. A
template rendered with a field missing would otherwise reach the model with a
literal `<%letters%>` in it — which reads as a slightly odd prompt rather than
as an error, and produces a plausible answer built on absent evidence. That is
exactly the failure this project exists to catch, so it raises.
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
        raise FileNotFoundError(f"no prompt {name!r} in {PROMPTS_DIR} (have: {available})")

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
