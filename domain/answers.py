from dataclasses import dataclass

from domain.chunks import RetrievedChunk


@dataclass(frozen=True)
class GeneratedAnswer:
    question: str
    text: str
    context: list[RetrievedChunk]
    prompt_name: str
    model: str
