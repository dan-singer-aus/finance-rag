from functools import cache

from openai import OpenAI

from domain.answers import GeneratedAnswer
from domain.chunks import RetrievedChunk
from prompts import load as load_prompt

MODEL = 'gpt-5.5-2026-04-23'
DEFAULT_PROMPT = 'answer'


def generate(question: str, context: list[RetrievedChunk], prompt_name: str = DEFAULT_PROMPT) -> GeneratedAnswer:
    """Generate an answer to a question, given a list of retrieved chunks as context."""
    filings = [chunk for chunk in context if chunk.corpus == 'filings']
    letters = [chunk for chunk in context if chunk.corpus == 'letters']
    ordered = filings + letters

    prompt = load_prompt(prompt_name)
    answer_text = _model_call(
        system=prompt.system,
        user=prompt.render(
            question=question,
            filings=_format_evidence(filings, 1),
            letters=_format_evidence(letters, len(filings) + 1),
        ),
        model=MODEL,
    )

    return GeneratedAnswer(
        question=question,
        text=answer_text,
        context=ordered,
        prompt_name=prompt_name,
        model=MODEL
    )
        
def _format_evidence(chunks: list[RetrievedChunk], start_at: int) -> str:
    """Format a list of retrieved chunks as evidence for a claim."""
    if not chunks:
        return "(none)"
    formatted_chunks = [
        f"[{i}] {chunk.provenance}\n{chunk.chunk_text}"
        for i, chunk in enumerate(chunks, start_at)
    ]
    return "\n".join(formatted_chunks)


@cache
def _client() -> OpenAI:
    return OpenAI()

def _model_call(system: str, user: str, model: str) -> str:
    """Send one system+user pair to the model and return its text.

    Knows nothing about answering — no prompt names, no placeholders, no
    evidence. That is deliberate: L4's splitter and judge need the identical
    call, and when the second caller arrives this moves out to a root-level
    `llm.py` beside `embedding.py` — an adapter no layer owns.
    """
    response = _client().responses.create(
        model=model,
        instructions=system,
        input=user,
    )
    return response.output_text
