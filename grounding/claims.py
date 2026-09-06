from pydantic import BaseModel

from domain.citations import Claim
from llm import parse_call
from prompts import load as load_prompt
import re

MODEL = 'gpt-5.5-2026-04-23'
SPLIT_PROMPT = 'split'
_MARKER = re.compile(r"\[(\d+)\]")

class _SplitClaim(BaseModel):
    claim: str
    source_sentence: str

class _SplitResult(BaseModel):
    claims: list[_SplitClaim]

def split_claims(answer: str) -> list[Claim]:
    """Split an answer into individual claims."""
    prompt = load_prompt(SPLIT_PROMPT)
    results = parse_call(
        system=prompt.system,
        user=prompt.render(answer=answer),
        model=MODEL,
        schema=_SplitResult
    )
    return [
    Claim(claim=c.claim, citations=_citations_in(c.source_sentence))
    for c in results.claims
]

def _citations_in(sentence: str) -> list[int]:
    """The [n] markers in one sentence, as integers."""
    return [int(n) for n in _MARKER.findall(sentence)]

