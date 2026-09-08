from pydantic import BaseModel

from domain.chunks import RetrievedChunk
from domain.citations import Claim, ClaimVerdict, Entailment, LocatedClaim
from llm import parse_call
from prompts import load as load_prompt

JUDGE_MODEL = 'gpt-5.5-2026-04-23'
JUDGE_PROMPT = 'judge'

class _Judgement(BaseModel):
    reason: str
    entailment: Entailment

def locate_citations(claims: list[Claim], chunks: list[RetrievedChunk]) -> list[LocatedClaim]:
    """Find the chunks that are cited by each claim"""

    results: list[LocatedClaim] = []
    for claim in claims:
        cited_chunks = []
        unresolved_citations = []
        for citation in claim.citations:
            if 1 <= citation <= len(chunks):
                cited_chunks.append(chunks[citation - 1])
            else:
                unresolved_citations.append(citation)
        results.append(LocatedClaim(claim=claim, cited_chunks=cited_chunks, unresolved_citations=unresolved_citations))
    return results

def judge_claims(located_claims: list[LocatedClaim], chunks: list[RetrievedChunk]) -> list[ClaimVerdict]:
    verdicts: list[ClaimVerdict] = []
    formatted_evidence = _format_evidence(chunks)
    prompt = load_prompt(JUDGE_PROMPT)

    for located in located_claims:
        judgement = parse_call(
            system=prompt.system,
            user=prompt.render(
                claim=located.claim.text,
                evidence=formatted_evidence,
            ),
            model=JUDGE_MODEL,
            schema=_Judgement
        )
        verdicts.append(ClaimVerdict(
            located=located,
            entailment=judgement.entailment,
            reason=judgement.reason
        ))

    return verdicts


def _format_evidence(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "(none)"
    formatted_chunks = [_format_chunk(chunk) for chunk in chunks]
    return "\n".join(formatted_chunks)

def _format_chunk(chunk: RetrievedChunk) -> str:
    return f"<chunk>\n{chunk.provenance}\n{chunk.chunk_text}\n</chunk>"