from domain.chunks import RetrievedChunk
from domain.citations import Claim, LocatedClaim


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

