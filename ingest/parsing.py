from datetime import date, datetime

import frontmatter

from domain.documents import Document, FilingDocument, LetterDocument


def _require_str(metadata: dict[str, object], key: str) -> str:
    value = metadata.get(key)
    if not isinstance(value, str):
        raise TypeError(f"frontmatter '{key}': expected str, got {type(value).__name__}")
    return value

def _require_int(metadata: dict[str, object], key: str) -> int:
    value = metadata.get(key)
    # bool subclasses int, so a YAML `true` would pass a bare isinstance check.
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"frontmatter '{key}': expected int, got {type(value).__name__}")
    return value

def _require_date(metadata: dict[str, object], key: str) -> date:
    value = metadata.get(key)
    # datetime subclasses date. PyYAML yields a datetime as soon as the value
    # carries a time, and `period_end` is deliberately a date (2026-08-07) —
    # so a datetime here means the frontmatter changed shape, not that it's fine.
    if not isinstance(value, date) or isinstance(value, datetime):
        raise TypeError(f"frontmatter '{key}': expected date, got {type(value).__name__}")
    return value




def parse_document(post: frontmatter.Post) -> Document:
    match post.metadata.get("corpus"):
        case "letters":
            return LetterDocument(
                title=_require_str(post.metadata, "title"),
                doc_type=_require_str(post.metadata, "doc_type"),
                fiscal_year=_require_int(post.metadata, "fiscal_year"),
                source_url=_require_str(post.metadata, "source_url"),
                content=post.content,
                corpus="letters",
                author=_require_str(post.metadata, "author"),
            )
        case "filings":
            return FilingDocument(
                title=_require_str(post.metadata, "title"),
                doc_type=_require_str(post.metadata, "doc_type"),
                fiscal_year=_require_int(post.metadata, "fiscal_year"),
                source_url=_require_str(post.metadata, "source_url"),
                content=post.content,
                corpus="filings",
                company=_require_str(post.metadata, "company"),
                ticker=_require_str(post.metadata, "ticker"),
                cik=_require_int(post.metadata, "cik"),
                section=_require_str(post.metadata, "section"),
                period_end=_require_date(post.metadata, "period_end"),
                accession=_require_str(post.metadata, "accession"),
            )
        case other:
            raise ValueError(f"Unknown corpus: {other!r} in document {post.metadata.get('title')!r}") 