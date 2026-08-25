SEARCH_SQL = """
    SELECT
        chunks.chunk_text,
        chunks.chunk_index,
        chunks.source_id,
        sources.title,
        sources.doc_type,
        sources.fiscal_year,
        sources.source_url,
        sources.corpus,
        sources.company,
        sources.ticker,
        sources.section,
        sources.period
        
        
        _end,
        1 - (chunks.embedding <=> %(embedding)s::vector) AS score
    FROM chunks
    JOIN sources ON chunks.source_id = sources.id
    ORDER BY chunks.embedding <=> %(embedding)s::vector
    LIMIT %(k)s
"""







