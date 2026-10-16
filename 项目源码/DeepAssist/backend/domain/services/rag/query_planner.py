import re
from typing import Iterable, List, Set

from backend.domain.entities.retrieval import QueryPlan
from backend.domain.services.rag.query_rewriter import QueryRewriteService


class QueryPlanner:
    """Deterministic query planning before retrieval.

    This is a small replaceable stage. Later we can add LLM rewrite or multi-query
    expansion here without changing the retriever contract.
    """

    _term_pattern = re.compile(r"[A-Za-z0-9_./#:+-]+|[\u4e00-\u9fff]{2,}")
    _quote_pattern = re.compile(r'"([^"]+)"|\'([^\']+)\'')
    _stop_words: Set[str] = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "for",
        "how",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "what",
        "when",
        "where",
        "which",
        "why",
        "with",
    }

    def __init__(self, query_rewriter: QueryRewriteService | None = None):
        self.query_rewriter = query_rewriter or QueryRewriteService()

    def plan(self, query: str) -> QueryPlan:
        original_query = query or ""
        normalized_query = self._normalize(original_query)
        quoted_phrases = self._extract_quoted_phrases(normalized_query)
        key_terms = self._extract_key_terms(normalized_query, quoted_phrases)
        keyword_query = self._build_keyword_query(
            normalized_query=normalized_query,
            quoted_phrases=quoted_phrases,
            key_terms=key_terms,
        )
        expansion = self.query_rewriter.expand(
            normalized_query=normalized_query,
            keyword_query=keyword_query,
            key_terms=key_terms,
            quoted_phrases=quoted_phrases,
        )

        return QueryPlan(
            original_query=original_query,
            normalized_query=normalized_query,
            semantic_query=normalized_query,
            keyword_query=keyword_query,
            rewritten_query=expansion.rewritten_query,
            semantic_queries=expansion.semantic_queries,
            keyword_queries=expansion.keyword_queries,
            sub_queries=expansion.sub_queries,
            key_terms=key_terms,
            quoted_phrases=quoted_phrases,
            strategy=expansion.strategy,
            metadata={
                "planner": "deterministic_v1",
                "rewriter": "deterministic_v1",
                "term_count": len(key_terms),
                "quoted_phrase_count": len(quoted_phrases),
                "semantic_query_count": len(expansion.semantic_queries),
                "keyword_query_count": len(expansion.keyword_queries),
                "rewrite_notes": expansion.rewrite_notes,
                "stripped_fillers": expansion.stripped_fillers,
                "domain_hints": expansion.domain_hints,
                "rewrite_applied": (
                    expansion.rewritten_query != normalized_query
                    or bool(expansion.semantic_queries[1:])
                    or bool(expansion.rewrite_notes)
                ),
            },
        )

    def _normalize(self, query: str) -> str:
        return re.sub(r"\s+", " ", query).strip()

    def _extract_quoted_phrases(self, query: str) -> List[str]:
        phrases: List[str] = []
        for match in self._quote_pattern.finditer(query):
            phrase = next(group for group in match.groups() if group)
            phrase = self._normalize(phrase)
            if phrase:
                phrases.append(phrase)
        return self._dedupe(phrases)

    def _extract_key_terms(self, query: str, quoted_phrases: List[str]) -> List[str]:
        query_without_quotes = query
        for phrase in quoted_phrases:
            query_without_quotes = query_without_quotes.replace(phrase, " ")

        terms: List[str] = []
        for term in self._term_pattern.findall(query_without_quotes):
            normalized = term.strip().strip(".,!?;:()[]{}")
            if not normalized:
                continue
            for token in self._segment_term(normalized):
                if token.lower() in self._stop_words:
                    continue
                terms.append(token)
        return self._dedupe(terms)

    def _segment_term(self, term: str) -> List[str]:
        if not re.search(r"[\u4e00-\u9fff]", term):
            return [term]
        try:
            import jieba

            return [item.strip() for item in jieba.lcut(term) if item.strip()]
        except Exception:
            return [term]

    def _build_keyword_query(
        self,
        normalized_query: str,
        quoted_phrases: List[str],
        key_terms: List[str],
    ) -> str:
        parts = [*quoted_phrases, *key_terms]
        if not parts:
            return normalized_query
        return " ".join(parts)

    def _dedupe(self, values: Iterable[str]) -> List[str]:
        seen: Set[str] = set()
        deduped: List[str] = []
        for value in values:
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(value)
        return deduped
