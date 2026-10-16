import re
import uuid
from typing import Any, Dict, List, Tuple

from backend.common.logger import get_logger
from backend.domain.entities.document import DocumentChunk

logger = get_logger(__name__)


class DocumentChunker:
    """Token-aware Markdown chunker with structural metadata and overlap."""

    def __init__(
        self,
        max_tokens: int = 480,
        overlap_tokens: int = 80,
        min_chars: int = 10,
        encoding_name: str = "cl100k_base",
    ):
        self.max_tokens = max_tokens
        self.overlap_tokens = min(overlap_tokens, max_tokens // 2)
        self.min_chars = min_chars
        self.encoder = self._load_encoder(encoding_name)

    def split_markdown(self, text: str, source_name: str = "unknown") -> List[DocumentChunk]:
        try:
            sections = self._split_markdown_sections(text)
            chunks: List[DocumentChunk] = []

            for section_text, section_meta in sections:
                if len(section_text.strip()) < self.min_chars:
                    continue

                for piece in self._split_text_with_overlap(section_text):
                    piece = piece.strip()
                    if len(piece) < self.min_chars:
                        continue

                    metadata = section_meta.copy()
                    metadata["source_file"] = source_name
                    metadata["chunk_index"] = len(chunks)
                    metadata["token_count"] = self.count_tokens(piece)

                    chunks.append(
                        DocumentChunk(
                            id=str(uuid.uuid4()),
                            content=piece,
                            metadata=metadata,
                        )
                    )

            logger.info("Split document [%s] into %s smart chunks", source_name, len(chunks))
            return chunks

        except Exception as exc:
            logger.error("Document chunking failed [%s]: %s", source_name, exc)
            return []

    def count_tokens(self, text: str) -> int:
        if self.encoder:
            return len(self.encoder.encode(text))

        # Conservative fallback: count CJK characters, words, and symbols as token-like units.
        token_like_units = re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\s]", text)
        return len(token_like_units)

    def _load_encoder(self, encoding_name: str):
        try:
            import tiktoken

            return tiktoken.get_encoding(encoding_name)
        except Exception:
            logger.warning("tiktoken is unavailable; SmartChunker will use fallback token estimates")
            return None

    def _split_markdown_sections(self, text: str) -> List[Tuple[str, Dict[str, Any]]]:
        sections: List[Tuple[str, Dict[str, Any]]] = []
        current_lines: List[str] = []
        current_headers: Dict[int, str] = {}
        current_meta: Dict[str, Any] = {}

        for line in text.splitlines():
            header_match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
            if header_match:
                if current_lines:
                    sections.append(("\n".join(current_lines).strip(), current_meta.copy()))
                    current_lines = []

                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                current_headers = {
                    header_level: header_title
                    for header_level, header_title in current_headers.items()
                    if header_level < level
                }
                current_headers[level] = title
                current_meta = self._headers_to_metadata(current_headers)

            current_lines.append(line)

        if current_lines:
            sections.append(("\n".join(current_lines).strip(), current_meta.copy()))

        return sections

    def _headers_to_metadata(self, headers: Dict[int, str]) -> Dict[str, Any]:
        metadata = {f"Header {level}": title for level, title in sorted(headers.items())}
        metadata["heading_path"] = " > ".join(title for _, title in sorted(headers.items()))
        return metadata

    def _split_text_with_overlap(self, text: str) -> List[str]:
        if self.count_tokens(text) <= self.max_tokens:
            return [text]

        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        if not paragraphs:
            return self._split_long_text(text)

        chunks: List[str] = []
        current = ""

        for paragraph in paragraphs:
            if self.count_tokens(paragraph) > self.max_tokens:
                combined_text = f"{current}\n\n{paragraph}" if current else paragraph
                long_parts = self._split_long_text(combined_text)
                chunks.extend(long_parts[:-1])
                current = long_parts[-1]
                continue

            candidate = paragraph if not current else f"{current}\n\n{paragraph}"
            if self.count_tokens(candidate) <= self.max_tokens:
                current = candidate
                continue

            if current:
                chunks.append(current.strip())
                overlap = self._tail_overlap(current)
                current = f"{overlap}\n\n{paragraph}" if overlap else paragraph
            else:
                current = paragraph

        if current:
            if self.count_tokens(current) <= self.max_tokens:
                chunks.append(current.strip())
            else:
                chunks.extend(self._split_long_text(current))

        return chunks

    def _split_long_text(self, text: str) -> List[str]:
        if self.encoder:
            token_ids = self.encoder.encode(text)
            chunks = []
            start = 0
            step = self.max_tokens - self.overlap_tokens

            while start < len(token_ids):
                end = min(start + self.max_tokens, len(token_ids))
                chunks.append(self.encoder.decode(token_ids[start:end]).strip())
                if end == len(token_ids):
                    break
                start += step

            return chunks

        max_chars = self.max_tokens * 3
        overlap_chars = self.overlap_tokens * 3
        chunks = []
        start = 0
        step = max_chars - overlap_chars

        while start < len(text):
            end = min(start + max_chars, len(text))
            chunks.append(text[start:end].strip())
            if end == len(text):
                break
            start += step

        return chunks

    def _tail_overlap(self, text: str) -> str:
        if not text.strip() or self.overlap_tokens <= 0:
            return ""

        if self.encoder:
            token_ids = self.encoder.encode(text)
            return self.encoder.decode(token_ids[-self.overlap_tokens :]).strip()

        overlap_chars = self.overlap_tokens * 3
        return text[-overlap_chars:].strip()
