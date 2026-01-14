"""Chunking strategy and context management for translation."""

from dataclasses import dataclass, field
from typing import Callable

from .parsers.base import TextSegment


@dataclass
class Chunk:
    """A chunk of content ready for translation."""

    segments: list[TextSegment]
    token_count: int
    chunk_index: int
    total_chunks: int = 0


class ChunkingStrategy:
    """Intelligent chunking strategy for translation."""

    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens

    def chunk_segments(
        self,
        segments: list[TextSegment],
        token_counter: Callable[[str], int],
    ) -> list[Chunk]:
        """
        Chunk segments intelligently:
        1. Never split mid-paragraph
        2. Keep related content together
        3. Handle large segments by splitting at sentence boundaries
        """
        chunks = []
        current_segments: list[TextSegment] = []
        current_tokens = 0

        for segment in segments:
            # Skip segments that shouldn't be translated (like code blocks)
            if segment.skip_translation:
                # Add to current chunk as-is
                current_segments.append(segment)
                continue

            segment_tokens = token_counter(segment.text)

            # If single segment exceeds max, split by sentences
            if segment_tokens > self.max_tokens:
                # Flush current chunk first
                if current_segments:
                    chunks.append(
                        Chunk(
                            segments=current_segments,
                            token_count=current_tokens,
                            chunk_index=len(chunks),
                        )
                    )
                    current_segments = []
                    current_tokens = 0

                # Split large segment by sentences
                sub_segments = self._split_large_segment(segment, token_counter)
                for sub in sub_segments:
                    sub_tokens = token_counter(sub.text)
                    chunks.append(
                        Chunk(
                            segments=[sub],
                            token_count=sub_tokens,
                            chunk_index=len(chunks),
                        )
                    )

            elif current_tokens + segment_tokens > self.max_tokens:
                # Flush current chunk and start new one
                if current_segments:
                    chunks.append(
                        Chunk(
                            segments=current_segments,
                            token_count=current_tokens,
                            chunk_index=len(chunks),
                        )
                    )
                current_segments = [segment]
                current_tokens = segment_tokens

            else:
                current_segments.append(segment)
                current_tokens += segment_tokens

        # Don't forget the last chunk
        if current_segments:
            chunks.append(
                Chunk(
                    segments=current_segments,
                    token_count=current_tokens,
                    chunk_index=len(chunks),
                )
            )

        # Update total_chunks for all chunks
        total = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total

        return chunks

    def _split_large_segment(
        self,
        segment: TextSegment,
        token_counter: Callable[[str], int],
    ) -> list[TextSegment]:
        """Split a large segment by sentences."""
        import re

        sentences = re.split(r"(?<=[.!?])\s+", segment.text)

        sub_segments = []
        current_text = ""

        for sentence in sentences:
            potential_text = (
                f"{current_text} {sentence}".strip() if current_text else sentence
            )

            if token_counter(potential_text) > self.max_tokens:
                if current_text:
                    sub_segments.append(
                        TextSegment(
                            text=current_text.strip(),
                            metadata=segment.metadata.copy(),
                            segment_type=segment.segment_type,
                        )
                    )
                current_text = sentence
            else:
                current_text = potential_text

        if current_text:
            sub_segments.append(
                TextSegment(
                    text=current_text.strip(),
                    metadata=segment.metadata.copy(),
                    segment_type=segment.segment_type,
                )
            )

        return sub_segments


@dataclass
class ContextManager:
    """Manages context continuity across chunks for translation."""

    context_length: int = 200
    previous_translations: list[str] = field(default_factory=list)

    def get_context(self, chunk_index: int) -> str | None:
        """Get context summary from previous chunks."""
        if chunk_index == 0 or not self.previous_translations:
            return None

        # Get text from recent translations
        recent_text = " ".join(self.previous_translations[-2:])

        # Truncate to context length, trying to find a sentence boundary
        if len(recent_text) > self.context_length:
            truncated = recent_text[-self.context_length :]
            sentence_start = truncated.find(". ")
            if sentence_start != -1:
                truncated = truncated[sentence_start + 2 :]
            recent_text = truncated

        return recent_text

    def add_translation(self, translated_text: str) -> None:
        """Add completed translation to context history."""
        self.previous_translations.append(translated_text)
