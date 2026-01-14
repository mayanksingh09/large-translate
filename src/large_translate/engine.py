"""Translation engine - orchestrates the translation process."""

import asyncio
from pathlib import Path

from rich.progress import Progress, TaskID
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from .chunking import Chunk, ChunkingStrategy, ContextManager
from .models.base import BaseLLMProvider
from .parsers.base import BaseParser, TextSegment


class TranslationEngine:
    """Main translation orchestrator."""

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        parser: BaseParser,
        chunk_size: int = 4000,
    ):
        self.llm = llm_provider
        self.parser = parser
        self.chunker = ChunkingStrategy(max_tokens=chunk_size)
        self.context_manager = ContextManager()

    async def translate_file(
        self,
        input_path: Path,
        output_path: Path,
        target_language: str,
        source_language: str | None = None,
        progress: Progress | None = None,
        verbose: bool = False,
    ) -> dict:
        """
        Translate a file end-to-end.

        Returns:
            Dictionary with translation statistics.
        """
        # Parse input file
        segments = self.parser.parse(input_path)

        if verbose and progress:
            progress.console.print(f"Parsed {len(segments)} segments from input file")

        # Create chunks
        chunks = self.chunker.chunk_segments(segments, self.llm.count_tokens)

        if verbose and progress:
            progress.console.print(f"Created {len(chunks)} chunks for translation")

        # Set up progress tracking
        task_id: TaskID | None = None
        if progress:
            task_id = progress.add_task(
                f"Translating to {target_language}",
                total=len(chunks),
            )

        # Translate each chunk
        translated_segments: list[TextSegment] = []
        total_input_chars = 0
        total_output_chars = 0

        for chunk in chunks:
            context = self.context_manager.get_context(chunk.chunk_index)

            translated_chunk = await self._translate_chunk(
                chunk=chunk,
                target_language=target_language,
                source_language=source_language,
                context=context,
            )

            translated_segments.extend(translated_chunk)

            # Track stats
            for seg in chunk.segments:
                total_input_chars += len(seg.text)
            for seg in translated_chunk:
                total_output_chars += len(seg.text)

            # Update progress
            if progress and task_id is not None:
                progress.advance(task_id)

        # Write output
        self.parser.write(translated_segments, output_path, input_path)

        return {
            "input_chars": total_input_chars,
            "output_chars": total_output_chars,
            "chunks": len(chunks),
            "segments": len(translated_segments),
        }

    @retry(
        wait=wait_exponential_jitter(initial=1, max=60, jitter=5),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def _translate_chunk(
        self,
        chunk: Chunk,
        target_language: str,
        source_language: str | None,
        context: str | None,
    ) -> list[TextSegment]:
        """Translate a single chunk with retry logic."""
        # Separate translatable and non-translatable segments
        translatable_segments = [s for s in chunk.segments if not s.skip_translation]
        non_translatable = {
            i: s for i, s in enumerate(chunk.segments) if s.skip_translation
        }

        if not translatable_segments:
            # All segments are non-translatable
            return list(chunk.segments)

        # Combine segment texts for translation
        combined_text = "\n\n".join(s.text for s in translatable_segments)

        # Translate
        translated_text = await self.llm.translate(
            text=combined_text,
            target_language=target_language,
            source_language=source_language,
            context=context,
        )

        # Store for context
        self.context_manager.add_translation(translated_text)

        # Split back into segments
        translated_parts = translated_text.split("\n\n")

        # Reconstruct result maintaining order
        result = []
        trans_idx = 0

        for i, segment in enumerate(chunk.segments):
            if i in non_translatable:
                result.append(segment)
            else:
                translated_segment = TextSegment(
                    text=(
                        translated_parts[trans_idx]
                        if trans_idx < len(translated_parts)
                        else ""
                    ),
                    metadata=segment.metadata.copy(),
                    segment_type=segment.segment_type,
                )
                result.append(translated_segment)
                trans_idx += 1

        return result
