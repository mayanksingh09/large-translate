"""Translation engine - orchestrates the translation process."""

import asyncio
from pathlib import Path

from rich.progress import Progress, TaskID
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from .checkpoint import (
    CheckpointData,
    CheckpointManager,
    deserialize_segment,
    serialize_segment,
)
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
            Dictionary with translation statistics including 'resumed' if resuming from checkpoint.
        """
        # Initialize checkpoint manager
        checkpoint_mgr = CheckpointManager(output_path)

        # Parse input file
        segments = self.parser.parse(input_path)

        if verbose and progress:
            progress.console.print(f"Parsed {len(segments)} segments from input file")

        # Create chunks
        chunks = self.chunker.chunk_segments(segments, self.llm.count_tokens)

        if verbose and progress:
            progress.console.print(f"Created {len(chunks)} chunks for translation")

        # Check for existing checkpoint
        checkpoint = checkpoint_mgr.load()
        start_chunk = 0
        translated_segments: list[TextSegment] = []
        resumed = False

        if checkpoint and checkpoint.engine_type == "real-time":
            # Validate checkpoint matches current job
            if (
                checkpoint.input_path == str(input_path)
                and checkpoint.target_language == target_language
                and checkpoint.total_chunks == len(chunks)
            ):
                start_chunk = checkpoint.last_completed_chunk + 1
                # Restore translated segments
                translated_segments = [
                    TextSegment(**deserialize_segment(s))
                    for s in checkpoint.translated_segments
                ]
                # Restore context history
                self.context_manager.previous_translations = checkpoint.context_history
                resumed = True

                if progress:
                    progress.console.print(
                        f"[yellow]Resuming from checkpoint: {start_chunk}/{len(chunks)} chunks completed[/yellow]"
                    )

        # Set up progress tracking
        task_id: TaskID | None = None
        if progress:
            task_id = progress.add_task(
                f"Translating to {target_language}",
                total=len(chunks),
                completed=start_chunk,
            )

        # Calculate stats for already-translated chunks
        total_input_chars = 0
        total_output_chars = 0

        for i, chunk in enumerate(chunks[:start_chunk]):
            for seg in chunk.segments:
                total_input_chars += len(seg.text)
        for seg in translated_segments:
            total_output_chars += len(seg.text)

        # Translate remaining chunks
        for chunk in chunks[start_chunk:]:
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

            # Save checkpoint after each successful chunk
            checkpoint_mgr.save(
                CheckpointData(
                    engine_type="real-time",
                    input_path=str(input_path),
                    output_path=str(output_path),
                    target_language=target_language,
                    source_language=source_language,
                    chunk_size=self.chunker.max_tokens,
                    last_completed_chunk=chunk.chunk_index,
                    total_chunks=len(chunks),
                    translated_segments=[
                        serialize_segment(s) for s in translated_segments
                    ],
                    context_history=self.context_manager.previous_translations,
                )
            )

            # Update progress
            if progress and task_id is not None:
                progress.advance(task_id)

        # Write output
        self.parser.write(translated_segments, output_path, input_path)

        # Clean up checkpoint on success
        checkpoint_mgr.clean()

        result = {
            "input_chars": total_input_chars,
            "output_chars": total_output_chars,
            "chunks": len(chunks),
            "segments": len(translated_segments),
        }
        if resumed:
            result["resumed"] = True
        return result

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
