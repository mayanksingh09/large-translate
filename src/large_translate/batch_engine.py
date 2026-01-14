"""Batch translation engine - orchestrates batch translation process."""

import asyncio
from pathlib import Path

from rich.progress import Progress

from .checkpoint import (
    CheckpointData,
    CheckpointManager,
    deserialize_segment,
    serialize_segment,
)
from .chunking import Chunk, ChunkingStrategy
from .models.base import BaseLLMProvider, BatchRequest
from .parsers.base import BaseParser, TextSegment


class BatchTranslationEngine:
    """Batch translation orchestrator using async batch APIs."""

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        parser: BaseParser,
        chunk_size: int = 4000,
    ):
        self.llm = llm_provider
        self.parser = parser
        self.chunker = ChunkingStrategy(max_tokens=chunk_size)

    async def translate_file_batch(
        self,
        input_path: Path,
        output_path: Path,
        target_language: str,
        source_language: str | None = None,
        poll_interval: int = 60,
        progress: Progress | None = None,
        verbose: bool = False,
    ) -> dict:
        """
        Translate a file using batch API.

        Args:
            input_path: Path to the input file.
            output_path: Path to write the translated file.
            target_language: Target language for translation.
            source_language: Source language (auto-detect if None).
            poll_interval: Seconds between status checks.
            progress: Optional Rich progress instance.
            verbose: Enable verbose output.

        Returns:
            Dictionary with translation statistics including 'resumed' if resuming from checkpoint.
        """
        # Initialize checkpoint manager
        checkpoint_mgr = CheckpointManager(output_path)

        # 1. Parse input file
        segments = self.parser.parse(input_path)
        if verbose and progress:
            progress.console.print(f"Parsed {len(segments)} segments from input file")

        # 2. Create chunks
        chunks = self.chunker.chunk_segments(segments, self.llm.count_tokens)
        if verbose and progress:
            progress.console.print(f"Created {len(chunks)} chunks for batch processing")

        # 3. Build batch requests
        batch_requests = []
        chunk_mapping: dict[str, tuple[int, Chunk]] = {}  # custom_id -> (index, chunk)

        for i, chunk in enumerate(chunks):
            # Get translatable text
            translatable = [s for s in chunk.segments if not s.skip_translation]
            if not translatable:
                continue

            combined_text = "\n\n".join(s.text for s in translatable)
            custom_id = f"chunk-{i}"
            batch_requests.append(
                BatchRequest(
                    custom_id=custom_id,
                    text=combined_text,
                    target_language=target_language,
                    source_language=source_language,
                )
            )
            chunk_mapping[custom_id] = (i, chunk)

        if not batch_requests:
            # No translatable content
            self.parser.write(segments, output_path, input_path)
            return {
                "input_chars": sum(len(s.text) for s in segments),
                "output_chars": sum(len(s.text) for s in segments),
                "chunks": 0,
                "batch_id": None,
            }

        # Serialize chunk mapping for checkpoint
        chunk_mapping_serialized = {
            custom_id: {
                "index": idx,
                "segments": [serialize_segment(s) for s in chunk.segments],
            }
            for custom_id, (idx, chunk) in chunk_mapping.items()
        }

        # Check for existing checkpoint
        checkpoint = checkpoint_mgr.load()
        batch_id = None
        resumed = False

        if checkpoint and checkpoint.engine_type == "batch":
            # Validate checkpoint matches current job
            if (
                checkpoint.input_path == str(input_path)
                and checkpoint.target_language == target_language
                and checkpoint.total_chunks == len(chunks)
                and checkpoint.batch_id
            ):
                batch_id = checkpoint.batch_id
                resumed = True
                if progress:
                    progress.console.print(
                        f"[yellow]Resuming from checkpoint: batch {batch_id} (stage: {checkpoint.batch_stage})[/yellow]"
                    )

        # 4. Submit batch (if not resuming)
        if batch_id is None:
            if progress:
                progress.console.print(f"Submitting batch with {len(batch_requests)} requests...")
            batch_id = await self.llm.create_batch(batch_requests)
            if progress:
                progress.console.print(f"Batch created: {batch_id}")

            # CRITICAL: Save checkpoint immediately after getting batch_id
            checkpoint_mgr.save(
                CheckpointData(
                    engine_type="batch",
                    input_path=str(input_path),
                    output_path=str(output_path),
                    target_language=target_language,
                    source_language=source_language,
                    chunk_size=self.chunker.max_tokens,
                    last_completed_chunk=-1,
                    total_chunks=len(chunks),
                    translated_segments=[],
                    context_history=[],
                    batch_id=batch_id,
                    batch_stage="submitted",
                    chunk_mapping=chunk_mapping_serialized,
                )
            )

        # 5. Poll for completion
        task_id = None
        if progress:
            task_id = progress.add_task(
                "Waiting for batch completion",
                total=len(batch_requests),
            )

        while True:
            status = await self.llm.get_batch_status(batch_id)
            if progress and task_id is not None:
                progress.update(task_id, completed=status.completed)

            # Update checkpoint during polling
            checkpoint_mgr.save(
                CheckpointData(
                    engine_type="batch",
                    input_path=str(input_path),
                    output_path=str(output_path),
                    target_language=target_language,
                    source_language=source_language,
                    chunk_size=self.chunker.max_tokens,
                    last_completed_chunk=status.completed - 1,
                    total_chunks=len(chunks),
                    translated_segments=[],
                    context_history=[],
                    batch_id=batch_id,
                    batch_stage="polling",
                    chunk_mapping=chunk_mapping_serialized,
                )
            )

            if status.status == "completed":
                break

            if verbose and progress:
                progress.console.print(
                    f"Status: {status.status} ({status.completed}/{status.total})"
                )

            await asyncio.sleep(poll_interval)

        # 6. Get results
        if progress:
            progress.console.print("Fetching results...")
        results = await self.llm.get_batch_results(batch_id)

        # Update checkpoint after fetching results
        checkpoint_mgr.save(
            CheckpointData(
                engine_type="batch",
                input_path=str(input_path),
                output_path=str(output_path),
                target_language=target_language,
                source_language=source_language,
                chunk_size=self.chunker.max_tokens,
                last_completed_chunk=len(chunks) - 1,
                total_chunks=len(chunks),
                translated_segments=[],
                context_history=[],
                batch_id=batch_id,
                batch_stage="results_fetched",
                chunk_mapping=chunk_mapping_serialized,
            )
        )

        # 7. Build result mapping
        translations: dict[str, str] = {}
        for result in results:
            if result.translated_text:
                translations[result.custom_id] = result.translated_text
            elif result.error:
                if progress:
                    progress.console.print(
                        f"[yellow]Warning: {result.custom_id} failed: {result.error}[/yellow]"
                    )

        # 8. Reconstruct segments
        translated_segments: list[TextSegment] = []
        for i, chunk in enumerate(chunks):
            custom_id = f"chunk-{i}"
            translation = translations.get(custom_id)

            if translation:
                # Split translated text back into segments
                translated_parts = translation.split("\n\n")
                trans_idx = 0

                for segment in chunk.segments:
                    if segment.skip_translation:
                        translated_segments.append(segment)
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
                        translated_segments.append(translated_segment)
                        trans_idx += 1
            else:
                # No translation available, keep original
                translated_segments.extend(chunk.segments)

        # 9. Write output
        self.parser.write(translated_segments, output_path, input_path)

        # 10. Clean up checkpoint on success
        checkpoint_mgr.clean()

        # 11. Calculate stats
        total_input = sum(len(s.text) for s in segments)
        total_output = sum(len(s.text) for s in translated_segments)

        result = {
            "input_chars": total_input,
            "output_chars": total_output,
            "chunks": len(chunks),
            "batch_id": batch_id,
        }
        if resumed:
            result["resumed"] = True
        return result
