"""Batch sentiment analysis engine - uses batch API for cost savings."""

import asyncio
import json
from pathlib import Path

from rich.progress import Progress

from .models.base import BaseLLMProvider, SentimentBatchRequest
from .parsers.base import BaseParser
from .sentence_splitter import SentenceSplitter
from .sentiment_engine import SentimentOutput, SentimentResult


class BatchSentimentEngine:
    """Batch sentiment analysis using async batch APIs."""

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        parser: BaseParser,
        batch_size: int = 20,
    ):
        """
        Initialize the batch sentiment engine.

        Args:
            llm_provider: The LLM provider to use for analysis.
            parser: The file parser to use.
            batch_size: Number of sentences per batch request.
        """
        self.llm = llm_provider
        self.parser = parser
        self.splitter = SentenceSplitter()
        self.batch_size = batch_size

    async def analyze_file_batch(
        self,
        input_path: Path,
        output_path: Path,
        labels: list[str],
        poll_interval: int = 60,
        progress: Progress | None = None,
        verbose: bool = False,
    ) -> dict:
        """
        Analyze sentiment using batch API.

        Args:
            input_path: Path to the input file.
            output_path: Path for the output JSON file.
            labels: List of sentiment labels to use.
            poll_interval: Seconds between status checks.
            progress: Optional Rich progress instance.
            verbose: Enable verbose output.

        Returns:
            Dictionary with analysis statistics.
        """
        # 1. Parse input file and split into sentences
        segments = self.parser.parse(input_path)
        full_text = "\n\n".join(s.text for s in segments if not s.skip_translation)
        sentences = self.splitter.split(full_text)

        if verbose and progress:
            progress.console.print(f"Found {len(sentences)} sentences to analyze")

        # 2. Create sentence batches
        sentence_batches = [
            sentences[i : i + self.batch_size]
            for i in range(0, len(sentences), self.batch_size)
        ]

        if not sentence_batches:
            # No sentences to analyze
            output = SentimentOutput(
                source_file=str(input_path),
                labels=labels,
                total_sentences=0,
                results=[],
                label_counts={label: 0 for label in labels},
            )
            self._write_output(output, output_path)
            return {
                "sentences": 0,
                "batches": 0,
                "batch_id": None,
                "label_counts": output.label_counts,
            }

        # 3. Build batch requests
        batch_requests = []
        for i, batch in enumerate(sentence_batches):
            batch_requests.append(
                SentimentBatchRequest(
                    custom_id=f"sentiment-{i}",
                    sentences=[s.text for s in batch],
                    labels=labels,
                )
            )

        if progress:
            progress.console.print(
                f"Submitting batch with {len(batch_requests)} requests..."
            )

        # 4. Submit batch
        batch_id = await self.llm.create_sentiment_batch(batch_requests)

        if progress:
            progress.console.print(f"Batch created: {batch_id}")

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

        # 7. Parse and reconstruct results
        all_results: list[SentimentResult] = []

        for result in results:
            if result.translated_text:  # reusing BatchResult structure
                try:
                    data = json.loads(result.translated_text)
                    # Extract batch index from custom_id
                    batch_idx = int(result.custom_id.split("-")[1])
                    batch_sentences = sentence_batches[batch_idx]

                    for i, item in enumerate(data.get("results", [])):
                        if i < len(batch_sentences):
                            all_results.append(
                                SentimentResult(
                                    sentence=item.get(
                                        "sentence", batch_sentences[i].text
                                    ),
                                    label=item.get("label", labels[0]),
                                    paragraph_index=batch_sentences[i].paragraph_index,
                                )
                            )
                except (json.JSONDecodeError, IndexError, ValueError) as e:
                    if verbose and progress:
                        progress.console.print(
                            f"[yellow]Warning: Failed to parse result {result.custom_id}: {e}[/yellow]"
                        )
            elif result.error:
                if progress:
                    progress.console.print(
                        f"[yellow]Warning: {result.custom_id} failed: {result.error}[/yellow]"
                    )

        # 8. Build output
        label_counts = {label: 0 for label in labels}
        for r in all_results:
            if r.label in label_counts:
                label_counts[r.label] += 1

        output = SentimentOutput(
            source_file=str(input_path),
            labels=labels,
            total_sentences=len(sentences),
            results=all_results,
            label_counts=label_counts,
        )

        # 9. Write JSON output
        self._write_output(output, output_path)

        return {
            "sentences": len(sentences),
            "batches": len(sentence_batches),
            "batch_id": batch_id,
            "label_counts": label_counts,
        }

    def _write_output(self, output: SentimentOutput, path: Path) -> None:
        """Write sentiment results to JSON file."""
        data = {
            "source_file": output.source_file,
            "labels": output.labels,
            "total_sentences": output.total_sentences,
            "label_counts": output.label_counts,
            "results": [
                {
                    "sentence": r.sentence,
                    "label": r.label,
                    "paragraph_index": r.paragraph_index,
                }
                for r in output.results
            ],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
