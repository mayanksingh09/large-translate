"""Sentiment analysis engine - orchestrates the sentiment analysis process."""

import json
from dataclasses import dataclass
from pathlib import Path

from rich.progress import Progress, TaskID
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from .models.base import BaseLLMProvider
from .parsers.base import BaseParser
from .sentence_splitter import Sentence, SentenceSplitter


@dataclass
class SentimentResult:
    """Result of sentiment analysis for a sentence."""

    sentence: str
    label: str
    paragraph_index: int = 0


@dataclass
class SentimentOutput:
    """Complete output of sentiment analysis."""

    source_file: str
    labels: list[str]
    total_sentences: int
    results: list[SentimentResult]
    label_counts: dict[str, int]


class SentimentEngine:
    """Main sentiment analysis orchestrator."""

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        parser: BaseParser,
        batch_size: int = 20,
    ):
        """
        Initialize the sentiment engine.

        Args:
            llm_provider: The LLM provider to use for analysis.
            parser: The file parser to use.
            batch_size: Number of sentences per API call.
        """
        self.llm = llm_provider
        self.parser = parser
        self.splitter = SentenceSplitter()
        self.batch_size = batch_size

    async def analyze_file(
        self,
        input_path: Path,
        output_path: Path,
        labels: list[str],
        progress: Progress | None = None,
        verbose: bool = False,
    ) -> dict:
        """
        Analyze sentiment of a file and write JSON output.

        Args:
            input_path: Path to the input file.
            output_path: Path for the output JSON file.
            labels: List of sentiment labels to use.
            progress: Optional progress bar.
            verbose: Enable verbose output.

        Returns:
            Dictionary with analysis statistics.
        """
        # Parse input file
        segments = self.parser.parse(input_path)

        # Extract text and split into sentences
        full_text = "\n\n".join(s.text for s in segments if not s.skip_translation)
        sentences = self.splitter.split(full_text)

        if verbose and progress:
            progress.console.print(f"Found {len(sentences)} sentences to analyze")

        # Batch sentences for API calls
        sentence_batches = [
            sentences[i : i + self.batch_size]
            for i in range(0, len(sentences), self.batch_size)
        ]

        task_id: TaskID | None = None
        if progress:
            task_id = progress.add_task(
                "Analyzing sentiment",
                total=len(sentence_batches),
            )

        all_results: list[SentimentResult] = []

        for batch in sentence_batches:
            batch_results = await self._analyze_batch(
                sentences=batch,
                labels=labels,
            )
            all_results.extend(batch_results)

            if progress and task_id is not None:
                progress.advance(task_id)

        # Build output
        label_counts = {label: 0 for label in labels}
        for result in all_results:
            if result.label in label_counts:
                label_counts[result.label] += 1

        output = SentimentOutput(
            source_file=str(input_path),
            labels=labels,
            total_sentences=len(sentences),
            results=all_results,
            label_counts=label_counts,
        )

        # Write JSON output
        self._write_output(output, output_path)

        return {
            "sentences": len(sentences),
            "batches": len(sentence_batches),
            "label_counts": label_counts,
        }

    @retry(
        wait=wait_exponential_jitter(initial=1, max=60, jitter=5),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def _analyze_batch(
        self,
        sentences: list[Sentence],
        labels: list[str],
    ) -> list[SentimentResult]:
        """Analyze a batch of sentences with retry logic."""
        sentence_texts = [s.text for s in sentences]

        response = await self.llm.analyze_sentiment(
            sentences=sentence_texts,
            labels=labels,
        )

        # Parse JSON response
        try:
            data = json.loads(response)
            results = []
            for i, item in enumerate(data.get("results", [])):
                sent_obj = sentences[i] if i < len(sentences) else sentences[-1]
                results.append(
                    SentimentResult(
                        sentence=item.get("sentence", sentence_texts[i] if i < len(sentence_texts) else ""),
                        label=item.get("label", labels[0]),
                        paragraph_index=sent_obj.paragraph_index,
                    )
                )
            return results
        except json.JSONDecodeError:
            # Fallback: return default label for all sentences
            return [
                SentimentResult(
                    sentence=s.text,
                    label=labels[0],
                    paragraph_index=s.paragraph_index,
                )
                for s in sentences
            ]

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
