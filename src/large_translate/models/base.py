"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class BatchRequest:
    """A single request in a batch job."""

    custom_id: str
    text: str
    target_language: str
    source_language: str | None = None


@dataclass
class BatchStatus:
    """Status of a batch job."""

    status: str  # "processing", "completed", "failed", "canceling"
    completed: int
    total: int


@dataclass
class BatchResult:
    """Result of a single request in a batch job."""

    custom_id: str
    translated_text: str | None
    error: str | None = None


@dataclass
class SentimentBatchRequest:
    """A single sentiment analysis request in a batch job."""

    custom_id: str
    sentences: list[str]
    labels: list[str]


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Model identifier."""
        pass

    @property
    @abstractmethod
    def max_context_tokens(self) -> int:
        """Maximum context window size in tokens."""
        pass

    @property
    @abstractmethod
    def max_output_tokens(self) -> int:
        """Maximum output tokens per request."""
        pass

    @abstractmethod
    async def translate(
        self,
        text: str,
        target_language: str,
        source_language: str | None = None,
        context: str | None = None,
    ) -> str:
        """
        Translate text to target language.

        Args:
            text: The text to translate.
            target_language: The language to translate to.
            source_language: The source language (auto-detect if None).
            context: Previous context for continuity.

        Returns:
            The translated text.
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        pass

    # Batch inference methods

    @abstractmethod
    async def create_batch(self, requests: list[BatchRequest]) -> str:
        """
        Create a batch job for multiple translation requests.

        Args:
            requests: List of batch requests to process.

        Returns:
            The batch job ID.
        """
        pass

    @abstractmethod
    async def get_batch_status(self, batch_id: str) -> BatchStatus:
        """
        Get the status of a batch job.

        Args:
            batch_id: The batch job ID.

        Returns:
            The current status of the batch job.
        """
        pass

    @abstractmethod
    async def get_batch_results(self, batch_id: str) -> list[BatchResult]:
        """
        Get the results of a completed batch job.

        Args:
            batch_id: The batch job ID.

        Returns:
            List of results for each request in the batch.
        """
        pass

    # Sentiment analysis methods

    @abstractmethod
    async def analyze_sentiment(
        self,
        sentences: list[str],
        labels: list[str],
    ) -> str:
        """
        Analyze sentiment of sentences.

        Args:
            sentences: List of sentences to analyze.
            labels: List of valid sentiment labels.

        Returns:
            JSON string with sentiment results.
        """
        pass

    @abstractmethod
    async def create_sentiment_batch(
        self,
        requests: list[SentimentBatchRequest],
    ) -> str:
        """
        Create a batch job for sentiment analysis requests.

        Args:
            requests: List of sentiment batch requests.

        Returns:
            The batch job ID.
        """
        pass
