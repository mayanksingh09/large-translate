"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod


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
