"""Abstract base class for file parsers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TextSegment:
    """A segment of text with associated metadata for reconstruction."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    segment_type: str = "paragraph"
    skip_translation: bool = False


class BaseParser(ABC):
    """Abstract base class for file parsers."""

    @property
    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """List of supported file extensions."""
        pass

    @abstractmethod
    def parse(self, file_path: Path) -> list[TextSegment]:
        """
        Parse file into segments.

        Args:
            file_path: Path to the input file.

        Returns:
            List of text segments.
        """
        pass

    @abstractmethod
    def write(
        self,
        segments: list[TextSegment],
        output_path: Path,
        template_path: Path | None = None,
    ) -> None:
        """
        Write segments back to file.

        Args:
            segments: List of translated text segments.
            output_path: Path to write the output file.
            template_path: Original file to use as template (for preserving formatting).
        """
        pass
