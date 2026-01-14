"""Sentence splitting utilities for sentiment analysis."""

import re
from dataclasses import dataclass


@dataclass
class Sentence:
    """A sentence with its position information."""

    text: str
    start_char: int
    end_char: int
    paragraph_index: int


class SentenceSplitter:
    """Split text into sentences using regex-based approach."""

    # Common abbreviations that shouldn't end sentences
    ABBREVIATIONS = {"Mr", "Mrs", "Ms", "Dr", "Prof", "Jr", "Sr", "Inc", "Ltd", "Corp", "etc", "vs", "Fig", "No", "Vol"}

    def __init__(self):
        """Initialize the sentence splitter."""
        # Simple pattern: split on .!? followed by space and capital letter
        self._pattern = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")

    def split(self, text: str) -> list[Sentence]:
        """
        Split text into sentences with position information.

        Args:
            text: The text to split into sentences.

        Returns:
            List of Sentence objects with text and position info.
        """
        sentences = []
        paragraphs = text.split("\n\n")
        char_offset = 0

        for para_idx, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                char_offset += len(paragraph) + 2
                continue

            # Split paragraph into sentences
            para_sentences = self._split_paragraph(paragraph)

            for sent_text in para_sentences:
                sent_text = sent_text.strip()
                if sent_text:
                    # Find position in original text
                    start = text.find(sent_text, char_offset)
                    if start == -1:
                        start = char_offset
                    sentences.append(
                        Sentence(
                            text=sent_text,
                            start_char=start,
                            end_char=start + len(sent_text),
                            paragraph_index=para_idx,
                        )
                    )

            char_offset += len(paragraph) + 2

        return sentences

    def _split_paragraph(self, paragraph: str) -> list[str]:
        """Split a single paragraph into sentences."""
        # Use the pattern to split
        parts = self._pattern.split(paragraph)

        # Merge back sentences that were incorrectly split after abbreviations
        sentences = []
        for part in parts:
            part = part.strip()
            if not part:
                continue

            # Check if previous sentence ended with an abbreviation
            if sentences:
                prev = sentences[-1]
                # Check if previous sentence ends with abbreviation + period
                words = prev.rstrip(".").split()
                if words and words[-1] in self.ABBREVIATIONS:
                    # Merge with previous
                    sentences[-1] = f"{prev} {part}"
                    continue

            sentences.append(part)

        # If no splits occurred, return the whole paragraph as one sentence
        if not sentences and paragraph.strip():
            sentences = [paragraph.strip()]

        return sentences

    def split_simple(self, text: str) -> list[str]:
        """
        Simple split returning just sentence strings.

        Args:
            text: The text to split.

        Returns:
            List of sentence strings.
        """
        return [s.text for s in self.split(text)]
