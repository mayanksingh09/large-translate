"""Markdown file parser with structure preservation."""

import re
from pathlib import Path

from .base import BaseParser, TextSegment


class MarkdownParser(BaseParser):
    """Parser for Markdown files that preserves structure."""

    # Pattern for code blocks (fenced)
    CODE_BLOCK_PATTERN = re.compile(r"^```[\w]*\n[\s\S]*?^```", re.MULTILINE)

    # Pattern for inline code
    INLINE_CODE_PATTERN = re.compile(r"`[^`]+`")

    # Pattern for links [text](url)
    LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

    # Pattern for images ![alt](url)
    IMAGE_PATTERN = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

    @property
    def supported_extensions(self) -> list[str]:
        return [".md"]

    def parse(self, file_path: Path) -> list[TextSegment]:
        content = file_path.read_text(encoding="utf-8")
        segments = []

        # First, extract and replace code blocks with placeholders
        code_blocks = []
        placeholder_prefix = "___CODE_BLOCK_"

        def replace_code_block(match):
            idx = len(code_blocks)
            code_blocks.append(match.group(0))
            return f"{placeholder_prefix}{idx}___"

        content_with_placeholders = self.CODE_BLOCK_PATTERN.sub(
            replace_code_block, content
        )

        # Split by double newlines to identify blocks
        blocks = content_with_placeholders.split("\n\n")

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            # Check if this is a code block placeholder
            if block.startswith(placeholder_prefix) and block.endswith("___"):
                try:
                    idx = int(block[len(placeholder_prefix) : -3])
                    segments.append(
                        TextSegment(
                            text=code_blocks[idx],
                            metadata={"original": code_blocks[idx]},
                            segment_type="code_block",
                            skip_translation=True,
                        )
                    )
                except (ValueError, IndexError):
                    segments.append(
                        TextSegment(text=block, segment_type="paragraph")
                    )
                continue

            # Detect block type and store metadata
            segment_type = self._detect_type(block)
            metadata = {
                "original_format": block,
                "has_links": bool(self.LINK_PATTERN.search(block)),
                "has_images": bool(self.IMAGE_PATTERN.search(block)),
                "has_inline_code": bool(self.INLINE_CODE_PATTERN.search(block)),
            }

            segments.append(
                TextSegment(
                    text=block,
                    metadata=metadata,
                    segment_type=segment_type,
                )
            )

        return segments

    def _detect_type(self, block: str) -> str:
        """Detect the type of markdown block."""
        first_line = block.split("\n")[0]

        if first_line.startswith("#"):
            return "heading"
        if first_line.startswith(">"):
            return "blockquote"
        if re.match(r"^\s*[-*+]\s+", first_line):
            return "unordered_list"
        if re.match(r"^\s*\d+\.\s+", first_line):
            return "ordered_list"
        if first_line.startswith("|"):
            return "table"

        return "paragraph"

    def write(
        self,
        segments: list[TextSegment],
        output_path: Path,
        template_path: Path | None = None,
    ) -> None:
        output_parts = []

        for segment in segments:
            if segment.skip_translation:
                # Keep code blocks unchanged
                output_parts.append(segment.metadata.get("original", segment.text))
            else:
                output_parts.append(segment.text)

        output_path.write_text("\n\n".join(output_parts), encoding="utf-8")
