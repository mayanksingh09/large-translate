"""Plain text file parser."""

from pathlib import Path

from .base import BaseParser, TextSegment


class TxtParser(BaseParser):
    """Parser for plain text files."""

    @property
    def supported_extensions(self) -> list[str]:
        return [".txt"]

    def parse(self, file_path: Path) -> list[TextSegment]:
        content = file_path.read_text(encoding="utf-8")

        # Split by double newlines to identify paragraphs
        paragraphs = content.split("\n\n")

        segments = []
        for para in paragraphs:
            para = para.strip()
            if para:
                segments.append(
                    TextSegment(
                        text=para,
                        metadata={},
                        segment_type="paragraph",
                    )
                )

        return segments

    def write(
        self,
        segments: list[TextSegment],
        output_path: Path,
        template_path: Path | None = None,
    ) -> None:
        output_parts = [seg.text for seg in segments]
        output_path.write_text("\n\n".join(output_parts), encoding="utf-8")
