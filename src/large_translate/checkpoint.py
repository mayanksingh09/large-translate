"""Checkpoint management for fault-tolerant translation."""

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class CheckpointData:
    """Data stored in a checkpoint file."""

    engine_type: str  # "real-time" or "batch"
    input_path: str
    output_path: str
    target_language: str
    source_language: str | None
    chunk_size: int
    last_completed_chunk: int
    total_chunks: int
    translated_segments: list[dict[str, Any]]  # Serialized TextSegments
    context_history: list[str]  # For context continuity
    # Batch-specific fields
    batch_id: str | None = None
    batch_stage: str | None = None  # "submitted", "polling", "results_fetched"
    # Chunk mapping for batch mode
    chunk_mapping: dict[str, Any] | None = None


class CheckpointManager:
    """Manages checkpoint files for translation recovery."""

    def __init__(self, output_path: Path):
        """
        Initialize checkpoint manager.

        Args:
            output_path: Path to the output file (checkpoint stored alongside).
        """
        self.output_path = output_path
        self.checkpoint_path = output_path.parent / f"{output_path.stem}.checkpoint.json"

    def save(self, data: CheckpointData) -> None:
        """
        Save checkpoint data atomically.

        Uses write-to-temp-then-rename pattern to prevent corruption.
        """
        checkpoint_dict = asdict(data)

        # Write to temporary file first
        fd, temp_path = tempfile.mkstemp(
            suffix=".json",
            prefix=".checkpoint_",
            dir=self.checkpoint_path.parent,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(checkpoint_dict, f, ensure_ascii=False, indent=2)

            # Atomic rename
            os.replace(temp_path, self.checkpoint_path)
        except Exception:
            # Clean up temp file on failure
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

    def load(self) -> CheckpointData | None:
        """
        Load checkpoint if it exists.

        Returns:
            CheckpointData if checkpoint exists and is valid, None otherwise.
        """
        if not self.exists():
            return None

        try:
            with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return CheckpointData(**data)
        except (json.JSONDecodeError, TypeError, KeyError):
            # Invalid checkpoint file
            return None

    def exists(self) -> bool:
        """Check if a checkpoint file exists."""
        return self.checkpoint_path.exists()

    def clean(self) -> None:
        """Remove checkpoint file after successful completion."""
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()


def serialize_segment(segment: Any) -> dict[str, Any]:
    """Serialize a TextSegment to a dictionary."""
    return {
        "text": segment.text,
        "metadata": segment.metadata,
        "segment_type": segment.segment_type,
        "skip_translation": segment.skip_translation,
    }


def deserialize_segment(data: dict[str, Any]) -> dict[str, Any]:
    """
    Deserialize a segment dictionary.

    Returns dict that can be passed to TextSegment constructor.
    """
    return {
        "text": data["text"],
        "metadata": data.get("metadata", {}),
        "segment_type": data.get("segment_type", "paragraph"),
        "skip_translation": data.get("skip_translation", False),
    }
