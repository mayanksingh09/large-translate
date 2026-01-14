"""CLI entry point for the translation tool."""

import asyncio
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from .batch_engine import BatchTranslationEngine
from .batch_sentiment_engine import BatchSentimentEngine
from .config import get_settings
from .engine import TranslationEngine
from .models import AnthropicProvider, GoogleProvider, OpenAIProvider
from .parsers import DocxParser, MarkdownParser, TxtParser
from .sentiment_engine import SentimentEngine

app = typer.Typer(
    name="translate",
    help="Translate large files using LLM APIs with intelligent chunking",
    no_args_is_help=True,
)

console = Console()


class ModelProvider(str, Enum):
    """Available LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


# Map extensions to parsers
PARSERS = {
    ".txt": TxtParser,
    ".md": MarkdownParser,
    ".docx": DocxParser,
}

# Map providers to implementations
PROVIDERS = {
    ModelProvider.OPENAI: OpenAIProvider,
    ModelProvider.ANTHROPIC: AnthropicProvider,
    ModelProvider.GOOGLE: GoogleProvider,
}


def get_parser(file_path: Path):
    """Get the appropriate parser for a file."""
    ext = file_path.suffix.lower()
    if ext not in PARSERS:
        raise typer.BadParameter(
            f"Unsupported file type: {ext}. Supported: {', '.join(PARSERS.keys())}"
        )
    return PARSERS[ext]()


def get_provider(model: ModelProvider):
    """Get the LLM provider instance."""
    return PROVIDERS[model]()


def create_progress() -> Progress:
    """Create a configured progress bar."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    )


@app.command()
def translate(
    input_file: Path = typer.Argument(
        ...,
        help="Input file to translate (.txt, .docx, .md)",
        exists=True,
        readable=True,
    ),
    target_language: str = typer.Argument(
        ...,
        help="Target language (e.g., 'Spanish', 'French', 'Japanese')",
    ),
    output_file: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path (default: input_<lang>.ext)",
    ),
    model: ModelProvider = typer.Option(
        ModelProvider.OPENAI,
        "--model",
        "-m",
        help="LLM provider to use",
    ),
    source_language: str = typer.Option(
        None,
        "--source",
        "-s",
        help="Source language (auto-detect if not specified)",
    ),
    chunk_size: int = typer.Option(
        4000,
        "--chunk-size",
        "-c",
        help="Maximum tokens per chunk",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
    batch: bool = typer.Option(
        False,
        "--batch",
        "-b",
        help="Use batch API (50% cost reduction, 24h turnaround)",
    ),
    poll_interval: int = typer.Option(
        60,
        "--poll-interval",
        help="Seconds between batch status checks (batch mode only)",
    ),
):
    """Translate a file to the target language."""
    # Validate input file
    parser = get_parser(input_file)

    # Generate output path if not specified
    if output_file is None:
        lang_slug = target_language.lower().replace(" ", "_")[:10]
        output_file = input_file.with_stem(f"{input_file.stem}_{lang_slug}")

    # Get LLM provider
    try:
        provider = get_provider(model)
    except Exception as e:
        console.print(f"[red]Error initializing {model.value} provider: {e}[/red]")
        raise typer.Exit(1)

    # Run translation
    console.print(f"[bold]Translating[/bold] {input_file.name}")
    console.print(f"  Target: {target_language}")
    console.print(f"  Model: {model.value} ({provider.model_id})")
    console.print(f"  Mode: {'Batch (50% cost, 24h turnaround)' if batch else 'Real-time'}")
    console.print(f"  Output: {output_file}")
    console.print()

    try:
        if batch:
            # Use batch translation engine
            engine = BatchTranslationEngine(
                llm_provider=provider,
                parser=parser,
                chunk_size=chunk_size,
            )
            with create_progress() as progress:
                stats = asyncio.run(
                    engine.translate_file_batch(
                        input_path=input_file,
                        output_path=output_file,
                        target_language=target_language,
                        source_language=source_language,
                        poll_interval=poll_interval,
                        progress=progress,
                        verbose=verbose,
                    )
                )

            console.print()
            if stats.get("resumed"):
                console.print("[green]Batch translation complete (resumed from checkpoint)![/green]")
            else:
                console.print("[green]Batch translation complete![/green]")
            console.print(f"  Input: {stats['input_chars']:,} characters")
            console.print(f"  Output: {stats['output_chars']:,} characters")
            console.print(f"  Chunks processed: {stats['chunks']}")
            if stats.get("batch_id"):
                console.print(f"  Batch ID: {stats['batch_id']}")
            console.print(f"  Output file: {output_file}")
        else:
            # Use real-time translation engine
            engine = TranslationEngine(
                llm_provider=provider,
                parser=parser,
                chunk_size=chunk_size,
            )
            with create_progress() as progress:
                stats = asyncio.run(
                    engine.translate_file(
                        input_path=input_file,
                        output_path=output_file,
                        target_language=target_language,
                        source_language=source_language,
                        progress=progress,
                        verbose=verbose,
                    )
                )

            console.print()
            if stats.get("resumed"):
                console.print("[green]Translation complete (resumed from checkpoint)![/green]")
            else:
                console.print("[green]Translation complete![/green]")
            console.print(f"  Input: {stats['input_chars']:,} characters")
            console.print(f"  Output: {stats['output_chars']:,} characters")
            console.print(f"  Chunks processed: {stats['chunks']}")
            console.print(f"  Output file: {output_file}")

    except Exception as e:
        console.print(f"[red]Translation failed: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command("validate")
def validate_file(
    input_file: Path = typer.Argument(
        ...,
        help="File to validate",
        exists=True,
        readable=True,
    ),
):
    """Validate that a file can be processed."""
    try:
        parser = get_parser(input_file)
        segments = parser.parse(input_file)
        total_chars = sum(len(s.text) for s in segments)

        console.print(f"[green]File is valid![/green]")
        console.print(f"  Type: {input_file.suffix}")
        console.print(f"  Segments: {len(segments)}")
        console.print(f"  Total characters: {total_chars:,}")

    except Exception as e:
        console.print(f"[red]Validation failed: {e}[/red]")
        raise typer.Exit(1)


@app.command("models")
def list_models():
    """List available translation models."""
    console.print("[bold]Available Models:[/bold]")
    console.print()

    for provider in ModelProvider:
        try:
            impl = PROVIDERS[provider]()
            console.print(f"  [cyan]{provider.value}[/cyan]")
            console.print(f"    Model ID: {impl.model_id}")
            console.print(f"    Max context: {impl.max_context_tokens:,} tokens")
            console.print(f"    Max output: {impl.max_output_tokens:,} tokens")
            console.print()
        except Exception as e:
            console.print(f"  [cyan]{provider.value}[/cyan]")
            console.print(f"    [yellow]Not configured: {e}[/yellow]")
            console.print()


@app.command("sentiment")
def sentiment(
    input_file: Path = typer.Argument(
        ...,
        help="Input file to analyze (.txt, .docx, .md)",
        exists=True,
        readable=True,
    ),
    labels: str = typer.Option(
        "positive,negative,neutral",
        "--labels",
        "-l",
        help="Comma-separated sentiment labels (e.g., 'positive,negative,neutral')",
    ),
    output_file: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Output JSON file path (default: input_sentiment.json)",
    ),
    model: ModelProvider = typer.Option(
        ModelProvider.OPENAI,
        "--model",
        "-m",
        help="LLM provider to use",
    ),
    batch_size: int = typer.Option(
        20,
        "--batch-size",
        "-s",
        help="Sentences per API call",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
    batch: bool = typer.Option(
        False,
        "--batch",
        "-b",
        help="Use batch API (50% cost reduction, 24h turnaround)",
    ),
    poll_interval: int = typer.Option(
        60,
        "--poll-interval",
        help="Seconds between batch status checks (batch mode only)",
    ),
):
    """Analyze sentiment of a file and output JSON results."""
    # Parse labels
    label_list = [lbl.strip() for lbl in labels.split(",") if lbl.strip()]
    if len(label_list) < 2:
        console.print("[red]Error: At least 2 labels are required[/red]")
        raise typer.Exit(1)

    # Validate input file
    parser = get_parser(input_file)

    # Generate output path if not specified
    if output_file is None:
        output_file = input_file.with_suffix(".sentiment.json")

    # Get LLM provider
    try:
        provider = get_provider(model)
    except Exception as e:
        console.print(f"[red]Error initializing {model.value} provider: {e}[/red]")
        raise typer.Exit(1)

    # Display configuration
    console.print(f"[bold]Analyzing Sentiment[/bold] {input_file.name}")
    console.print(f"  Labels: {', '.join(label_list)}")
    console.print(f"  Model: {model.value} ({provider.model_id})")
    console.print(
        f"  Mode: {'Batch (50% cost, 24h turnaround)' if batch else 'Real-time'}"
    )
    console.print(f"  Output: {output_file}")
    console.print()

    try:
        if batch:
            engine = BatchSentimentEngine(
                llm_provider=provider,
                parser=parser,
                batch_size=batch_size,
            )
            with create_progress() as progress:
                stats = asyncio.run(
                    engine.analyze_file_batch(
                        input_path=input_file,
                        output_path=output_file,
                        labels=label_list,
                        poll_interval=poll_interval,
                        progress=progress,
                        verbose=verbose,
                    )
                )
        else:
            engine = SentimentEngine(
                llm_provider=provider,
                parser=parser,
                batch_size=batch_size,
            )
            with create_progress() as progress:
                stats = asyncio.run(
                    engine.analyze_file(
                        input_path=input_file,
                        output_path=output_file,
                        labels=label_list,
                        progress=progress,
                        verbose=verbose,
                    )
                )

        console.print()
        console.print("[green]Sentiment analysis complete![/green]")
        console.print(f"  Sentences analyzed: {stats['sentences']}")
        console.print("  Label distribution:")
        for label, count in stats["label_counts"].items():
            pct = (count / stats["sentences"] * 100) if stats["sentences"] > 0 else 0
            console.print(f"    {label}: {count} ({pct:.1f}%)")
        console.print(f"  Output file: {output_file}")

    except Exception as e:
        console.print(f"[red]Sentiment analysis failed: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
