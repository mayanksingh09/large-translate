# Large Translate

A CLI tool for translating large files using LLM APIs with intelligent chunking.

## Features

- **Multi-provider support**: OpenAI (gpt-5.2), Anthropic (claude-sonnet-4-5), Google (gemini-3-flash-preview)
- **Multiple file formats**: `.txt`, `.docx`, `.md`
- **Batch inference**: 50% cost reduction with 24-hour turnaround using batch APIs
- **Intelligent chunking**: Splits large files by paragraphs, never mid-sentence
- **Context continuity**: Maintains translation consistency across chunks
- **Format preservation**: DOCX preserves bold/italic/fonts, Markdown preserves structure
- **Fault tolerance**: Automatic checkpointing with resume on failure
- **Retry logic**: Automatic exponential backoff for API rate limits
- **Progress tracking**: Rich progress bar during translation

## Installation

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
# Clone the repository
git clone <repo-url>
cd large-translate

# Install dependencies
uv sync
```

## Configuration

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AI...
```

You only need to configure the API key for the provider you plan to use.

## Usage

### Basic Translation

```bash
# Translate using OpenAI (default)
uv run translate translate document.txt Spanish

# Translate using Anthropic
uv run translate translate document.docx French --model anthropic

# Translate using Google
uv run translate translate notes.md Japanese --model google
```

### Options

```
uv run translate translate [OPTIONS] INPUT_FILE TARGET_LANGUAGE

Arguments:
  INPUT_FILE       Path to the file to translate (.txt, .docx, .md)
  TARGET_LANGUAGE  Target language (e.g., "Spanish", "French", "Japanese")

Options:
  -o, --output PATH         Output file path (default: input_<lang>.ext)
  -m, --model PROVIDER      LLM provider: openai, anthropic, google (default: openai)
  -s, --source LANGUAGE     Source language (auto-detect if not specified)
  -c, --chunk-size INT      Maximum tokens per chunk (default: 4000)
  -v, --verbose             Enable verbose output
  -b, --batch               Use batch API (50% cost, 24h turnaround)
  --poll-interval INT       Seconds between batch status checks (default: 60)
```

### Batch Mode

For large-scale translations, use batch mode to reduce costs by 50%:

```bash
# Translate using batch API (50% cheaper, completes within 24 hours)
uv run translate translate large_document.txt Spanish --batch

# Batch mode with custom poll interval
uv run translate translate book.docx French --batch --poll-interval 120 -m anthropic
```

Batch mode submits all chunks as a single batch job and polls for completion. This is ideal for:
- Large documents that don't need immediate results
- Cost-sensitive bulk translations
- Non-urgent translation workflows

### Examples

```bash
# Translate a text file to Spanish
uv run translate translate report.txt Spanish

# Translate a Word document to French, specifying output path
uv run translate translate contract.docx French -o contract_fr.docx

# Translate Markdown to German using Anthropic with verbose output
uv run translate translate docs.md German -m anthropic -v

# Translate from English to Japanese using Google
uv run translate translate article.txt Japanese -m google -s English
```

### Other Commands

```bash
# Validate a file can be processed
uv run translate validate document.docx

# List available models and their configuration
uv run translate models
```

## Supported File Types

| Format | Extension | Notes |
|--------|-----------|-------|
| Plain Text | `.txt` | UTF-8 encoded |
| Word Document | `.docx` | Preserves paragraph styles, bold, italic, fonts |
| Markdown | `.md` | Preserves headings, lists, code blocks, links |

## How It Works

1. **Parsing**: The tool reads your file and splits it into segments (paragraphs)
2. **Chunking**: Segments are grouped into chunks that fit within the model's context window
3. **Translation**: Each chunk is sent to the LLM API with context from previous chunks
4. **Reconstruction**: Translated segments are reassembled with original formatting preserved

## Fault Tolerance

The tool automatically saves progress during translation, allowing recovery from failures:

- **Checkpoint files**: Progress is saved to `{output_name}.checkpoint.json` after each chunk
- **Automatic resume**: If translation fails (network error, API limit, crash), simply re-run the same command
- **Validation**: Checkpoints are validated to ensure they match the current job (same input file, target language, chunk count)
- **Cleanup**: Checkpoint files are automatically deleted after successful completion

### Recovery Example

```bash
# Start a translation (interrupted mid-way)
uv run translate translate large_book.docx Spanish
# ... process crashes or network fails ...

# Simply re-run the same command to resume
uv run translate translate large_book.docx Spanish
# Output: "Resuming from checkpoint: 15/30 chunks completed"
```

This is especially useful for:
- Large documents with many chunks
- Unreliable network connections
- Batch translations that may take hours

## License

MIT
