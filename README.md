# Large Translate

A CLI tool for translating large files using LLM APIs with intelligent chunking.

## Features

- **Multi-provider support**: OpenAI (gpt-5.2), Anthropic (claude-sonnet-4-5), Google (gemini-3-flash-preview)
- **Multiple file formats**: `.txt`, `.docx`, `.md`
- **Intelligent chunking**: Splits large files by paragraphs, never mid-sentence
- **Context continuity**: Maintains translation consistency across chunks
- **Format preservation**: DOCX preserves bold/italic/fonts, Markdown preserves structure
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
```

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

## License

MIT
