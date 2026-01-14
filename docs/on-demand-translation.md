# On-Demand Translation API with Chunking

This document explains how the large-translate on-demand translation API works, with a focus on the intelligent chunking mechanism that enables translation of large files.

## Overview

The on-demand translation API provides real-time translation of documents using LLM providers. It handles large files by intelligently splitting content into manageable chunks while preserving context and formatting.

### Supported File Formats
- **Plain Text** (`.txt`) - Simple paragraph-based splitting
- **Markdown** (`.md`) - Preserves headers, code blocks, links, and formatting
- **Word Documents** (`.docx`) - Maintains styles, fonts, and run formatting

### Supported LLM Providers
| Provider | Model | Max Context | Max Output |
|----------|-------|-------------|------------|
| OpenAI | gpt-5.2 | 128,000 tokens | 16,384 tokens |
| Anthropic | claude-sonnet-4-5 | 200,000 tokens | 8,192 tokens |
| Google | gemini-3-flash-preview | 1,000,000 tokens | 8,192 tokens |

## High-Level Architecture

```mermaid
flowchart TB
    subgraph Input
        A[Input File<br/>txt/md/docx]
    end

    subgraph Parsing
        B[Parser]
        C[TextSegments<br/>with metadata]
    end

    subgraph Chunking
        D[ChunkingStrategy]
        E[Chunks<br/>token-bounded]
    end

    subgraph Translation
        F[TranslationEngine]
        G[ContextManager]
        H[LLM Provider]
    end

    subgraph Output
        I[Reconstructed<br/>Segments]
        J[Output File]
    end

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    G <-.-> F
    F <--> H
    F --> I
    I --> J
```

## Chunking Strategy

### Why Chunking is Needed

LLM providers have token limits for both input and output. A large document might exceed these limits, causing API failures. The chunking strategy:

1. **Respects token limits** - Each chunk stays within the configured `max_tokens`
2. **Preserves paragraph boundaries** - Never splits mid-paragraph
3. **Handles oversized content** - Splits large segments by sentences
4. **Maintains context** - Passes previous translation context between chunks

### How Chunking Works

```mermaid
flowchart TD
    subgraph Input["Input: TextSegments"]
        S1[Segment 1<br/>500 tokens]
        S2[Segment 2<br/>800 tokens]
        S3[Segment 3<br/>6000 tokens]
        S4[Segment 4<br/>300 tokens]
        S5[Segment 5<br/>400 tokens]
    end

    subgraph Process["Chunking Process"]
        direction TB
        P1{Fits in<br/>current chunk?}
        P2[Add to current chunk]
        P3[Flush chunk]
        P4{Segment > max_tokens?}
        P5[Split by sentences]
        P6[Create individual chunks]
    end

    subgraph Output["Output: Chunks"]
        C1[Chunk 1<br/>S1 + S2<br/>1300 tokens]
        C2[Chunk 2<br/>S3 part 1<br/>4000 tokens]
        C3[Chunk 3<br/>S3 part 2<br/>2000 tokens]
        C4[Chunk 4<br/>S4 + S5<br/>700 tokens]
    end

    S1 --> P1
    S2 --> P1
    S3 --> P1
    S4 --> P1
    S5 --> P1

    P1 -->|Yes| P2
    P1 -->|No| P3
    P3 --> P4
    P4 -->|Yes| P5
    P5 --> P6
    P4 -->|No| P2

    P2 --> C1
    P6 --> C2
    P6 --> C3
    P2 --> C4
```

### Chunking Rules

1. **Accumulation**: Segments are accumulated until adding another would exceed `max_tokens`
2. **Overflow Handling**: When limit is reached, current chunk is flushed and a new one starts
3. **Large Segment Splitting**: Segments exceeding `max_tokens` are split by sentence boundaries
4. **Skip Preservation**: Segments marked `skip_translation=True` (e.g., code blocks) are included but not translated

### Token Counting

Each provider implements token counting differently:

| Provider | Method |
|----------|--------|
| OpenAI | Precise counting via `tiktoken` library |
| Anthropic | Estimate: `len(text) // 4` |
| Google | Estimate: `len(text) // 4` |

## Context Preservation

The `ContextManager` maintains translation continuity across chunks by passing context from previous translations.

```mermaid
flowchart LR
    subgraph Chunk1["Chunk 1"]
        T1[Translation 1]
    end

    subgraph Context["Context Manager"]
        CM[Stores last 200 chars<br/>from recent translations]
    end

    subgraph Chunk2["Chunk 2"]
        T2[Translation 2]
    end

    subgraph Chunk3["Chunk 3"]
        T3[Translation 3]
    end

    T1 -->|"add_translation()"| CM
    CM -->|"get_context()"| T2
    T2 -->|"add_translation()"| CM
    CM -->|"get_context()"| T3
```

### How Context Works

1. **First Chunk**: No context provided (starts fresh)
2. **Subsequent Chunks**: Last 200 characters from previous translations are included
3. **Sentence Boundary Respect**: Context is truncated at sentence boundaries for clarity
4. **Prompt Integration**: Context is added as `[Previous context for consistency: ...]`

## Translation Flow

```mermaid
sequenceDiagram
    participant CLI
    participant Parser
    participant Chunker as ChunkingStrategy
    participant Engine as TranslationEngine
    participant Context as ContextManager
    participant LLM as LLM Provider

    CLI->>Parser: parse(input_file)
    Parser-->>CLI: TextSegment[]

    CLI->>Engine: translate_file()
    Engine->>Chunker: chunk_segments(segments)
    Chunker-->>Engine: Chunk[]

    loop For each Chunk
        Engine->>Context: get_context()
        Context-->>Engine: previous_text (200 chars)

        Engine->>Engine: _translate_chunk()
        Note over Engine: Combine translatable segments<br/>with "\n\n" separator

        Engine->>LLM: translate(text, target_lang, context)
        LLM-->>Engine: translated_text

        Note over Engine: Split response by "\n\n"<br/>Reconstruct segments

        Engine->>Context: add_translation(text)
    end

    Engine->>Parser: write(translated_segments)
    Parser-->>CLI: output_file
```

## Chunk Processing Detail

### Segment Separation

When processing a chunk, segments are categorized:

```mermaid
flowchart TD
    subgraph Input["Chunk Segments"]
        S1[Paragraph 1<br/>skip=false]
        S2[Code Block<br/>skip=true]
        S3[Paragraph 2<br/>skip=false]
        S4[Paragraph 3<br/>skip=false]
    end

    subgraph Process["Translation Process"]
        P1[Translatable:<br/>P1, P2, P3]
        P2[Combined with '\n\n']
        P3[Send to LLM]
        P4[Split response by '\n\n']
    end

    subgraph Output["Reconstructed"]
        O1[Translated P1]
        O2[Original Code Block]
        O3[Translated P2]
        O4[Translated P3]
    end

    S1 --> P1
    S3 --> P1
    S4 --> P1
    P1 --> P2
    P2 --> P3
    P3 --> P4

    P4 --> O1
    S2 --> O2
    P4 --> O3
    P4 --> O4
```

### Retry Logic

Translation uses exponential backoff for resilience:

```
Retry Strategy:
- Max attempts: 5
- Backoff: Exponential (wait=1, wait_multiplier=2)
- Handles: Rate limits, temporary failures
```

## Configuration Options

### CLI Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--chunk-size` | 4000 | Maximum tokens per chunk |
| `--model` | openai | LLM provider (openai/anthropic/google) |
| `--source` | auto | Source language (optional) |
| `--verbose` | false | Enable detailed logging |

### Example Usage

```bash
# Basic translation
large-translate translate input.md spanish

# Custom chunk size for large files
large-translate translate large-doc.docx french --chunk-size 8000

# Using Anthropic with verbose output
large-translate translate article.txt german --model anthropic --verbose
```

### Choosing Chunk Size

| Scenario | Recommended Size | Reason |
|----------|------------------|--------|
| Short documents | 4000 (default) | Balanced performance |
| Long documents | 8000-16000 | Fewer API calls |
| Complex content | 2000-4000 | Better context handling |
| Technical docs | 4000 | Preserves code blocks well |

## Data Flow Summary

```mermaid
flowchart LR
    subgraph Stage1["1. Parse"]
        A[File] --> B[Segments]
    end

    subgraph Stage2["2. Chunk"]
        B --> C[Chunks]
    end

    subgraph Stage3["3. Translate"]
        C --> D{For each<br/>chunk}
        D --> E[Get Context]
        E --> F[Call LLM]
        F --> G[Store Result]
        G --> D
    end

    subgraph Stage4["4. Write"]
        G --> H[Reconstruct]
        H --> I[Output File]
    end
```

## Key Components Reference

| Component | File | Purpose |
|-----------|------|---------|
| TranslationEngine | `engine.py` | Orchestrates translation flow |
| ChunkingStrategy | `chunking.py` | Token-aware segmentation |
| ContextManager | `chunking.py` | Cross-chunk continuity |
| BaseLLMProvider | `models/base.py` | Provider interface |
| BaseParser | `parsers/base.py` | Format parsing interface |
| TextSegment | `parsers/base.py` | Data transport object |
