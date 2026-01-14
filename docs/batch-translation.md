# Batch Translation API

This document explains how the batch translation API works, providing a cost-effective alternative to on-demand translation for processing large volumes of content.

## Overview

The batch translation API allows you to submit multiple translation requests as a single batch job. This approach offers:

- **Cost Savings**: Typically 50% cheaper than on-demand API calls
- **Higher Throughput**: Process large documents without rate limiting concerns
- **Asynchronous Processing**: Submit jobs and retrieve results later

### On-Demand vs Batch Comparison

| Aspect | On-Demand | Batch |
|--------|-----------|-------|
| Response Time | Immediate (seconds) | Delayed (minutes to hours) |
| Cost | Standard pricing | ~50% discount |
| Rate Limits | Per-minute limits apply | No rate limiting |
| Context Continuity | Yes (via ContextManager) | No cross-chunk context |
| Best For | Interactive use, small files | Large documents, bulk processing |

## Architecture

```mermaid
flowchart TB
    subgraph Input
        A[Input File]
    end

    subgraph Preparation
        B[Parser]
        C[TextSegments]
        D[ChunkingStrategy]
        E[Chunks]
        F[BatchRequests]
    end

    subgraph Submission
        G[LLM Provider<br/>Batch API]
        H[Batch ID]
    end

    subgraph Polling["Polling Loop"]
        I{Status?}
        J[Processing...]
        K[Completed]
    end

    subgraph Results
        L[Fetch Results]
        M[Reconstruct<br/>Segments]
        N[Output File]
    end

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    I -->|processing| J
    J -->|wait| I
    I -->|completed| K
    K --> L
    L --> M
    M --> N
```

## Batch Processing Flow

```mermaid
sequenceDiagram
    participant Client as BatchTranslationEngine
    participant Parser
    participant Chunker as ChunkingStrategy
    participant Provider as LLM Provider
    participant API as Provider Batch API

    Client->>Parser: parse(input_file)
    Parser-->>Client: TextSegment[]

    Client->>Chunker: chunk_segments(segments)
    Chunker-->>Client: Chunk[]

    Note over Client: Build BatchRequest[] from chunks

    Client->>Provider: create_batch(requests)
    Provider->>API: Submit batch job
    API-->>Provider: batch_id
    Provider-->>Client: batch_id

    loop Poll for Completion
        Client->>Provider: get_batch_status(batch_id)
        Provider->>API: Check status
        API-->>Provider: BatchStatus
        Provider-->>Client: status, completed, total

        alt status == "completed"
            Note over Client: Exit loop
        else status == "processing"
            Note over Client: Sleep(poll_interval)
        end
    end

    Client->>Provider: get_batch_results(batch_id)
    Provider->>API: Fetch results
    API-->>Provider: Results[]
    Provider-->>Client: BatchResult[]

    Note over Client: Reconstruct translated segments
    Client->>Parser: write(segments, output_path)
</sequenceDiagram>
```

## Data Structures

### BatchRequest

Represents a single translation request within a batch:

```python
@dataclass
class BatchRequest:
    custom_id: str           # Unique identifier (e.g., "chunk-0")
    text: str                # Text to translate
    target_language: str     # Target language
    source_language: str     # Source language (optional)
```

### BatchStatus

Tracks the progress of a batch job:

```python
@dataclass
class BatchStatus:
    status: str      # "processing", "completed", "failed", "canceling"
    completed: int   # Number of completed requests
    total: int       # Total number of requests
```

### BatchResult

Contains the result for a single request:

```python
@dataclass
class BatchResult:
    custom_id: str              # Matches the request's custom_id
    translated_text: str | None # Translated content (if successful)
    error: str | None           # Error message (if failed)
```

## Provider-Specific Implementation

Each LLM provider implements batch processing differently:

```mermaid
flowchart LR
    subgraph Common["Common Interface"]
        A[create_batch]
        B[get_batch_status]
        C[get_batch_results]
    end

    subgraph OpenAI["OpenAI"]
        O1[JSONL File Upload]
        O2[Files API]
        O3[Batches API]
    end

    subgraph Anthropic["Anthropic"]
        AN1[Native Batch API]
        AN2[messages.batches.create]
        AN3[messages.batches.results]
    end

    subgraph Google["Google"]
        G1[Inline Requests]
        G2[batches.create]
        G3[batches.results]
    end

    A --> O1
    A --> AN1
    A --> G1
```

### OpenAI Batch API

1. **Create**: Generates JSONL file, uploads via Files API, creates batch
2. **Status**: Queries batch endpoint for completion status
3. **Results**: Downloads output file and parses JSONL results

```
Completion Window: 24 hours
File Format: JSONL with chat completion requests
```

### Anthropic Batch API

1. **Create**: Submits requests directly to `messages.batches.create()`
2. **Status**: Queries `messages.batches.retrieve()`
3. **Results**: Iterates through `messages.batches.results()`

```
Native batch support with direct API calls
```

### Google Batch API

1. **Create**: Submits inline requests via `batches.create()`
2. **Status**: Queries batch status endpoint
3. **Results**: Fetches results from completed batch

```
Uses display_name for batch identification
```

## Chunk-to-Request Mapping

```mermaid
flowchart TD
    subgraph Chunks["Chunks from ChunkingStrategy"]
        C0["Chunk 0<br/>P1, P2, P3"]
        C1["Chunk 1<br/>Code (skip), P4"]
        C2["Chunk 2<br/>P5, P6"]
    end

    subgraph Mapping["Chunk Mapping"]
        M["chunk_mapping dict<br/>custom_id → (index, chunk)"]
    end

    subgraph Requests["BatchRequests"]
        R0["custom_id: chunk-0<br/>text: P1\\n\\nP2\\n\\nP3"]
        R1["custom_id: chunk-1<br/>text: P4"]
        R2["custom_id: chunk-2<br/>text: P5\\n\\nP6"]
    end

    C0 --> R0
    C1 --> R1
    C2 --> R2

    R0 --> M
    R1 --> M
    R2 --> M
```

**Key Points:**
- Each chunk becomes one `BatchRequest`
- `custom_id` format: `chunk-{index}`
- Non-translatable segments (e.g., code blocks) are excluded from request text
- Mapping is stored for result reconstruction

## Result Reconstruction

```mermaid
flowchart TD
    subgraph Results["Batch Results"]
        BR0["chunk-0: 'T1\\n\\nT2\\n\\nT3'"]
        BR1["chunk-1: 'T4'"]
        BR2["chunk-2: 'T5\\n\\nT6'"]
    end

    subgraph Process["Reconstruction"]
        P1["Split by '\\n\\n'"]
        P2["Map to original segments"]
        P3["Preserve skip_translation segments"]
    end

    subgraph Output["Translated Segments"]
        S1[T1]
        S2[T2]
        S3[T3]
        S4[Original Code Block]
        S5[T4]
        S6[T5]
        S7[T6]
    end

    BR0 --> P1
    BR1 --> P1
    BR2 --> P1
    P1 --> P2
    P2 --> P3
    P3 --> S1
    P3 --> S2
    P3 --> S3
    P3 --> S4
    P3 --> S5
    P3 --> S6
    P3 --> S7
```

## Error Handling

The batch API handles failures gracefully:

```mermaid
flowchart TD
    A[Get BatchResult] --> B{Has translated_text?}
    B -->|Yes| C[Use translation]
    B -->|No| D{Has error?}
    D -->|Yes| E[Log warning]
    D -->|No| F[Unexpected state]
    E --> G[Keep original text]
    F --> G
    C --> H[Add to output]
    G --> H
```

**Fallback Behavior:**
- If a chunk translation fails, the original text is preserved
- Warnings are logged for failed requests
- Processing continues for remaining chunks

## Configuration

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `poll_interval` | 60 | Seconds between status checks |
| `chunk_size` | 4000 | Maximum tokens per chunk |
| `verbose` | false | Enable detailed logging |

### Usage Example

```python
from large_translate.batch_engine import BatchTranslationEngine
from large_translate.models.openai import OpenAIProvider
from large_translate.parsers.markdown import MarkdownParser

# Initialize components
provider = OpenAIProvider()
parser = MarkdownParser()
engine = BatchTranslationEngine(
    llm_provider=provider,
    parser=parser,
    chunk_size=4000,
)

# Run batch translation
stats = await engine.translate_file_batch(
    input_path=Path("large-document.md"),
    output_path=Path("translated.md"),
    target_language="Spanish",
    poll_interval=30,  # Check every 30 seconds
    verbose=True,
)

print(f"Batch ID: {stats['batch_id']}")
print(f"Chunks processed: {stats['chunks']}")
```

## Batch vs On-Demand Decision Flow

```mermaid
flowchart TD
    A[Translation Task] --> B{File Size?}
    B -->|Small < 10KB| C[On-Demand]
    B -->|Large > 10KB| D{Time Sensitive?}
    D -->|Yes| C
    D -->|No| E{Cost Sensitive?}
    E -->|Yes| F[Batch]
    E -->|No| G{Need Context<br/>Continuity?}
    G -->|Yes| C
    G -->|No| F

    C --> H[TranslationEngine<br/>translate_file]
    F --> I[BatchTranslationEngine<br/>translate_file_batch]
```

## Statistics Returned

The batch translation returns a statistics dictionary:

```python
{
    "input_chars": int,    # Total characters in input
    "output_chars": int,   # Total characters in output
    "chunks": int,         # Number of chunks processed
    "batch_id": str,       # Provider's batch job ID
}
```

## Key Differences from On-Demand

| Feature | On-Demand | Batch |
|---------|-----------|-------|
| ContextManager | Used for continuity | Not used |
| Retry Logic | Exponential backoff | Provider handles |
| Progress | Per-chunk updates | Polling-based |
| Error Recovery | Immediate retry | Logged, original preserved |
| Chunk Processing | Sequential | Parallel (provider-side) |

## Limitations

1. **No Cross-Chunk Context**: Unlike on-demand, batch processing doesn't pass context between chunks
2. **Delayed Results**: Results are not immediate; polling is required
3. **Provider Limits**: Each provider has batch size and time limits
4. **No Partial Results**: Must wait for entire batch to complete
