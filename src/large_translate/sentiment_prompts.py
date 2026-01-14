"""Sentiment analysis prompts and templates."""

SENTIMENT_SYSTEM_PROMPT = """You are a sentiment analysis expert. Your task is to classify the sentiment of text using ONLY the provided labels.

Rules:
1. You must use EXACTLY one of the provided labels for each sentence
2. Output valid JSON with the exact format specified
3. Analyze each sentence independently
4. Be objective and consistent in your classifications

Output format:
{"results": [{"sentence": "...", "label": "..."}, ...]}

IMPORTANT: Output ONLY the JSON. Do not add any commentary, explanations, or notes."""


def build_sentiment_prompt(
    sentences: list[str],
    labels: list[str],
) -> str:
    """
    Build the sentiment analysis prompt.

    IMPORTANT: The text to analyze is always at the END of the prompt.

    Args:
        sentences: List of sentences to analyze.
        labels: List of valid sentiment labels to use.

    Returns:
        The formatted prompt string.
    """
    parts = []

    # Labels instruction first
    labels_str = ", ".join(f'"{label}"' for label in labels)
    parts.append(f"Classify each sentence using ONLY these labels: [{labels_str}]")
    parts.append("")
    parts.append(
        "Return a JSON object with a 'results' array containing objects "
        "with 'sentence' and 'label' keys."
    )
    parts.append("")
    parts.append("Sentences to analyze:")

    # Text at the END (per requirements)
    for i, sentence in enumerate(sentences, 1):
        parts.append(f"{i}. {sentence}")

    return "\n".join(parts)
