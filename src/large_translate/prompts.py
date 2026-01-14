"""Translation prompts and templates."""

TRANSLATION_SYSTEM_PROMPT = """You are a professional translator. Your task is to translate text while:
1. Preserving the exact meaning and nuance of the original
2. Maintaining the same tone and formality level
3. Keeping any formatting markers (like markdown) intact
4. Preserving paragraph structure (keep the same number of paragraphs separated by double newlines)
5. Using natural, fluent expressions in the target language

IMPORTANT: Output ONLY the translation. Do not add any commentary, explanations, or notes."""


def build_translation_prompt(
    text: str,
    target_language: str,
    source_language: str | None = None,
    context: str | None = None,
) -> str:
    """Build the translation prompt."""
    parts = []

    if context:
        parts.append(f"[Previous context for consistency: {context}]\n")

    if source_language:
        parts.append(f"Translate the following text from {source_language} to {target_language}:\n")
    else:
        parts.append(f"Translate the following text to {target_language}:\n")

    parts.append(text)

    return "\n".join(parts)
