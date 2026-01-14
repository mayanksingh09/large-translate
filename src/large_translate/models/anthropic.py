"""Anthropic Claude provider implementation."""

from anthropic import AsyncAnthropic

from .base import BaseLLMProvider
from ..prompts import TRANSLATION_SYSTEM_PROMPT, build_translation_prompt


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude Sonnet 4.5 provider."""

    MODEL_NAME = "claude-sonnet-4-5"

    def __init__(self):
        self.client = AsyncAnthropic()

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def model_id(self) -> str:
        return self.MODEL_NAME

    @property
    def max_context_tokens(self) -> int:
        return 200000

    @property
    def max_output_tokens(self) -> int:
        return 8192

    async def translate(
        self,
        text: str,
        target_language: str,
        source_language: str | None = None,
        context: str | None = None,
    ) -> str:
        user_prompt = build_translation_prompt(
            text=text,
            target_language=target_language,
            source_language=source_language,
            context=context,
        )

        response = await self.client.messages.create(
            model=self.MODEL_NAME,
            max_tokens=self.max_output_tokens,
            system=TRANSLATION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )

        return response.content[0].text if response.content else ""

    def count_tokens(self, text: str) -> int:
        # Rough estimate: ~4 characters per token
        return len(text) // 4
