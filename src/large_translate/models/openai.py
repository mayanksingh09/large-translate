"""OpenAI GPT provider implementation."""

import tiktoken
from openai import AsyncOpenAI

from .base import BaseLLMProvider
from ..prompts import TRANSLATION_SYSTEM_PROMPT, build_translation_prompt


class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT-5.2 provider."""

    MODEL_NAME = "gpt-5.2"

    def __init__(self):
        self.client = AsyncOpenAI()
        # Use gpt-4o encoding as closest available
        try:
            self.encoder = tiktoken.encoding_for_model("gpt-4o")
        except KeyError:
            self.encoder = tiktoken.get_encoding("cl100k_base")

    @property
    def name(self) -> str:
        return "openai"

    @property
    def model_id(self) -> str:
        return self.MODEL_NAME

    @property
    def max_context_tokens(self) -> int:
        return 128000

    @property
    def max_output_tokens(self) -> int:
        return 16384

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

        response = await self.client.chat.completions.create(
            model=self.MODEL_NAME,
            messages=[
                {"role": "system", "content": TRANSLATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_completion_tokens=self.max_output_tokens,
            temperature=0.3,
        )

        return response.choices[0].message.content or ""

    def count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))
