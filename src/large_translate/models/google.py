"""Google Gemini provider implementation."""

from google import genai
from google.genai import types

from .base import BaseLLMProvider
from ..prompts import TRANSLATION_SYSTEM_PROMPT, build_translation_prompt


class GoogleProvider(BaseLLMProvider):
    """Google Gemini 3 Flash provider."""

    MODEL_NAME = "gemini-3-flash-preview"

    def __init__(self):
        self.client = genai.Client()

    @property
    def name(self) -> str:
        return "google"

    @property
    def model_id(self) -> str:
        return self.MODEL_NAME

    @property
    def max_context_tokens(self) -> int:
        return 1000000

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

        full_prompt = f"{TRANSLATION_SYSTEM_PROMPT}\n\n{user_prompt}"

        response = await self.client.aio.models.generate_content(
            model=self.MODEL_NAME,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=self.max_output_tokens,
                temperature=0.3,
            ),
        )

        return response.text or ""

    def count_tokens(self, text: str) -> int:
        # Rough estimate: ~4 characters per token
        return len(text) // 4
