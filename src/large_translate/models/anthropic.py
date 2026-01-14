"""Anthropic Claude provider implementation."""

from anthropic import AsyncAnthropic

from .base import BaseLLMProvider, BatchRequest, BatchResult, BatchStatus
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

    # Batch inference methods

    async def create_batch(self, requests: list[BatchRequest]) -> str:
        batch_requests = []
        for req in requests:
            user_prompt = build_translation_prompt(
                text=req.text,
                target_language=req.target_language,
                source_language=req.source_language,
            )
            batch_requests.append(
                {
                    "custom_id": req.custom_id,
                    "params": {
                        "model": self.MODEL_NAME,
                        "max_tokens": self.max_output_tokens,
                        "system": TRANSLATION_SYSTEM_PROMPT,
                        "messages": [{"role": "user", "content": user_prompt}],
                    },
                }
            )

        batch = await self.client.messages.batches.create(requests=batch_requests)
        return batch.id

    async def get_batch_status(self, batch_id: str) -> BatchStatus:
        batch = await self.client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        total = (
            counts.processing
            + counts.succeeded
            + counts.errored
            + counts.canceled
            + counts.expired
        )
        return BatchStatus(
            status="completed" if batch.processing_status == "ended" else "processing",
            completed=counts.succeeded,
            total=total,
        )

    async def get_batch_results(self, batch_id: str) -> list[BatchResult]:
        results = []
        async for result in self.client.messages.batches.results(batch_id):
            translated_text = None
            error = None
            if result.result and result.result.message:
                content = result.result.message.content
                if content:
                    translated_text = content[0].text
            if hasattr(result, "error") and result.error:
                error = str(result.error)
            results.append(
                BatchResult(
                    custom_id=result.custom_id,
                    translated_text=translated_text,
                    error=error,
                )
            )
        return results
