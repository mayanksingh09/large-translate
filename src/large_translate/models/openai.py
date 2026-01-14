"""OpenAI GPT provider implementation."""

import io
import json

import tiktoken
from openai import AsyncOpenAI

from .base import BaseLLMProvider, BatchRequest, BatchResult, BatchStatus
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

    # Batch inference methods

    async def create_batch(self, requests: list[BatchRequest]) -> str:
        # Create JSONL content
        lines = []
        for req in requests:
            user_prompt = build_translation_prompt(
                text=req.text,
                target_language=req.target_language,
                source_language=req.source_language,
            )
            lines.append(
                json.dumps(
                    {
                        "custom_id": req.custom_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": self.MODEL_NAME,
                            "messages": [
                                {"role": "system", "content": TRANSLATION_SYSTEM_PROMPT},
                                {"role": "user", "content": user_prompt},
                            ],
                            "max_completion_tokens": self.max_output_tokens,
                            "temperature": 0.3,
                        },
                    }
                )
            )

        # Upload file
        jsonl_content = "\n".join(lines)
        file = await self.client.files.create(
            file=io.BytesIO(jsonl_content.encode("utf-8")),
            purpose="batch",
        )

        # Create batch job
        batch = await self.client.batches.create(
            input_file_id=file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        return batch.id

    async def get_batch_status(self, batch_id: str) -> BatchStatus:
        batch = await self.client.batches.retrieve(batch_id)
        return BatchStatus(
            status=batch.status,
            completed=batch.request_counts.completed if batch.request_counts else 0,
            total=batch.request_counts.total if batch.request_counts else 0,
        )

    async def get_batch_results(self, batch_id: str) -> list[BatchResult]:
        batch = await self.client.batches.retrieve(batch_id)
        if not batch.output_file_id:
            return []

        content = await self.client.files.content(batch.output_file_id)
        results = []
        for line in content.text.split("\n"):
            if line.strip():
                data = json.loads(line)
                translated_text = None
                error = None
                if data.get("response") and data["response"].get("body"):
                    choices = data["response"]["body"].get("choices", [])
                    if choices:
                        translated_text = choices[0].get("message", {}).get(
                            "content", ""
                        )
                if data.get("error"):
                    error = str(data["error"])
                results.append(
                    BatchResult(
                        custom_id=data["custom_id"],
                        translated_text=translated_text,
                        error=error,
                    )
                )
        return results
