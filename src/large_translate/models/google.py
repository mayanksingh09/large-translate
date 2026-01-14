"""Google Gemini provider implementation."""

from datetime import datetime

from google import genai
from google.genai import types

from .base import BaseLLMProvider, BatchRequest, BatchResult, BatchStatus, SentimentBatchRequest
from ..prompts import TRANSLATION_SYSTEM_PROMPT, build_translation_prompt
from ..sentiment_prompts import SENTIMENT_SYSTEM_PROMPT, build_sentiment_prompt


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

    # Batch inference methods

    async def create_batch(self, requests: list[BatchRequest]) -> str:
        inline_requests = []
        for req in requests:
            user_prompt = build_translation_prompt(
                text=req.text,
                target_language=req.target_language,
                source_language=req.source_language,
            )
            full_prompt = f"{TRANSLATION_SYSTEM_PROMPT}\n\n{user_prompt}"
            inline_requests.append(
                {
                    "key": req.custom_id,
                    "request": {
                        "contents": [
                            {
                                "parts": [{"text": full_prompt}],
                                "role": "user",
                            }
                        ],
                        "generation_config": {
                            "temperature": 0.3,
                            "max_output_tokens": self.max_output_tokens,
                        },
                    },
                }
            )

        batch = await self.client.aio.batches.create(
            model=self.MODEL_NAME,
            src=inline_requests,
            config={"display_name": f"translate-{datetime.now().isoformat()}"},
        )
        return batch.name

    async def get_batch_status(self, batch_id: str) -> BatchStatus:
        batch = await self.client.aio.batches.get(name=batch_id)
        # Map Google batch states to our status
        state = getattr(batch, "state", "")
        if state in ("JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"):
            status = "completed"
        else:
            status = "processing"
        return BatchStatus(
            status=status,
            completed=getattr(batch, "succeeded_count", 0),
            total=getattr(batch, "total_count", 0),
        )

    async def get_batch_results(self, batch_id: str) -> list[BatchResult]:
        batch = await self.client.aio.batches.get(name=batch_id)
        results = []
        responses = getattr(batch, "responses", []) or []
        for resp in responses:
            translated_text = None
            error = None
            if hasattr(resp, "response") and resp.response:
                translated_text = getattr(resp.response, "text", None)
            if hasattr(resp, "error") and resp.error:
                error = str(resp.error)
            results.append(
                BatchResult(
                    custom_id=resp.key,
                    translated_text=translated_text,
                    error=error,
                )
            )
        return results

    # Sentiment analysis methods

    async def analyze_sentiment(
        self,
        sentences: list[str],
        labels: list[str],
    ) -> str:
        user_prompt = build_sentiment_prompt(
            sentences=sentences,
            labels=labels,
        )

        # Google doesn't support separate system role, so concatenate
        full_prompt = f"{SENTIMENT_SYSTEM_PROMPT}\n\n{user_prompt}"

        response = await self.client.aio.models.generate_content(
            model=self.MODEL_NAME,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=self.max_output_tokens,
                temperature=0.1,  # Lower temperature for classification
                response_mime_type="application/json",
            ),
        )

        return response.text or "{}"

    async def create_sentiment_batch(
        self,
        requests: list[SentimentBatchRequest],
    ) -> str:
        inline_requests = []
        for req in requests:
            user_prompt = build_sentiment_prompt(
                sentences=req.sentences,
                labels=req.labels,
            )
            full_prompt = f"{SENTIMENT_SYSTEM_PROMPT}\n\n{user_prompt}"
            inline_requests.append(
                {
                    "key": req.custom_id,
                    "request": {
                        "contents": [
                            {
                                "parts": [{"text": full_prompt}],
                                "role": "user",
                            }
                        ],
                        "generation_config": {
                            "temperature": 0.1,
                            "max_output_tokens": self.max_output_tokens,
                            "response_mime_type": "application/json",
                        },
                    },
                }
            )

        batch = await self.client.aio.batches.create(
            model=self.MODEL_NAME,
            src=inline_requests,
            config={"display_name": f"sentiment-{datetime.now().isoformat()}"},
        )
        return batch.name
