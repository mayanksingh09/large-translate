"""Configuration and settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Keys (loaded from environment)
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    google_api_key: str | None = None

    # Default settings
    default_chunk_size: int = 4000  # tokens
    default_context_length: int = 200  # characters


def get_settings() -> Settings:
    """Get application settings."""
    return Settings()
