"""
Enterprise Application Settings Engine
Validated against Pydantic v2 BaseSettings specifications.
Loads environment variables from `.env` with secure fallback defaults.
"""

from typing import Literal, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """
    Central type-safe application configuration register.
    Validates environment keys, ports, secrets, and provider endpoints at runtime.
    """
    # Environment & Host
    APP_ENV: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Active application runtime tier."
    )
    APP_SECRET_KEY: str = Field(
        default="enterprise-multiagent-dev-secret-key-32-chars-long!",
        description="Master encryption and JWT signing secret key."
    )
    API_PORT: int = Field(default=8000, description="FastAPI server port.")
    UI_PORT: int = Field(default=8501, description="Streamlit dashboard port.")

    # High-Tier LLM Providers (Groq & Gemini)
    GROQ_API_KEY: Optional[str] = Field(default=None, description="Groq Cloud API key.")
    GROQ_MODEL: str = Field(
        default="llama-3.3-70b-versatile",
        description="Groq high-tier reasoning model identifier."
    )
    GEMINI_API_KEY: Optional[str] = Field(default=None, description="Google Gemini API key.")
    GEMINI_MODEL: str = Field(
        default="gemini-2.0-flash",
        description="Google Gemini model identifier."
    )

    # Local Ollama Worker Configuration
    OLLAMA_BASE_URL: str = Field(
        default="http://localhost:11434",
        description="Local Ollama daemon REST endpoint."
    )
    OLLAMA_WORKER_MODEL: str = Field(
        default="qwen2.5:7b",
        description="Target lightweight worker model running inside Ollama."
    )

    # Payment Gateway & Webhook Cryptography
    RAZORPAY_KEY_ID: str = Field(default="rzp_test_mock_key_id", description="Razorpay key ID.")
    RAZORPAY_KEY_SECRET: str = Field(default="mock_key_secret", description="Razorpay key secret.")
    PAYMENT_WEBHOOK_SECRET: str = Field(
        default="sandbox_webhook_hmac_secret_key_12345",
        description="HMAC-SHA256 signature verification secret for incoming payment webhooks."
    )

    # Persistence / Database
    DATABASE_URL: Optional[str] = Field(
        default=None,
        description="PostgreSQL connection URI. If None in development, SqliteSaver is used."
    )
    SQLITE_DB_PATH: str = Field(
        default="dev_checkpoints.db",
        description="Local SQLite checkpoint file path for development."
    )

    # Cache Layer (Dual-Mode: In-Memory / Redis)
    REDIS_URL: Optional[str] = Field(
        default=None,
        description="Redis connection URI. If None, in-memory LRU cache with TTL is used."
    )
    CACHE_TTL_SECONDS: int = Field(
        default=1800,
        description="Cache expiration TTL in seconds (Default: 30 minutes)."
    )

    # Observability (LangSmith)
    LANGCHAIN_TRACING_V2: bool = Field(default=False, description="Enable LangSmith tracing.")
    LANGCHAIN_ENDPOINT: str = Field(
        default="https://api.smith.langchain.com",
        description="LangSmith API endpoint."
    )
    LANGCHAIN_API_KEY: Optional[str] = Field(default=None, description="LangSmith API key.")
    LANGCHAIN_PROJECT: str = Field(
        default="enterprise-multiagent-nutrition-commerce",
        description="LangSmith project name."
    )

    # Pydantic v2 Settings configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


# Global singleton instance for settings
settings = AppSettings()
