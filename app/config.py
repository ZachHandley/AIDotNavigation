"""
Configuration for the application.
"""

import os
from functools import lru_cache
from typing import Optional, Any
from pydantic import Field
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
from loguru import logger
import openai
import nest_asyncio

nest_asyncio.apply()

# Load .env file if it exists
load_dotenv()

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Project settings
    PROJECT_NAME: str = Field(default="AIDotNavigation", description="Project name")

    # OpenAI settings
    OPENAI_API_KEY: Optional[str] = Field(default=None, description="OpenAI API key")
    OPENAI_API_BASE: Optional[str] = Field(
        default=None, description="Custom OpenAI API base URL"
    )
    AI_MODEL: str = Field(default="gpt-4o-mini", description="Default OpenAI model")
    TOO_LARGE_THRESHOLD: int = Field(default=80000, description="Threshold for too large responses")

    # Model config
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=True, extra="ignore"
    )

    def __init__(self, **data: Any):
        """Initialize settings with defaults and calculated values."""
        super().__init__(**data)

    def model_dump(self, *args, **kwargs) -> dict[str, Any]:
        """Get settings as a dictionary with sensitive values redacted."""
        settings_dict = super().model_dump(*args, **kwargs)

        # Redact sensitive information for logging/display
        if settings_dict.get("OPENAI_API_KEY"):
            settings_dict["OPENAI_API_KEY"] = "***REDACTED***"

        return settings_dict

    def configure_openai(self) -> None:
        """Configure OpenAI with the current settings."""
        try:
            # Configure OpenAI API key if available
            if self.OPENAI_API_KEY:
                os.environ["OPENAI_API_KEY"] = self.OPENAI_API_KEY
                logger.info("Set OpenAI API key")
            else:
                raise ValueError("OPENAI_API_KEY is not set")

            # Configure OpenAI base URL if provided
            if self.OPENAI_API_BASE:
                os.environ["OPENAI_API_BASE"] = self.OPENAI_API_BASE
                logger.info(f"Set OpenAI API base to: {self.OPENAI_API_BASE}")

            openai.api_key = self.OPENAI_API_KEY
            if self.OPENAI_API_BASE:
                openai.api_base = self.OPENAI_API_BASE
            logger.success("OpenAI configuration complete")

        except Exception as e:
            logger.error(f"Error configuring OpenAI: {e}")


@lru_cache()
def get_settings() -> Settings:
    """Return settings instance with caching for performance."""
    try:
        settings = Settings()
        logger.info(f"Loaded settings for {settings.PROJECT_NAME}")
        return settings
    except Exception as e:
        logger.error(f"Error loading settings: {e}")
        raise


# Create global instance of settings
settings = get_settings()

# Initialize OpenAI configuration
try:
    settings.configure_openai()
except Exception as e:
    logger.error(f"Error configuring OpenAI: {e}")
