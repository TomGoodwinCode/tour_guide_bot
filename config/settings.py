from pydantic_settings import BaseSettings
from typing import Optional
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()


class BotSettings(BaseSettings):
    """
    Bot-specific settings loaded from environment variables or .env file.

    This class defines all the configuration settings for the bot service,
    including API keys and other bot-related configurations.
    """

    # Add bot-specific settings here
    bot_api_key: Optional[str] = None
    bot_service_url: Optional[str] = None

    # You can also include relevant settings from the main application
    openai_api_key: Optional[str] = None
    elevenlabs_api_key: Optional[str] = None

    # Add any other bot-specific settings as needed

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def get_bot_settings() -> BotSettings:
    """Load and return the bot service settings."""
    load_dotenv()
    return BotSettings()
