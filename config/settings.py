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

    bot_api_key: Optional[str] = None
    bot_service_url: Optional[str] = None

    openai_api_key: Optional[str] = None
    elevenlabs_api_key: Optional[str] = None
    cartesia_api_key: Optional[str] = None
    daily_api_key: Optional[str] = None
    daily_api_url: Optional[str] = None
    tour_guide_room_url: Optional[str] = None
    supabase_url: Optional[str] = None
    supabase_key: Optional[str] = None
    google_places_api_key: Optional[str] = None

    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None

    langchain_api_key: Optional[str] = None
    langchain_endpoint: Optional[str] = None
    langchain_project: Optional[str] = None
    langchain_tracing_v2: Optional[str] = None

    tavily_api_key: Optional[str] = None

    fly_api_host: Optional[str] = None
    fly_api_key: Optional[str] = None
    fly_app_name: Optional[str] = None

    # Add any other bot-specific settings as needed
    PYTHONPATH: str = "/Users/tom/repositories/tour-guide/backend/shared_library"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def get_bot_settings() -> BotSettings:
    """Load and return the bot service settings."""
    load_dotenv()
    return BotSettings()
