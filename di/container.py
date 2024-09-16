from dependency_injector import containers, providers
from config.settings import get_bot_settings


class BotContainer(containers.DeclarativeContainer):
    config = providers.Singleton(get_bot_settings)


# Create factory functions for bot services
async def get_bot_service():
    return BotContainer.bot_service()


# Add more factory functions as needed
