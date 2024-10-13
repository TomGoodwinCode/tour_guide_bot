from dependency_injector import containers, providers
from config.settings import get_bot_settings
from services.redis_cache import RedisCache


class BotContainer(containers.DeclarativeContainer):
    config = providers.Singleton(get_bot_settings)

    redis_cache = providers.Singleton(
        RedisCache,
        host=config.provided.REDIS_HOST,
        port=config.provided.REDIS_PORT,
        db=config.provided.REDIS_DB,
    )


# Create factory functions for bot services
async def get_bot_service():
    return BotContainer.bot_service()


async def get_redis_cache():
    return BotContainer.redis_cache()


# Add more factory functions as needed
