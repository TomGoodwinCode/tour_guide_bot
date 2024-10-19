import argparse
import asyncio
import os
import sys
from typing import Any, Dict, List, NewType, Optional

from pipecat.transports.services.daily import DailyTransport, DailyParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.frames.frames import (
    LLMMessagesFrame,
    Frame,
    AudioRawFrame,
    EndFrame,
)
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.vad.silero import SileroVADAnalyzer
from pipecat.services.whisper import WhisperSTTService
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantResponseAggregator,
    LLMUserResponseAggregator,
    FrameDirection,
)
from pipecat.processors.frameworks.langchain import LangchainProcessor, FrameProcessor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from di.container_instance import bot_container


from loguru import logger


from graph import graph
from langgraph_processor import LanggraphProcessor
from models.schemas import UserID

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

ItemID = NewType("ItemID", str)


class WikiData(BaseModel):
    title: str
    contentType: str
    date: str
    extract: str  # Short extract from Wikipedia
    description: str  # Full description from Wikipedia
    photo_urls: List[str]


# Import other necessary Pipecat components
class DetailedFrameLogger(FrameProcessor):
    def __init__(self, name):
        self.name = name

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        logger.debug(f"{self.name}: Received frame type {type(frame).__name__}")
        if isinstance(frame, AudioRawFrame):
            logger.debug(
                f"{self.name}: Audio frame received, length: {len(frame.audio)}"
            )
        elif hasattr(frame, "text"):
            logger.debug(f"{self.name}: Frame content: {frame.text}")
        await self.push_frame(frame, direction)


async def main(
    room_url: str,
    bot_token: str,
    item_id: ItemID,
):
    user_id = UserID(
        "7e47340c-f3cd-5da4-8aa7-d7675cb036f5"
    )  # TODO: Replace with actual user authentication
    redis_cache = bot_container.redis_cache()
    bot_settings = bot_container.config()

    # Fetch POI data with WikiData
    poi_data = redis_cache.get_item_from_item_store(user_id, item_id)
    if not poi_data:
        raise ValueError(f"Item with item_id {item_id} not found")

    graph.get_state

    transport = DailyTransport(
        room_url=room_url,
        token=bot_token,
        bot_name="Tour Guide Bot 2",
        params=DailyParams(
            api_url=bot_settings.daily_api_url,
            api_key=bot_settings.daily_api_key,
            audio_out_enabled=True,
            transcription_enabled=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    stt = WhisperSTTService()
    tts = CartesiaTTSService(
        api_key=bot_settings.cartesia_api_key,
        voice_id="f114a467-c40a-4db8-964d-aaba89cd08fa",  # Yoga Man
        # "a0e99841-438c-4a64-b679-ae501e7d6091",  # Barbershop Man
    )

    lc = LanggraphProcessor(graph=graph)

    user_response = LLMUserResponseAggregator()
    assistant_response = LLMAssistantResponseAggregator()
    logger = DetailedFrameLogger("TTS")

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_response,
            lc,
            logger,
            tts,
            transport.output(),
            assistant_response,
        ]
    )

    pipeline_task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True))
    runner = PipelineRunner()

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        transport.capture_participant_transcription(participant["id"])
        lc.set_participant_id(participant["id"])
        messages = [{"content": "Please briefly introduce yourself to the user."}]
        await pipeline_task.queue_frames([LLMMessagesFrame(messages)])

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        lc.set_participant_id(None)
        await pipeline_task.queue_frames([EndFrame()])

    try:
        await runner.run(pipeline_task)
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tour Guide Bot")
    parser.add_argument("--room_url", required=True, help="Daily room URL")
    parser.add_argument("--bot_token", required=True, help="Bot token for Daily room")
    parser.add_argument("--item_id", required=True, help="Point of Interest ItemID")
    args = parser.parse_args()

    asyncio.run(
        main(
            args.room_url,
            args.bot_token,
            ItemID(args.item_id),  # Convert to ItemID
        )
    )
