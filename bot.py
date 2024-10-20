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
    TextFrame,
    TransportMessageFrame,
)
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.vad.silero import SileroVADAnalyzer
from pipecat.services.whisper import WhisperSTTService
from pipecat.services.cartesia import CartesiaTTSService, ExperimentalControls
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
import prompts

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

ItemID = NewType("ItemID", str)


class Guide(BaseModel):
    role: str
    voice_id: str
    voice_name: str
    model_id: str
    language: str
    experimental_controls: ExperimentalControls


guides = [
    Guide(
        role="normal",
        voice_id="f114a467-c40a-4db8-964d-aaba89cd08fa",
        voice_name="Yogaman",
        model_id="sonic-english",
        language="en",
        experimental_controls=ExperimentalControls(
            speed="normal",
            emotion=[],
        ),
    ),
    Guide(
        role="expert",  # (adults)
        voice_id="573e3144-a684-4e72-ac2b-9b2063a50b53",
        voice_name="Teacher Lady",
        model_id="sonic-english",
        language="en",
        experimental_controls=ExperimentalControls(
            speed="fast",
            emotion=[],
        ),
    ),
    Guide(
        role="basic",  # (kids)
        voice_id="79a125e8-cd45-4c13-8a67-188112f4dd22",
        voice_name="British Lady",
        model_id="sonic-english",
        language="en",
        experimental_controls=ExperimentalControls(
            speed="normal",
            emotion=[],
        ),
    ),
]


class WikiData(BaseModel):
    title: str
    contentType: str
    date: str
    extract: str  # Short extract from Wikipedia
    description: str  # Full description from Wikipedia
    photo_urls: List[str]


def find_guide_by_role(role: str, guides: list[Guide]) -> Guide | None:
    for guide in guides:
        if guide.role == role:
            return guide
    return None


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
    guide_role: str,
    tour_length: int,  # TODO: implement length
):
    guide = find_guide_by_role(guide_role, guides)

    user_id = UserID(
        "7e47340c-f3cd-5da4-8aa7-d7675cb036f5"
    )  # TODO: Replace with actual user authentication
    bot_settings = bot_container.config()

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
        voice_id=guide.voice_id,
        model_id=guide.model_id,
        language=guide.language,
        experimental_controls=guide.experimental_controls.model_dump(),
    )

    lc = LanggraphProcessor(
        graph=graph,
        config={
            "configurable": {
                "guide_role": guide_role,
                "tour_length": tour_length,
                "thread_id": f"{user_id}-{transport.participant_id}",
                "user_id": user_id,
                "item_id": item_id,
            }
        },
    )

    class BroadcastService(FrameProcessor):
        def __init__(self, transport: DailyTransport):
            super().__init__()
            self.transport = transport

        async def process_frame(self, frame: Frame, direction: FrameDirection) -> Frame:
            if isinstance(frame, TextFrame):
                message = "Broadcasting based on a specific event!"
                await self.transport.output().send_message(
                    TransportMessageFrame(message=message)
                )
            await self.push_frame(frame, direction)
            return frame

    user_response = LLMUserResponseAggregator()
    assistant_response = LLMAssistantResponseAggregator()
    logger = DetailedFrameLogger("TTS")

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_response,
            lc,
            BroadcastService(transport),
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
        messages = [{"content": "Begin your tour in it's entirety."}]
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
    parser.add_argument("--guide_role", required=True, help="Guide role")
    parser.add_argument("--tour_length", required=True, help="Tour length in minutes")
    args = parser.parse_args()

    asyncio.run(
        main(
            args.room_url,
            args.bot_token,
            ItemID(args.item_id),  # Convert to ItemID
            args.guide_role,
            args.tour_length,
        )
    )
