import argparse
import asyncio
from typing import Any, Dict, List, NewType, Optional

from pipecat.transports.services.daily import DailyTransport, DailyParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.frames.frames import (
    LLMMessagesFrame,
    Frame,
    AudioRawFrame,
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
from supabase import create_client, Client
from postgrest.base_request_builder import APIResponse
from pipecat.processors.logger import FrameLogger

from loguru import logger

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


class SupabaseService:
    def __init__(self, supabase_client: Client):
        self.supabase_client = supabase_client

    def get_item_with_wiki_data(self, item_id: ItemID) -> Optional[Dict[str, Any]]:
        # First, try to get the supplementary data (WikiData)
        supplementary_response = (
            self.supabase_client.table("supplementary_item_data")
            .select("supplementary_info")
            .eq("item_id", item_id)
            .execute()
        )

        wiki_data = None
        if supplementary_response.data:
            wiki_data = WikiData.model_validate(
                supplementary_response.data[0]["supplementary_info"]
            )

        # Now, get the basic item data
        item_response = (
            self.supabase_client.table("api_item_data")
            .select("*")
            .eq("item_id", item_id)
            .execute()
        )

        if item_response.data:
            item_data = item_response.data[0]
            if wiki_data:
                item_data["wiki_data"] = wiki_data
            return item_data

        return None


async def main(
    room_url: str,
    bot_token: str,
    cartesia_api_key: str,
    item_id: ItemID,
    supabase_url: str,
    supabase_key: str,
):
    # Initialize Supabase client
    supabase_client: Client = create_client(supabase_url, supabase_key)
    supabase_service = SupabaseService(supabase_client)

    # Fetch POI data with WikiData
    poi_data = supabase_service.get_item_with_wiki_data(item_id)
    if not poi_data:
        raise ValueError(f"Item with item_id {item_id} not found")

    # Use poi_data in your bot logic
    system_message = f"""
    You are a tourguide bot. Do not introduce yourself to the user, immediately begin giving them fascinating details about the place you've been tasked to talk about.
    Your output will be converted to audio so don't include special characters in your answer, and pronounce abbreviations like ltd. and etc. as their full form.
    Respond to what the user said in a creative and helpful way.
    
    Here is some information about the item of interest:
    Name: {poi_data['name']},
    
    Additional information from Wikipedia:
    Title: {poi_data['wiki_data'].title if 'wiki_data' in poi_data else 'Not available'}
    Extract: {poi_data['wiki_data'].extract if 'wiki_data' in poi_data else 'Not available'}
    Description: {poi_data['wiki_data'].description if 'wiki_data' in poi_data else 'Not available'}
    
    Please be nice and helpful and tell the user succinctly all about this place, incorporating both the basic information and the Wikipedia details.
    Answer any questions they have about the place, do not repeat yourself. 
    """
    transport = DailyTransport(
        room_url=room_url,
        token=bot_token,
        bot_name="Tour Guide Bot",
        params=DailyParams(
            audio_out_enabled=True,
            transcription_enabled=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        transport.capture_participant_transcription(participant["id"])
        lc.set_participant_id(participant["id"])
        messages = [{"content": "Please briefly introduce yourself to the user."}]
        await pipeline_task.queue_frames([LLMMessagesFrame(messages)])

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        lc.set_participant_id(None)
        # Implement graceful shutdown here

    stt = WhisperSTTService()
    tts = CartesiaTTSService(
        api_key=cartesia_api_key,
        voice_id="a0e99841-438c-4a64-b679-ae501e7d6091",  # Barbershop Man
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_message,
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    chain = prompt | ChatOpenAI(model="gpt-4-1106-preview", temperature=0.7)
    history_chain = RunnableWithMessageHistory(
        chain,  # type: ignore[arg-type]
        lambda session_id: ChatMessageHistory(),
        history_messages_key="chat_history",
        input_messages_key="input",
    )
    lc = LangchainProcessor(history_chain)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. "
            "Your output will be converted to audio so don't include special characters in your answers. "
            "Respond to what the user said in a creative and helpful way.",
        },
    ]

    user_response = LLMUserResponseAggregator()
    assistant_response = LLMAssistantResponseAggregator()
    logger = DetailedFrameLogger("TTS")

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_response,
            lc,
            tts,
            transport.output(),
            assistant_response,
            # logger,
        ]
    )

    pipeline_task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True))
    runner = PipelineRunner()

    try:
        await runner.run(pipeline_task)
    except Exception as e:
        logger.exception(f"Error in main function: {str(e)}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tour Guide Bot")
    parser.add_argument("--room_url", required=True, help="Daily room URL")
    parser.add_argument("--bot_token", required=True, help="Bot token for Daily room")
    parser.add_argument("--cartesia_api_key", required=True, help="Cartesia API key")
    parser.add_argument("--item_id", required=True, help="Point of Interest ItemID")
    parser.add_argument("--supabase_url", required=True, help="Supabase URL")
    parser.add_argument("--supabase_key", required=True, help="Supabase API key")
    args = parser.parse_args()

    asyncio.run(
        main(
            args.room_url,
            args.bot_token,
            args.cartesia_api_key,
            ItemID(args.item_id),  # Convert to ItemID
            args.supabase_url,
            args.supabase_key,
        )
    )
