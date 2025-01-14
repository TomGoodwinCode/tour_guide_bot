from dataclasses import dataclass
from typing import Any, Optional, Union

from langchain.schema import Document
from langchain_core.messages import HumanMessage

from loguru import logger
from pipecat.frames.frames import (
    DataFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
    # LLMResponseEndFrame,
    # LLMResponseStartFrame,
    TextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from langgraph.graph.state import CompiledStateGraph

try:
    from langchain_core.messages import AIMessageChunk
except ModuleNotFoundError as e:
    logger.exception(
        "In order to use Langgraph, you need to `pip install pipecat-ai[langchain]`. "
    )
    raise Exception(f"Missing module: {e}")


@dataclass
class ToolResultMessage(DataFrame):
    result: Any
    type: str = "tool_result"

    def __str__(self):
        return f"{self.name}(result: {self.result})"


class LanggraphProcessor(FrameProcessor):
    def __init__(self, graph: CompiledStateGraph, config: Optional[dict] = None):
        super().__init__()
        self._graph: CompiledStateGraph = graph
        self._config: dict = config or {}

    def set_participant_id(self, participant_id: str):
        self._participant_id = participant_id

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, LLMMessagesFrame):
            # Messages are accumulated by the `LLMUserResponseAggregator` in a list of messages.
            # The last one by the human is the one we want to send to the LLM.
            logger.debug(f"Got transcription frame {frame}")
            text: str = frame.messages[-1]["content"]

            await self._ainvoke(text.strip())
        else:
            await self.push_frame(frame, direction)

    @staticmethod
    def __get_token_value(text: Union[str, AIMessageChunk]) -> str:
        match text:
            case str():
                return text
            case AIMessageChunk():
                return str(text.content)
            case _:
                return ""

    async def _ainvoke(self, text: str):
        logger.debug(f"Invoking agent with {text}")
        await self.push_frame(LLMFullResponseStartFrame())
        try:
            async for event in self._graph.astream_events(
                {"messages": [HumanMessage(content=text)]},
                config=self._config,
                version="v2",
            ):
                match event["event"]:
                    case "on_chat_model_stream":
                        logger.info(f"Received event: {event}")
                        # await self.push_frame(LLMResponseStartFrame())
                        await self.push_frame(
                            TextFrame(self.__get_token_value(event["data"]["chunk"]))
                        )
                        # await self.push_frame(LLMResponseEndFrame()) !I think these has been deprecated and is no longer needed
                    case "summary_event":
                        logger.info(f"Received summary event: {event}")
                        print(f"Received event: {event['data']['chunk']}")
                    case "on_tool_start":
                        logger.debug(
                            f"Starting tool: {event['name']} with inputs: {event['data'].get('input')}"
                        )
                    case "on_tool_end":
                        # TODO Implement non-retriever tools that return strings
                        pass
                    case "on_retriever_end":
                        docs: list[Document] = event["data"]["output"]["documents"]
                        logger.debug(f"Sending {len(docs)} docs")
                        for doc in docs:
                            await self.push_frame(
                                ToolResultMessage(doc.dict(exclude_none=True))
                            )
                    case _:
                        pass
        except GeneratorExit:
            logger.exception(f"{self} generator was closed prematurely")
        except Exception as e:
            logger.exception(f"{self} an unknown error occurred: {e}")

        await self.push_frame(LLMFullResponseEndFrame())
