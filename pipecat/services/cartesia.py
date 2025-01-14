#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import json
import uuid
import base64
import asyncio
import time

from typing import AsyncGenerator

from pipecat.processors.frame_processor import FrameDirection
from pipecat.frames.frames import (
    CancelFrame,
    ErrorFrame,
    Frame,
    AudioRawFrame,
    StartInterruptionFrame,
    StartFrame,
    EndFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TextFrame,
    LLMFullResponseEndFrame,
)
from pipecat.services.ai_services import TTSService
from pydantic import BaseModel, Field, field_validator
from typing import Union, Literal, List
from pydantic.types import confloat

from loguru import logger

# See .env.example for Cartesia configuration needed
try:
    import websockets
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Cartesia, you need to `pip install pipecat-ai[cartesia]`. Also, set `CARTESIA_API_KEY` environment variable."
    )
    raise Exception(f"Missing module: {e}")


# ----------------------------- Tom's Code ----------------------------- #
class EmotionTag(BaseModel):
    name: Literal["anger", "positivity", "surprise", "sadness", "curiosity"]
    level: Literal["lowest", "low", "high", "highest"] = None

    def __str__(self):
        if self.level:
            return f"{self.name}:{self.level}"
        return self.name


class ExperimentalControls(BaseModel):
    speed: Union[
        Literal["slowest", "slow", "normal", "fast", "fastest"],
        float,
    ] = Field(default="normal", description="Speech rate control")

    emotion: List[EmotionTag] = Field(
        default_factory=list, description="Emotion controls"
    )

    @field_validator("emotion", mode="before")
    @classmethod
    def parse_emotion_tag(cls, v):
        if isinstance(v, list):
            return [
                (
                    EmotionTag(
                        name=parts[0], level=parts[1] if len(parts) > 1 else None
                    )
                    if isinstance(item, str)
                    else item
                )
                for item in v
                for parts in [item.split(":")]
            ]
        return v

    def model_dump(self, *args, **kwargs):
        d = super().model_dump(*args, **kwargs)
        d["emotion"] = [str(e) for e in d["emotion"]]
        return d

    @field_validator("speed")
    @classmethod
    def validate_speed(cls, v):
        if isinstance(v, float) and not (-1.0 <= v <= 1.0):
            raise ValueError("Float speed must be between -1.0 and 1.0")
        return v


# ----------------------------- Tom's Code ----------------------------- #


class CartesiaTTSService(TTSService):

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        cartesia_version: str = "2024-06-10",
        url: str = "wss://api.cartesia.ai/tts/websocket",
        model_id: str = "sonic-english",
        encoding: str = "pcm_s16le",
        sample_rate: int = 16000,
        language: str = "en",
        experimental_controls: ExperimentalControls = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Aggregating sentences still gives cleaner-sounding results and fewer
        # artifacts than streaming one word at a time. On average, waiting for
        # a full sentence should only "cost" us 15ms or so with GPT-4o or a Llama 3
        # model, and it's worth it for the better audio quality.
        self._aggregate_sentences = True

        # we don't want to automatically push LLM response text frames, because the
        # context aggregators will add them to the LLM context even if we're
        # interrupted. cartesia gives us word-by-word timestamps. we can use those
        # to generate text frames ourselves aligned with the playout timing of the audio!
        self._push_text_frames = False

        self._api_key = api_key
        self._cartesia_version = cartesia_version
        self._url = url
        self._voice_id = voice_id
        self._model_id = model_id
        self._output_format = {
            "container": "raw",
            "encoding": encoding,
            "sample_rate": sample_rate,
        }
        self._language = language

        self._websocket = None
        self._context_id = None
        self._context_id_start_timestamp = None
        self._timestamped_words_buffer = []
        self._receive_task = None
        self._context_appending_task = None
        self._experimental_controls = experimental_controls

    def can_generate_metrics(self) -> bool:
        return True

    async def set_voice(self, voice: str):
        logger.debug(f"Switching TTS voice to: [{voice}]")
        self._voice_id = voice

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    async def _connect(self):
        try:
            self._websocket = await websockets.connect(
                f"{self._url}?api_key={self._api_key}&cartesia_version={self._cartesia_version}"
            )
            self._receive_task = self.get_event_loop().create_task(
                self._receive_task_handler()
            )
            self._context_appending_task = self.get_event_loop().create_task(
                self._context_appending_task_handler()
            )
        except Exception as e:
            logger.exception(f"{self} initialization error: {e}")
            self._websocket = None

    async def _disconnect(self):
        try:
            await self.stop_all_metrics()

            if self._context_appending_task:
                self._context_appending_task.cancel()
                await self._context_appending_task
                self._context_appending_task = None
            if self._receive_task:
                self._receive_task.cancel()
                await self._receive_task
                self._receive_task = None
            if self._websocket:
                await self._websocket.close()
                self._websocket = None

            self._context_id = None
            self._context_id_start_timestamp = None
            self._timestamped_words_buffer = []
        except Exception as e:
            logger.exception(f"{self} error closing websocket: {e}")

    async def _handle_interruption(
        self, frame: StartInterruptionFrame, direction: FrameDirection
    ):
        await super()._handle_interruption(frame, direction)
        self._context_id = None
        self._context_id_start_timestamp = None
        self._timestamped_words_buffer = []
        await self.stop_all_metrics()
        await self.push_frame(LLMFullResponseEndFrame())

    async def _receive_task_handler(self):
        try:
            async for message in self._websocket:
                msg = json.loads(message)
                if not msg or msg["context_id"] != self._context_id:
                    continue
                if msg["type"] == "done":
                    await self.stop_ttfb_metrics()
                    await self.push_frame(TTSStoppedFrame())
                    # Unset _context_id but not the _context_id_start_timestamp
                    # because we are likely still playing out audio and need the
                    # timestamp to set send context frames.
                    self._context_id = None
                    self._timestamped_words_buffer.append(
                        ("LLMFullResponseEndFrame", 0)
                    )
                elif msg["type"] == "timestamps":
                    # logger.debug(f"TIMESTAMPS: {msg}")
                    self._timestamped_words_buffer.extend(
                        list(
                            zip(
                                msg["word_timestamps"]["words"],
                                msg["word_timestamps"]["end"],
                            )
                        )
                    )
                elif msg["type"] == "chunk":
                    await self.stop_ttfb_metrics()
                    if not self._context_id_start_timestamp:
                        self._context_id_start_timestamp = time.time()
                    frame = AudioRawFrame(
                        audio=base64.b64decode(msg["data"]),
                        sample_rate=self._output_format["sample_rate"],
                        num_channels=1,
                    )
                    await self.push_frame(frame)
                elif msg["type"] == "error":
                    logger.error(f"{self} error: {msg}")
                    await self.push_frame(TTSStoppedFrame())
                    await self.stop_all_metrics()
                    await self.push_error(ErrorFrame(f'{self} error: {msg["error"]}'))
                else:
                    logger.error(f"Cartesia error, unknown message type: {msg}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.exception(f"{self} exception: {e}")

    async def _context_appending_task_handler(self):
        try:
            while True:
                await asyncio.sleep(0.1)
                if not self._context_id_start_timestamp:
                    continue
                elapsed_seconds = time.time() - self._context_id_start_timestamp
                # Pop all words from self._timestamped_words_buffer that are
                # older than the elapsed time and print a message about them to
                # the console.
                while (
                    self._timestamped_words_buffer
                    and self._timestamped_words_buffer[0][1] <= elapsed_seconds
                ):
                    word, timestamp = self._timestamped_words_buffer.pop(0)
                    if word == "LLMFullResponseEndFrame" and timestamp == 0:
                        await self.push_frame(LLMFullResponseEndFrame())
                        continue
                    await self.push_frame(TextFrame(word))
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.exception(f"{self} exception: {e}")

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"Generating TTS: [{text}]")

        try:
            if not self._websocket:
                await self._connect()

            if not self._context_id:
                await self.push_frame(TTSStartedFrame())
                await self.start_ttfb_metrics()
                self._context_id = str(uuid.uuid4())

            msg = {
                "transcript": text + " ",
                "continue": True,
                "context_id": self._context_id,
                "model_id": self._model_id,
                "voice": {
                    "mode": "id",
                    "id": self._voice_id,
                    "__experimental_controls": (
                        self._experimental_controls
                        if self._experimental_controls
                        else {}
                    ),
                },
                "output_format": self._output_format,
                "language": self._language,
                "add_timestamps": True,
            }
            try:
                await self._websocket.send(json.dumps(msg))
                await self.start_tts_usage_metrics(text)
            except Exception as e:
                logger.error(f"{self} error sending message: {e}")
                await self.push_frame(TTSStoppedFrame())
                await self._disconnect()
                await self._connect()
                return
            yield None
        except Exception as e:
            logger.exception(f"{self} exception: {e}")
