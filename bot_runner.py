from fastapi import FastAPI, HTTPException
import asyncio
import sys
from typing import Dict
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from loguru import logger

app = FastAPI()


class BotRequest(BaseModel):
    room_url: str
    bot_token: str
    item_id: str
    guide_role: str
    tour_length: int  # tour_length in minuites


active_bots: Dict[str, asyncio.subprocess.Process] = {}


async def log_stream(stream, logger_func):
    while True:
        line = await stream.readline()
        if not line:
            break
        logger_func(line.decode().strip())


@app.post("/start_bot")
async def start_bot(request: BotRequest) -> JSONResponse:
    if request.room_url in active_bots:
        raise HTTPException(status_code=400, detail="Bot already running for this room")

    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "bot.py",
            "--room_url",
            request.room_url,
            "--bot_token",
            request.bot_token,
            "--item_id",
            request.item_id,
            "--guide_role",
            request.guide_role,
            "--tour_length",
            str(request.tour_length),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        active_bots[request.room_url] = process
        logger.info(f"Started bot for room: {request.room_url}")

        # Create tasks to log stdout and stderr
        stdout_task = asyncio.create_task(log_stream(process.stdout, logger.info))
        stderr_task = asyncio.create_task(log_stream(process.stderr, logger.error))

        # Create a task to wait for the process to complete
        wait_task = asyncio.create_task(process.wait())

        # Return immediately, allowing the bot to run in the background
        return JSONResponse({"status": "started", "room_url": request.room_url})
    except Exception as e:
        logger.error(f"Failed to start bot: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start bot: {str(e)}")


@app.post("/stop_bot/{room_url}")
async def stop_bot(room_url: str):
    if room_url not in active_bots:
        raise HTTPException(status_code=404, detail="Bot not found for this room")

    process = active_bots[room_url]
    try:
        process.terminate()
        await process.wait()
        del active_bots[room_url]
        logger.info(f"Stopped bot for room: {room_url}")
        return {"status": "stopped", "room_url": room_url}
    except Exception as e:
        logger.error(f"Failed to stop bot: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to stop bot: {str(e)}")


@app.get("/bot_status/{room_url}")
async def get_bot_status(room_url: str):
    if room_url not in active_bots:
        return {"status": "not_running", "room_url": room_url}

    process = active_bots[room_url]
    if process.returncode is None:
        return {"status": "running", "room_url": room_url}
    else:
        return {"status": "stopped", "room_url": room_url}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8010)
