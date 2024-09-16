from fastapi import FastAPI, HTTPException
import asyncio
import sys
from typing import Dict
from pydantic import BaseModel
from loguru import logger

app = FastAPI()


class BotRequest(BaseModel):
    room_url: str
    bot_token: str
    item_id: str
    cartesia_api_key: str
    supabase_url: str
    supabase_key: str


active_bots: Dict[str, asyncio.subprocess.Process] = {}


@app.post("/start_bot")
async def start_bot(request: BotRequest):
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
            "--cartesia_api_key",
            request.cartesia_api_key,
            "--supabase_url",
            request.supabase_url,
            "--supabase_key",
            request.supabase_key,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        active_bots[request.room_url] = process
        logger.info(f"Started bot for room: {request.room_url}")
        return {"status": "started", "room_url": request.room_url}
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
