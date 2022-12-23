import uvicorn
import asyncio
from time import sleep

from api.api import app
# from bot import bot

async def api_start():
    config = uvicorn.Config(app)
    server = uvicorn.Server(config)
    await server.serve()
  
loop = asyncio.get_event_loop()
task = loop.create_task(api_start())
loop.run_until_complete(task)
# bot.start_bot()

