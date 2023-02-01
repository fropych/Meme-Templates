import asyncio
import json
from io import BytesIO

import requests
from aiogram import Bot, Dispatcher, executor, types
from aiogram.types import ContentType
from config import API_URL, BOT_TOKEN

# TODO add logging

loop = asyncio.get_event_loop()
bot = Bot(token=BOT_TOKEN, loop=loop)
dp = Dispatcher(bot)

@dp.message_handler(commands=["start"])
async def send_welcome(message: types.Message):
    await message.reply("Use /tamplate with image attachment to get the original")


@dp.message_handler(content_types=ContentType.PHOTO)
async def send(message: types.Message):
    # if message.caption != '/template': return
    image = BytesIO()
    await message.photo[-1].download(destination_file=image)
    image.seek(0)

    predict_response = requests.post(f"{API_URL}/predict/image", files={"file": image})
    predict_response = json.loads(predict_response.content.decode("utf8"))
    pred, pred_idx, probs = (
        predict_response["pred"],
        predict_response["pred_idx"],
        predict_response["probs"],
    )

    image_response = requests.get(f"{API_URL}/get/image?label={pred}").content

    await message.reply_photo(
        BytesIO(image_response),
        caption=f"{pred}\nProbability: {probs[pred_idx]*100:.3f}%",
    )


def start_bot():
    executor.start_polling(dp, loop=loop, skip_updates=True)

if __name__=='__main__':
    start_bot()