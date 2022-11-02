from io import BytesIO
from aiogram import Bot, Dispatcher,executor, types
from aiogram.types import ContentType
from config import API_TOKEN
import requests
import json

#TODO add logging
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    await message.reply("Use /tamplate with image attachment to get the original")
    
@dp.message_handler(content_types=ContentType.PHOTO)
async def send(message: types.Message):
    #if message.caption != '/template': return
    image = BytesIO()
    await message.photo[-1].download(destination_file=image)
    image.seek(0)
    
    predict_response = requests.post('http://127.0.0.1:8000/predict/image', files={'file': image})
    predict_response = json.loads(predict_response.content.decode('utf8'))
    pred, pred_idx, probs = predict_response['pred'], predict_response['pred_idx'], predict_response['probs']

    image_response = requests.get(f'http://127.0.0.1:8000/get/image?label={pred}').content

    await message.reply_photo(BytesIO(image_response), caption=f'{pred}\nProbability: {probs[pred_idx]*100:.3f}%')
    
if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)