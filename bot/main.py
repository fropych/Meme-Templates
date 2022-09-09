from io import BytesIO
from pathlib import Path
from aiogram import Bot, Dispatcher,executor, types
from aiogram.types import InputFile, ContentType
from fastai.vision.learner import load_learner
from fastai.vision.core import PILImage
from PIL import Image
from utils.resize import resize
from config import API_TOKEN

#TODO add logging
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

images_path = Path.cwd()/'raw_images'
path_to_download = Path.cwd()/'downloaded'
model_path = Path.cwd()/'models/model.pkl'
model = load_learner(model_path)

@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    await message.reply("Use /tamplate with image attachment to get the original")
    
@dp.message_handler(content_types=ContentType.PHOTO)
async def send(message: types.Message):
    print('aaaaaaaaaaaa')
    if message.caption != '/template': return
    image = BytesIO()
    await message.photo[-1].download(destination_file=image)
    
    pilimage = Image.open(image)
    pilimage = resize(pilimage)
    image = PILImage.create(pilimage.to_bytes_format())
    
    pred, pred_idx, probs = model.predict(image)

    template_img = InputFile(images_path/f'{pred}.jpg')
    await message.reply_photo(template_img, caption=f'{pred}\nProbability: {probs[pred_idx]*100:.3f}%')
    
if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)