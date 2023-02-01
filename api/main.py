import pickle
from io import BytesIO
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
import json
import torch
from torchvision import transforms

IMG_SIZE = 224
label_converter = json.load(open('data/image_templates.json', "r"))

pred_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

images_path = Path("data/image_templates")
model_path = "models/vit_model.pkl"

model = pickle.load(open(model_path, "rb"))
model.eval()
app = FastAPI()


@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    # extension = file.filename. in ("jpg", "jpeg", "png")
    # if not extension:
    #     return "Image must be jpg or png format!"
    file = await file.read()
    image = Image.open(BytesIO(file)).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = pred_transforms(image)
    with torch.no_grad():
        prediction = model(image.unsqueeze(0))
        prediction = torch.softmax(prediction, dim=-1).numpy()[0]
    prediction = {
        'pred': label_converter[str(prediction.argmax(-1))],
        'pred_idx': prediction.argmax(-1).tolist(),
        'probs': prediction .tolist(),
    }
    return prediction


@app.get("/get/image")
async def get_api(label: str):
    return FileResponse(
        images_path / f"{label}.jpg", media_type="image/jpeg", filename=label
    )


if __name__ == "__main__":
    uvicorn.run(app,)
