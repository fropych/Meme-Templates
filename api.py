import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, Response
import pickle
from io import BytesIO
from PIL import Image
from pathlib import Path

model = pickle.load(open("models/rn18_model.pkl", "rb"))
model.eval()
images_path = Path.cwd() / "raw_images"
app = FastAPI()


@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    # extension = file.filename. in ("jpg", "jpeg", "png")
    # if not extension:
    #     return "Image must be jpg or png format!"
    file = await file.read()
    image = Image.open(BytesIO(file))
    prediction = model.predict(image)
    return prediction


@app.get("/get/image")
async def predict_api(label: str):
    return FileResponse(
        images_path / f"{label}.jpg", media_type="image/jpeg", filename=label
    )


if __name__ == "__main__":
    uvicorn.run(
        app,
    )
