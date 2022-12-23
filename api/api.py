import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import pickle
from io import BytesIO
from PIL import Image
from pathlib import Path
from model.vision_learner import VisionModel

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'VisionModel':
            return VisionModel
        return super().find_class(module, name)

images_path = Path.cwd() / "raw_images"
model_path = "model/models/rn18_model.pkl"

byte_model = open(model_path, "rb")
model = CustomUnpickler(byte_model).load()
model.eval()
app = FastAPI()
byte_model.close()

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
    uvicorn.run(app,)
