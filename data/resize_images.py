import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm

SIZE = 224
IMG_RESIZED_DIR = Path("data/resized_images")

IMG_RESIZED_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("data/images.csv")
for image in tqdm(df["filename"]):
    if (IMG_RESIZED_DIR / image).exists(): continue
    
    img = Image.open(Path("data/raw_images/") / image).convert('RGB')
    img = img.resize((SIZE, SIZE))
    img.save(IMG_RESIZED_DIR / image)