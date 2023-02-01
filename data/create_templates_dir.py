import pandas as pd
from pathlib import Path
import shutil
import json

Path("data/image_templates").mkdir(parents=True, exist_ok=True)
df = pd.read_csv("data/images.csv")
df = df[df["isTemplate"]].reset_index(drop=True)
for filename in df["filename"]:
    shutil.copyfile(
        Path("data/raw_images/") / filename, Path("data/image_templates/") / filename
    )

image_templates = dict()
for i in range(len(df)):
    image_templates[i] = df.loc[i, "name"]
    
json.dump(image_templates, open("data/image_templates.json", "w"))
    