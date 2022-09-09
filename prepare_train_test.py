from PIL import Image, ImageFont, ImageDraw, UnidentifiedImageError
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from utils import *

def create_train(files):
    text = 'test text'
    font = ImageFont.truetype('impact.ttf', size=32)
    path = Path.cwd()/'train'
    path.mkdir(parents=True, exist_ok=True)
    train_data = []
    for file in tqdm(files):
        file = Path(file)
        image = Image.open(file).convert('RGB')
        image = resize(image)
        draw_text = ImageDraw.Draw(image)
        
        x_text = x_sq = y_text = y_sq = 0

        for i_text in range(16):
            draw_text.text((x_text,y_text), text, fill=(0, 0, 0), font=font)
            # for i_sq in range(4):
            #     img_copy = image.copy()
            #     draw_sq = ImageDraw.Draw(img_copy)
            #     draw_sq.rectangle([(x_sq,y_sq), (x_sq+64, y_sq+64)], fill=(0, 0, 0))
            filepath = path/f'{file.stem}_{(i_text)}.jpg'    
            image.save(filepath)
            train_data.append((
                file.stem,
                filepath
                ))
                

                # y_sq += 64    
                # x_sq += 64
            x_sq = y_sq = 0   
                
            if (i_text+1) % 2 == 0:
                y_text += 32
                x_text = 0
            else:    
                x_text += 128
    return train_data

def create_test(files):
    path = (Path.cwd()/f'test')
    path.mkdir(parents=True, exist_ok=True)
    test_data = []
    for file in tqdm(files):
        file = Path(file)
        try:
            image = Image.open(file)
        except UnidentifiedImageError as ex:
            print(f'\nEXCEPTION WITH: {file.stem}')
            continue
        image = resize(image)
        
        filepath = path/f'{file.stem}.jpg'
        image.save(filepath)
        test_data.append((
            ''.join(file.stem.split('_')[:-1]),
            filepath
            ))   
    return test_data
            
def main():
    image_df = pd.read_csv('images.csv')
    
    train_files = image_df[image_df.isTemplate].path
    train_data = create_train(train_files)
    train_df = pd.DataFrame(train_data, columns=['name', 'path'])
    train_df['isTrain'] = True
    
    test_files = image_df[~image_df.isTemplate].path
    test_data = create_test(test_files)
    test_df = pd.DataFrame(test_data, columns=['name', 'path'])
    test_df['isTrain'] = False
    
    train_test_df = pd.concat([train_df, test_df])
    train_test_df.to_csv(Path.cwd()/'train_test.csv',  encoding='utf-8', index=False)
    
if __name__ == '__main__':
    main()