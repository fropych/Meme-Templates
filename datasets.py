from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
class MyDataset(Dataset):
    def __init__(self, dataframe, mode, transform, get_x, get_y=None, label_encoder = None, random_seed=42):
        super().__init__()
        self.dataframe = dataframe.copy()
        self.mode = mode
        self.get_x = get_x
        self.get_y = get_y
        self.transform = transform
        self.label_encoder = label_encoder
        
        if self.mode != "test":
            self.labels = self.dataframe[get_y].unique()
            if self.mode == 'train':
                self.label_encoder = LabelEncoder()
                self.label_encoder.fit(self.labels)
            self.dataframe['encodedTarget'] = self.label_encoder.transform(self.dataframe[get_y]).astype(np.int64)
                
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        
        if self.mode == 'test':
            return self.transform(Image.open(row[self.get_x]).convert('RGB')),
        return (
            self.transform(Image.open(row[self.get_x]).convert('RGB')),
            row['encodedTarget'],
        )