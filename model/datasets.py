from torch.utils.data import Dataset
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, dataframe, mode, transform, get_x, get_y=None, random_seed=42):
        super().__init__()
        self.dataframe = dataframe.copy()
        self.mode = mode
        self.get_x = get_x
        self.get_y = get_y
        self.transform = transform
       
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        
        if self.mode == 'test':
            return self.transform(Image.open(row[self.get_x]).convert('RGB')),
        return (
            self.transform(Image.open(row[self.get_x]).convert('RGB')),
            row['label'],
        )