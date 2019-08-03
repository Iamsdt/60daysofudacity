import pandas as pd
from torch.utils.data import Dataset
import os
from PIL import Image


class MyDataset(Dataset):

    def __init__(self, root, csv, transform=None):
        super(MyDataset, self).__init__()
        self.root = root
        self.data = pd.read_csv(csv)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        label = item[1]
        img_idx = item[0]
        img = os.listdir(self.root)[img_idx]
        img = Image.open(self.root + "/" + img)

        if self.transform is not None:
            img = self.transform(img)

        return [img, label]




