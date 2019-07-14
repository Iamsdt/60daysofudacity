from torch.utils.data import Dataset
import pandas as pd
import os
from skimage import io
import numpy as np


class MyDataset(Dataset):

    def __init__(self, path, cvsfile, transform=None):
        super(MyDataset, self).__init__()
        self.data = pd.read_csv(cvsfile)
        self.root = path
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        path = os.path.join(self.root, img_name)
        img = io.imread(path)

        labels = self.data.iloc[idx, 1:]
        labels = np.array([labels])
        labels = labels.astype('float').reshape(-1, 2)

        if self.transform:
            img = self.transform(img)

        return img, labels
