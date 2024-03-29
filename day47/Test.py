from torch.utils.data.dataset import Dataset
import os
from PIL import Image


class MyDatasets(Dataset):

    def __init__(self, root, transform):
        super(MyDatasets, self).__init__()

        self.root = root
        self.length = os.listdir(root)

    
    def __len__(self):
        return self.length

    
    def __getitem__(self, idx):

        name = os.listdir(self.root)[idx]

        path = self.root+"/"+name

        img = Image.open(path).convert("RGBA")
        
