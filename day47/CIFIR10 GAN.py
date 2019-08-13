import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import numpy as np
import torchvision.transforms.functional as TF

# create data sets
batch_size = 64
train_data = datasets.CIFAR10(
    root='../data', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(train_data, batch_size=64,
                          shuffle=True, num_workers=3)

# Show data

images, _ = next(iter(train_loader))

fig = plt.figure(figsize=(25, 5))
for idx in range(2):
    ax = fig.add_subplot(1, 5, idx+1, xticks=[], ytricks=[])
    img = images[idx]
    img = img / 2 + 0.5
    img = img.numpy()
    img = np.transpose(img, (1, 2, 0))
    ax.imshow(img, cmap='grey')


# Define classes
def init_weight(m):
    class_name = m.__class__.__name__

    if class_name.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)

    if class_name.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomHorizontalFlip(),
    transforms.Grayscale(3),
    transforms.RandomRotation(20),
    # transforms.ColorJitter(),
    transforms.ToTensor(),
    transforms.Lambda(lambda img:TF.rotate(img,20))
    # transforms.Normalize(mean, std)
])