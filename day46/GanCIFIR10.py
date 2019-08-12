import torch
from torch.autograd import Variable
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.nn.functional as F


## Prepare Datasets
transform = transforms.Compose([
    transforms.ToTensor()
])
train_data = datasets.CIFAR10(root='data', train=True, transform=transform, download=True)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=5)

len(train_loader)

data_iter = iter(train_loader)
images, labels = data_iter.next()

fig = plt.figure(figsize=(25,5))
for idx in range(5):
        ax = fig.add_subplot(1, 5, idx + 1, xticks=[], yticks=[])
        # denormalize first
        img = images[idx] / 2 + 0.5
        npimg = img.numpy()
        img = np.transpose(npimg, (1, 2, 0))  # transpose
        ax.imshow(img, cmap='gray')


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# Create Discriminator
class DNet(nn.Module):

    def __init__(self):
        super(DNet, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 2, bias=False),
            nn.Sigmoid()
        )
        self.model.apply(weights_init)

    def forward(self, x):
        return self.model(x)


class GNet(nn.Module):

    def __init__(self):
        super(GNet, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

        self.model.apply(weights_init)

    def forward(self, x):
        return self.model(x)


# Create models
model_ds = DNet()
model_gn = GNet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_ds.to(device)
model_gn.to(device)

batch_size = 100
img_size = 128
nz = 100 # last layer

criterion = nn.BCELoss()
input = torch.FloatTensor(batch_size, 3, img_size, img_size)
noise = torch.FloatTensor(batch_size, nz, 1, 1)
fixed_noise = torch.FloatTensor(batch_size, nz, 1, 1).normal_(0, 1)
label = torch.FloatTensor(batch_size)
real_label = 1
fake_label = 0

from torch import optim

optim_ds = optim.Adam(model_ds.parameters(), lr=0.001)
optim_gn = optim.Adam(model_gn.parameters(), lr=0.001)

# training hyperparams
num_epochs = 100
last_output = 100

# keep track of loss and generated, "fake" samples
samples = []
losses = []

print_every = 400

sample_size = 16
fixed_z = np.random.uniform(-1, 1, size=(sample_size, last_output))
fixed_z = torch.from_numpy(fixed_z).float()

# train the network
model_ds.train()
model_gn.train()

for epoch in range(num_epochs):
    for i, data in enumerate(train_loader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        optim_ds.zero_grad()
        real_cpu, _ = data
        batch_size = real_cpu.size(0)
        if torch.cuda.is_available():
            real_cpu = real_cpu.cuda()
        input.resize_as_(real_cpu).copy_(real_cpu)
        label.resize_(batch_size).fill_(real_label)
        inputv = Variable(input)
        labelv = Variable(label)

        output = model_ds(inputv)
        errD_real = criterion(output, labelv)
        errD_real.backward()
        D_x = output.data.mean()

        # train with fake
        noise.resize_(batch_size, nz, 1, 1).normal_(0, 1)
        noisev = Variable(noise)
        fake = model_gn(noisev)
        labelv = Variable(label.fill_(fake_label))
        output = model_ds(fake.detach())
        errD_fake = criterion(output, labelv)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        optim_ds.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        optim_gn.zero_grad()
        labelv = Variable(label.fill_(real_label))  # fake labels are real for generator cost
        output = model_gn(fake)
        errG = criterion(output, labelv)
        errG.backward()
        D_G_z2 = output.data.mean()
        optim_gn.step()