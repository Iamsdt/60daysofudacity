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

# visualize data
images, labels = next(iter(train_loader))

img = np.squeeze(images[0])

fig = plt.figure(figsize=(25, 10))
ax = fig.add_subplot(10)
plt.imshow(img)


def xavier_init(m):
    """ Xavier initialization """
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)


# Create Discriminator
class DNet(nn.Module):

    def __init__(self):
        super(DNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.model.apply(xavier_init)

    def forward(self, x):
        x = x.view(-1, 784)
        return self.model(x)


class GNet(nn.Module):

    def __init__(self):
        super(GNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 784),
            nn.Tanh()
        )

        self.model.apply(xavier_init)

    def forward(self, x):
        return self.model(x)


# Create models
model_ds = DNet()
model_gn = GNet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_ds.to(device)
model_gn.to(device)

criterion = nn.BCEWithLogitsLoss()

from torch import optim

optim_ds = optim.Adam(model_ds.parameters(), lr=0.001)
optim_gn = optim.Adam(model_gn.parameters(), lr=0.001)


# Calculate losses
def real_loss(D_out, smooth=False):
    batch_size = D_out.size(0)
    # label smoothing
    if smooth:
        # smooth, real labels = 0.9
        labels = torch.ones(batch_size) * 0.9
    else:
        labels = torch.ones(batch_size)  # real labels = 1

    # numerically stable loss
    criterion = nn.BCEWithLogitsLoss()
    # calculate loss
    loss = criterion(D_out.squeeze(), labels)
    return loss


def fake_loss(D_out):
    batch_size = D_out.size(0)
    labels = torch.zeros(batch_size)  # fake labels = 0
    criterion = nn.BCEWithLogitsLoss()
    # calculate loss
    loss = criterion(D_out.squeeze(), labels)
    return loss



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

    for batch_i, (real_images, _) in enumerate(train_loader):

        batch_size = real_images.size(0)
        real_images = real_images * 2 - 1

        # remove gradients
        optim_ds.zero_grad()

        # train with real image
        D_real = model_ds(real_images.to(device))
        d_real_loss = real_loss(D_real.cpu(), smooth=True)

        # generae fake image
        z = np.random.uniform(-1, 1, size=(batch_size, last_output))
        z = torch.from_numpy(z).float()
        fake_images = model_gn(z.to(device))

        D_fake = model_ds(fake_images.to(device))
        d_fake_loss = fake_loss(D_fake.cpu())

        # calculate loss
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        optim_ds.step()

        ## Generator
        optim_gn.zero_grad()

        fake_images = model_gn(z.to(device))

        D_fake = model_ds(fake_images.to(device))
        g_loss = real_loss(D_fake.cpu())  # use real loss to flip labels

        g_loss.backward()
        optim_gn.step()

        if batch_i % print_every == 0:
            # print discriminator and generator loss
            print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                epoch + 1, num_epochs, d_loss.item(), g_loss.item()))

    losses.append((d_loss.item(), g_loss.item()))

    model_gn.eval()  # eval mode for generating samples
    samples_z = model_gn(fixed_z.to(device))
    samples.append(samples_z)
    model_gn.train()  # back to train mode



# another training
# Defining the training for loop
for epoch in range(20):
    G_loss_run = 0.0
    D_loss_run = 0.0
    for i, data in enumerate(train_loader):
        X, _ = data
        batch_size = X.size(0)  # 4d

        # Definig labels for real (1s) and fake (0s) images
        one_labels = torch.ones(batch_size, 1)  # True labels
        zero_labels = torch.zeros(batch_size, 1)  # Fake Labels

        # Random normal distribution for each image
        z = torch.randn(batch_size, 100)

        # Feed forward in discriminator both
        # fake and real images
        D_real = model_ds(X)
        fakes = model_gn(z)
        D_fake = model_ds(fakes)

        # Defining the loss for Discriminator
        D_real_loss = F.binary_cross_entropy(D_real, one_labels)
        D_fake_loss = F.binary_cross_entropy(D_fake, zero_labels)
        D_loss = D_fake_loss + D_real_loss

        # backward propagation for discriminator
        optim_ds.zero_grad()
        D_loss.backward()
        optim_ds.step()

        ##############
        ### GN
        ################
        # Feed forward for generator
        z = torch.randn(batch_size, 100)
        fakes = model_gn(z)
        D_fake = model_ds(fakes)

        # loss function of generator
        G_loss = F.binary_cross_entropy(D_fake, one_labels)

        # backward propagation for generator
        optim_gn.zero_grad()
        G_loss.backward()
        optim_gn.step()

        G_loss_run += G_loss.item()
        D_loss_run += D_loss.item()

    # printing loss after each epoch
    print('Epoch:{},   G_loss:{},   D_loss:{}'.format(epoch, G_loss_run / (i + 1), D_loss_run / (i + 1)))