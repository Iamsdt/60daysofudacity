import torch.nn as nn
import torch


class Discriminator(nn.Module):
    def __init__(self, nf, input_channel=3):
        super(Discriminator, self).__init__()
        self.nf = nf
        self.input_channel = input_channel

        self.conv1 = nn.Sequential(
            # input number of image channel
            nn.Conv2d(input_channel, nf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(nf * 2, nf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.fc = nn.Linear(nf * 4 * 4 * 4, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, self.nf * 4 * 4 * 4)
        x = self.fc(x)
        return torch.sigmoid(x)


# Define Generator
class Generator(nn.Module):
    def __init__(self, z_size, nf, input_channel=3):
        super(Generator, self).__init__()
        self.z_size = z_size
        self.nf = nf
        self.fc = nn.Linear(self.z_size, self.nf * 4 * 4 * 4)

        self.dconv1 = nn.Sequential(
            nn.ConvTranspose2d(nf * 4, nf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 2),
            nn.ReLU(inplace=True)
        )

        self.dconv2 = nn.Sequential(
            nn.ConvTranspose2d(nf * 2, nf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf),
            nn.ReLU(inplace=True)
        )

        self.dconv3 = nn.Sequential(
            nn.ConvTranspose2d(nf, input_channel, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.nf*8, 4, 4)
        x = self.dconv1(x)
        x = self.dconv2(x)
        return self.dconv3(x)



