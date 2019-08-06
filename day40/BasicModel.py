import torch.nn as nn
import torch


class BasicResNet(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)

        self.bach1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bach1(x)
        x = nn.ReLU(x)
        return x


class InceptionModel(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(InceptionModel, self).__init__()

        self.branch1x1 = BasicResNet(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = BasicResNet(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicResNet(48, 64, kernel_size=5, padding=2)

        self.branch3x3_1 = BasicResNet(in_channels, 64, kernel_size=1)
        self.branch3x3_1 = BasicResNet(64, 96, kernel_size=3, padding=1)

        self.pool = BasicResNet(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch_pool = nn.AvgPool2d(x, kernel_size=2, stride=1, padding=1)
        branch_pool = self.pool(branch_pool)

        output = [branch1x1, branch3x3, branch5x5, branch_pool]

        return torch.cat(output, dim=1)
