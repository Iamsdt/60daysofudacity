import torch.nn as nn
import torch


class MyEnsemble(nn.Module):

    def __init__(self, modelA, modelB, modelC, input):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC

        self.fc1 = nn.Linear(input, 16)

    def forward(self, *input):
        out1 = self.modelA(input)
        out2 = self.modelB(input)
        out3 = self.modelC(input)

        out = out1 + out2 + out3

        x = self.fc1(out)
        return torch.softmax(x, dim=1)
