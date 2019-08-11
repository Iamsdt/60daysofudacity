import torch.nn as nn
import torch.utils.data.dataset as Dataset


class FullyConnectedModel(nn.Module):

    def __init__(self, in_size, hidden1, hidden2, out_size):
        super().__init__()
        self.fc1 = nn.Linear(in_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, out_size)
        self.dropout1 = nn.Dropout(p=0.4)
        self.dropout2 = nn.Dropout(p=0.3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = nn.ReLU(self.fc1(x))
        x = self.dropout1(x)
        x = nn.ReLU(self.fc2(x))
        x = self.dropout2(x)

        x = self.fc3(x)
        return self.softmax(x)


net = FullyConnectedModel(2048, 1536, 1024, 4)
print(net)

