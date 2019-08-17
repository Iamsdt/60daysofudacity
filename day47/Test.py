import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable


# Prepare Data loader
x = np.array([1., 2., 3., 4., 5.])
y = np.array([3., 5., 7., 9., 11.])


# Create model

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.fc = nn.Linear(1, 1)


    def forward(self, x):
        x = self.fc(x)
        return x
    


model = Net()

# create tensor
x = torch.from_numpy(x)
y = torch.from_numpy(y)

x = Variable(x)
y = Variable(y)

output = model(x)
print(output)

