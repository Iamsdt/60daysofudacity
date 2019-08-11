import torch.nn as nn
import torch
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 1)

    def forward(self, x):
        out = nn.Sigmoid(self.fc1(x))
        return out


torch.manual_seed(5)
model = Net()
data = torch.load('classifier.pt')

inputs = data[:, 0:2]
labels = data[:, 2:]

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.02)

epochs = 1000
losses = []

for epoch in range(epochs):
    model.train()
    model.zero_grad()
    pred = model(inputs)
    loss = criterion(pred, labels)
    loss.backward()
    optimizer.step()
    if epoch % 200 == 0:
        print("Loss: ", loss.item())

    losses.append(loss.item())
