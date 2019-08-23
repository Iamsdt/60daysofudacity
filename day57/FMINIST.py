import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

## Create Dataloader

transform = transforms.Compose([
    transforms.ToTensor()
])

train_data = datasets.FashionMNIST(root='data', train=True, transform=transforms.ToTensor(),
                                   download=True)

test_data = datasets.FashionMNIST(root='data', train=False, transform=transforms.ToTensor(), download=True)


train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle=True)


# Create Model
class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=1)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=2)

        # pooling
        self.pool = nn.MaxPool2d(2, padding=1)

        # linear
        self.fc1 = nn.Linear(1280, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        input_num = x.size(0)
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))

        # flaten the tensor
        x = x.view(input_num, -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


# train
model = Model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# training
n_epochs = 15

# compare overfited
train_loss_data, test_loss_data = [], []

relu_loss, relu_accuracy = [], []

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

for epoch in range(n_epochs):
    train_loss = 0.0
    test_loss = 0.0
    accuracy = 0

    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()  # *data.size(0)

    model.eval()
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model(inputs)
        batch_loss = criterion(logps, labels)

        test_loss += batch_loss.item()

        # Calculate accuracy
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    train_loss = train_loss / len(train_loader.dataset)
    test_loss = test_loss / len(test_loader.dataset)

    train_loss_data.append(train_loss)
    test_loss_data.append(test_loss)

    relu_loss.append(test_loss)

    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch + 1,
        train_loss,
        test_loss
    ))

    print('\t\tTest Accuracy: {}'.format(accuracy))


