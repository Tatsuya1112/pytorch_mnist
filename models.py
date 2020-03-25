import torch.nn as nn

class TwoLayerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 50)
        self.fc2 = nn.Linear(50, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SimpleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 30, kernel_size=5)
        self.fc1 = nn.Linear(12*12*30, 100)
        self.fc2 = nn.Linear(100, 10)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)

        x = x.view(-1, 12*12*30)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x