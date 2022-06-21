import torch.nn as nn
import torchvision.models.alexnet


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net = nn.Conv2d(3, 10, 3, 1, 0)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(9000, 10)

    def forward(self, X):
        print(X.shape)
        X = self.net(X)
        X = self.flatten(X)
        return self.linear(self.relu(X))