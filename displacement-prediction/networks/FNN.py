##################################################
## Description:
## This script contrains PyTorch Implementation 
## of a simple Feedforward Neural Network
##################################################
## Author: Saeed Mohammadzadeh
## Email: saeedmhz@bu.edu
## License: 
##################################################
import torch
from torch import nn

class FNN(nn.Module):
    def __init__(self, c):
        super(FNN, self).__init__()
        # Convolutional Block in Encoder Section at Level i: ei   --   Decoder Section: di
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=c)

        self.conv2 = nn.Conv2d(in_channels=c, out_channels=2*c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=2*c)

        self.conv3 = nn.Conv2d(in_channels=2*c, out_channels=4*c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=4*c)

        self.fc4 = nn.Linear(in_features=3*3*4*c, out_features=2*28*28, bias=False)
        self.bn4 = nn.BatchNorm1d(num_features=2*28*28)

        self.fc5 = nn.Linear(in_features=2*28*28, out_features=28*28, bias=False)
        self.bn5 = nn.BatchNorm1d(num_features=28*28)

        self.out = nn.Linear(in_features=28*28, out_features=28*28, bias=True)

        self.maxpool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        # encoder
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.maxpool(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.flatten(x)

        x = self.relu(self.bn4(self.fc4(x)))
        x = self.relu(self.bn5(self.fc5(x)))

        out = self.out(x)

        return out
