##################################################
## Description:
## This script contrains PyTorch Implementation 
## of UNet Architecture
##################################################
## Author: Saeed Mohammadzadeh
## Email: saeedmhz@bu.edu
## License: 
##################################################
import torch
from torch import nn

class UNet(nn.Module):
    def __init__(self, c):
        super(UNet, self).__init__()
        # Convolutional Block in Encoder Section at Level i: ei   --   Decoder Section: di
        self.e01 = nn.Conv2d(in_channels=1, out_channels=c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn01 = nn.BatchNorm2d(num_features=c)
        self.e02 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn02 = nn.BatchNorm2d(num_features=c)
        
        self.e11 = nn.Conv2d(in_channels=c, out_channels=2*c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn11 = nn.BatchNorm2d(num_features=2*c)
        self.e12 = nn.Conv2d(in_channels=2*c, out_channels=2*c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn12 = nn.BatchNorm2d(num_features=2*c)
        
        self.e21 = nn.Conv2d(in_channels=2*c, out_channels=4*c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn21 = nn.BatchNorm2d(num_features=4*c)
        self.e22 = nn.Conv2d(in_channels=4*c, out_channels=4*c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn22 = nn.BatchNorm2d(num_features=4*c)
        
        self.d11 = nn.Conv2d(in_channels=4*c, out_channels=2*c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn13 = nn.BatchNorm2d(num_features=2*c)
        self.d12 = nn.Conv2d(in_channels=2*c, out_channels=2*c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn14 = nn.BatchNorm2d(num_features=2*c)
        
        self.d01 = nn.Conv2d(in_channels=2*c, out_channels=c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn03 = nn.BatchNorm2d(num_features=c)
        self.d02 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn04 = nn.BatchNorm2d(num_features=c)
        
        # Increaseing the resolution of image from a lower level to match the upper level
        self.upconv2 = nn.ConvTranspose2d(in_channels=4*c, out_channels=2*c, kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(in_channels=2*c, out_channels=c, kernel_size=2, stride=2)

        # Output
        self.out = nn.Conv2d(in_channels=c, out_channels=1, kernel_size=3, stride=1, padding=1)
        
        self.maxpool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        # encoder
        x0 = self.relu(self.bn02(self.e02(self.relu(self.bn01(self.e01(x))))))
        x = self.maxpool(x0)

        x1 = self.relu(self.bn12(self.e12(self.relu(self.bn11(self.e11(x))))))
        x = self.maxpool(x1)

        x = self.relu(self.bn22(self.e22(self.relu(self.bn21(self.e21(x))))))

        # decoder
        x = torch.cat((self.upconv2(x), x1), dim=1)
        x = self.relu(self.bn14(self.d12(self.relu(self.bn13(self.d11(x))))))

        x = torch.cat((self.upconv1(x), x0), dim=1)
        x = self.relu(self.bn04(self.d02(self.relu(self.bn03(self.d01(x))))))

        out = self.out(x)

        return out