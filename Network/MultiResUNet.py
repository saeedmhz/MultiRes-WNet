##################################################
## Description:
## This script contrains PyTorch Implementation 
## of MultiRes-UNet Architecture
##################################################
## Author: Saeed Mohammadzadeh
## Email: saeedmhz@bu.edu
## License: 
##################################################
import torch
from torch import nn

from .BuildingBlocks import *


class UNet(nn.Module):
    """MultiRes-UNet

    Parameters
    ----------
    in_channels : int
        Channel number of the input tensor
    c : int
        Channel number of the output of the first layer encoder
    k : int
        Size of convolutional kernels used
    
    input --> MultiRes-UNet --> output

    """
    def __init__(self, c, k, in_channels):
        super(UNet, self).__init__()

        # Adjusting padding with kernel_size to prevent shrinking the input after each convolutional layer
        p = int((k-1)/2)

        # Convolutional Block in Encoder Section at Level i: ei   --   Decoder Section: di
        self.e0 = Forward_conv(in_channels=in_channels, out_channels=[int(c/6), int(c/3), int(c/2)], kernel_size=k, stride=1, padding=p, bias=False)
        self.e1 = ForwardDepthwise(in_channels=sum([int(c/6), int(c/3), int(c/2)]), out_channels=[int(c/3), int(2*c/3), int(c)], kernel_size=k, stride=1,padding=p, bias=False)
        self.e2 = ForwardDepthwise(in_channels=sum([int(c/3), int(2*c/3), int(c)]), out_channels=[int(2*c/3), int(4*c/3), int(2*c)], kernel_size=k, stride=1, padding=p, bias=False)

        self.d1 = ForwardDepthwise(in_channels=4*c, out_channels=[int(c/3), int(2*c/3), int(c)], kernel_size=k, stride=1, padding=p, bias=False)
        self.d0 = Forward_conv(in_channels=2*c, out_channels=[int(c/6), int(c/3), int(c/2)], kernel_size=k, stride=1, padding=p, bias=False)

        # Passing output of level i from encoder to decoder through a Respath
        self.respath0 = ResPath0(in_channels=sum([int(c/6), int(c/3), int(c/2)]), out_channels=c, kernel_size=k, stride=1, padding=p, bias=False)
        self.respath1 = ResPath1(in_channels=sum([int(c/3), int(2*c/3), int(c)]), out_channels=2*c, kernel_size=k, stride=1, padding=p, bias=False)
        
        # Increaseing the resolution of image from a lower level to match the upper level
        self.upconv2 = nn.ConvTranspose2d(in_channels=sum([int(2*c/3), int(4*c/3), int(2*c)]), out_channels=2*c, kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(in_channels=sum([int(c/3), int(2*c/3), int(c)]), out_channels=c, kernel_size=2, stride=2)

        # Output
        self.out = nn.Conv2d(in_channels=sum([int(c/6), int(c/3), int(c/2)]), out_channels=1, kernel_size=3, stride=1, padding=1)
        
        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # level 0 - encoder
        x0 = self.e0(x)
        x = self.maxpool(x0)
        x0 = self.respath0(x0)

        # level 1 - encoder
        x1 = self.e1(x)
        x = self.maxpool(x1)
        x1 = self.respath1(x1)

        # level 2
        x = self.e2(x)

        # level 1 - decoder
        x = torch.cat((self.upconv2(x), x1), dim=1)
        x = self.d1(x)

        # level 0 - decoder
        x = torch.cat((self.upconv1(x), x0), dim=1)
        x = self.d0(x)

        out = self.out(x)

        return out