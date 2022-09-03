##################################################
## Description:
## This script contrains PyTorch Implementation 
## of the Autoencoder architecture used in
## the paper
##################################################
## Author: Saeed Mohammadzadeh
## Email: saeedmhz@bu.edu
## License: 
##################################################
import torch
from torch import nn

from BuildingBlocks import Forward_conv, ForwardDepthwise


class Encoder(nn.Module):
    """The encoder section of our Autoencoder

    Parameters
    ----------
    in_channels : int
        Channel number of the input tensor
    c : int
        Channel number of the output of the first layer encoder
    k : int
        Size of convolutional kernels used
    
    input --> encoder --> encoded_info

    """
    def __init__(self, c, k, in_channels):
        super(Encoder, self).__init__()

        # padding
        p = int((k-1)/2)

        # Encoder Section: ei
        self.e0 = Forward_conv(in_channels=in_channels, out_channels=[int(c/6), int(c/3), int(c/2)], kernel_size=k, stride=1, padding=p, bias=False)
        
        self.e1 = ForwardDepthwise(in_channels=sum([int(c/6), int(c/3), int(c/2)]), out_channels=[int(c/3), int(2*c/3), int(c)], kernel_size=k, stride=1,padding=p, bias=False)
        
        self.e2 = ForwardDepthwise(in_channels=sum([int(c/3), int(2*c/3), int(c)]), out_channels=[int(2*c/3), int(4*c/3), int(2*c)], kernel_size=k, stride=1, padding=p, bias=False)

        self.out_img = nn.Conv2d(in_channels=sum([int(2*c/3), int(4*c/3), int(2*c)]), out_channels=1, kernel_size=3, stride=1, padding=1)
        
        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x0 = self.e0(x)
        x = self.maxpool(x0)

        x1 = self.e1(x)
        x = self.maxpool(x1)

        x = self.e2(x)

        out = self.out_img(x)

        return out


class Decoder(nn.Module):
    """The decoder section of our Autoencoder

    Parameters
    ----------
    in_channels : int
        Channel number of the input tensor
    c : int
        Channel number of the output of the first layer encoder
    k : int
        Size of convolutional kernels used
    
    encoded_info --> decoder --> output

    """
    def __init__(self, c, k, in_channels):
        super(Decoder, self).__init__()

        # padding
        p = int((k-1)/2)

        # Decoder Section: di
        self.d2 = Forward_conv(in_channels=in_channels, out_channels=[int(2*c/3), int(4*c/3), int(2*c)], kernel_size=k, stride=1, padding=p, bias=False)

        self.d1 = ForwardDepthwise(in_channels=sum([int(2*c/3), int(4*c/3), int(2*c)]), out_channels=[int(c/3), int(2*c/3), int(c)], kernel_size=k, stride=1, padding=p, bias=False)
        
        self.d0 = Forward_conv(in_channels=sum([int(c/3), int(2*c/3), int(c)]), out_channels=[int(c/6), int(c/3), int(c/2)], kernel_size=k, stride=1, padding=p, bias=False)

        self.upconv2 = nn.ConvTranspose2d(in_channels=sum([int(2*c/3), int(4*c/3), int(2*c)]), out_channels=sum([int(2*c/3), int(4*c/3), int(2*c)]), kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(in_channels=sum([int(c/3), int(2*c/3), int(c)]), out_channels=sum([int(c/3), int(2*c/3), int(c)]), kernel_size=2, stride=2)

        self.out_img = nn.Conv2d(in_channels=sum([int(c/6), int(c/3), int(c/2)]), out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x2 = self.d2(x)
        x = self.upconv2(x2)

        x1 = self.d1(x)
        x = self.upconv1(x1)

        x = self.d0(x)

        out = self.out_img(x)

        return out