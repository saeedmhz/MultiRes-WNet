""" Implementation of MultiRes-WNet """
import torch
from torch import nn


######################
# Creating the Model #
######################
class Forward_conv(nn.Module):
    """ Multi-Resolution Convolution Block ([MultiRes-UNet Paper]), w/o Depthwise Convolution
        input -> conv1 -> conv2 -> conv3
        input -> conv_res
        output: conv1 + conv2 + conv3 + conv_res
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, bias: bool):
        super(Forward_conv, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels[0], kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels[0])
        
        self.conv2 = nn.Conv2d(in_channels=out_channels[0], out_channels=out_channels[1], kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels[1])
        
        self.conv3 = nn.Conv2d(in_channels=out_channels[1], out_channels=out_channels[2], kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn3 = nn.BatchNorm2d(num_features=out_channels[2])
        
        self.conv_res = nn.Conv2d(in_channels=in_channels, out_channels=sum(out_channels), kernel_size=1, stride=1, padding=0, bias=bias)
        self.bn_res = nn.BatchNorm2d(num_features=sum(out_channels))

        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.relu(self.bn2(self.conv2(x1)))
        x3 = self.relu(self.bn3(self.conv3(x2)))

        X = torch.cat((x1, x2, x3), dim=1)
        Xres = self.relu(self.bn_res(self.conv_res(x)))

        out = Xres + X

        return out


class ForwardDepthwise(nn.Module):
    """ Multi-Resolution Convolution Block ([MultiRes-UNet Paper]), w/ Depthwise Convolutions
        input -> conv1 -> conv2 -> conv3
        input -> conv_res
        output: conv1 + conv2 + conv3 + conv_res
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super(ForwardDepthwise, self).__init__()

        self.conv1 = DepthwiseConv2d(in_channels=in_channels, out_channels=out_channels[0], kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels[0])
        
        self.conv2 = DepthwiseConv2d(in_channels=out_channels[0], out_channels=out_channels[1], kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels[1])
        
        self.conv3 = DepthwiseConv2d(in_channels=out_channels[1], out_channels=out_channels[2], kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn3 = nn.BatchNorm2d(num_features=out_channels[2])
        
        self.conv_res = nn.Conv2d(in_channels=in_channels, out_channels=sum(out_channels), kernel_size=1, stride=1, padding=0, bias=bias)
        self.bn_res = nn.BatchNorm2d(num_features=sum(out_channels))

        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.relu(self.bn2(self.conv2(x1)))
        x3 = self.relu(self.bn3(self.conv3(x2)))
        
        X = torch.cat((x1, x2, x3), dim=1)
        Xres = self.relu(self.bn_res(self.conv_res(x)))

        out = Xres + X
    
        return out


class DepthwiseConv2d(nn.Module):
    """ Replace a regular conv layer with a channel-wise conv layer followed by a regular conv with kernel_size = 1
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super(DepthwiseConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, groups=in_channels)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=bias, groups=1)

    def forward(self, x):
        x = self.conv1(x)

        out = self.conv2(x)

        return out


class Encoder(nn.Module):
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