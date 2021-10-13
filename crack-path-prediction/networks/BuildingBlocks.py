##################################################
## Description:
## This script contrains building blocks of the 
## MultiRes-WNet
##################################################
## Author: Saeed Mohammadzadeh
## Email: saeedmhz@bu.edu
## License: 
##################################################
import torch
from torch import nn


################################################################################
# MultiRes-WNet Building Blocks
################################################################################  
class Forward_conv(nn.Module):
    """Multi-Resolution Convolution Block ([MultiRes-UNet Paper]), w/o Depthwise Convolution

    Parameters
    ----------
    in_channels : int
        Channel number of the input tensor
    out_channels : int
        Channel number of the output tensor
    kernel_size, stride, padding, bias: int, int, int, bool

    input --> conv1 -> conv2 -> conv3 |
                                      |--> output (conv1 + conv2 + conv3 + conv_res)
    input --> conv_res                |

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
    """Multi-Resolution Convolution Block ([MultiRes-UNet Paper]), w/ Depthwise Convolution

    Parameters
    ----------
    in_channels : int
        Channel number of the input tensor
    out_channels : int
        Channel number of the output tensor
    kernel_size, stride, padding, bias: int, int, int, bool

    input --> conv1 -> conv2 -> conv3 |
                                      |--> output (conv1 + conv2 + conv3 + conv_res)
    input --> conv_res                |

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
    """Replace regular convolutions with depthwise convolutions

    Parameters
    ----------
    in_channels : int
        Channel number of the input tensor
    out_channels : int
        Channel number of the output tensor
    kernel_size, stride, padding, bias: int, int, int, bool

    input --> conv_channelwise -> conv_depthwise --> output

    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super(DepthwiseConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, groups=in_channels)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=bias, groups=1)

    def forward(self, x):
        x = self.conv1(x)

        out = self.conv2(x)

        return out


class ResPath0(nn.Module):
    """Residual path: sends encoder output to the decoder input through a series of convolutions

    Parameters
    ----------
    in_channels : int
        Channel number of the input tensor
    out_channels : int
        Channel number of the output tensor
    kernel_size, stride, padding, bias: int, int, int, bool

    input --------> conv1 -------------> conv2 -------------> conv3 -------------> conv4 --------> output
             |                |   |                |   |                |   |                |
             ---> conv1_res ---   ---> conv2_res ---   ---> conv3_res ---   ---> conv4_res ---

    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super(ResPath0, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.conv1_res = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bn1_res = nn.BatchNorm2d(num_features=out_channels)
        
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.conv2_res = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bn2_res = nn.BatchNorm2d(num_features=out_channels)

        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn3 = nn.BatchNorm2d(num_features=out_channels)
        self.conv3_res = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bn3_res = nn.BatchNorm2d(num_features=out_channels)

        self.conv4 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn4 = nn.BatchNorm2d(num_features=out_channels)
        self.conv4_res = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bn4_res = nn.BatchNorm2d(num_features=out_channels)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x))) + self.relu(self.bn1_res(self.conv1_res(x)))
        x = self.relu(self.bn2(self.conv2(x))) + self.relu(self.bn2_res(self.conv2_res(x)))
        x = self.relu(self.bn3(self.conv3(x))) + self.relu(self.bn3_res(self.conv3_res(x)))
        
        out = self.relu(self.bn4(self.conv4(x))) + self.relu(self.bn4_res(self.conv4_res(x)))

        return out


class ResPath1(nn.Module):
    """Residual path: sends encoder output to the decoder input through a series of convolutions

    Parameters
    ----------
    in_channels : int
        Channel number of the input tensor
    out_channels : int
        Channel number of the output tensor
    kernel_size, stride, padding, bias: int, int, int, bool

    input --------> conv1 -------------> conv2 -------------> conv3 --------> output
             |                |   |                |   |                |
             ---> conv1_res ---   ---> conv2_res ---   ---> conv3_res ---

    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super(ResPath1, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.conv1_res = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bn1_res = nn.BatchNorm2d(num_features=out_channels)
        
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.conv2_res = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bn2_res = nn.BatchNorm2d(num_features=out_channels)

        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn3 = nn.BatchNorm2d(num_features=out_channels)
        self.conv3_res = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bn3_res = nn.BatchNorm2d(num_features=out_channels)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x))) + self.relu(self.bn1_res(self.conv1_res(x)))
        x = self.relu(self.bn2(self.conv2(x))) + self.relu(self.bn2_res(self.conv2_res(x)))
        
        out = self.relu(self.bn3(self.conv3(x))) + self.relu(self.bn3_res(self.conv3_res(x)))

        return out