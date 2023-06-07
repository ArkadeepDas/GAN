import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log2

factors = [1, 1, 1, 1, 1/2, 1/4, 1/8, 1/16, 1/32]

# This is for equalize learningrate
# WS = Weighted Scaled
# We use it at one time in forward part
# It is for equalize learning rate for convolution layes
# Here '** 0.5' is easy way to get square root of a number
class WSConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels,
                              kernel_size = kernel_size, stride = stride, padding = padding)
        self.scale = (gain / (in_channels * kernel_size ** 2)) ** 0.5
        # We don't want the bias of the conv layer to be scaled
        self.bias = self.conv.bias
        # Remove the bias or set the bias 'None'
        self.conv.bias = None
        
        # Now initializing the conv layer and the bias. In place normalization
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x):
        # Reshaping the bias term so that we can add it
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)

# Pixel Normalization
# Discriminator don't use pixel normalization
class PixNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-8
    
    def forward(self, x):
        # This is the pixel normalization equation
        return x / torch.sqrt(torch.mean(x ** 2, dim = 1, keepdim = True) + self.epsilon)

# Simple Conv block for the blocks
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_pixelnorm = True):
        super().__init__()
        # Beacuse we only use equalize convolution layers
        self.conv1 = WSConv2D(in_channels = in_channels, out_channels = out_channels)
        self.conv2 = WSConv2D(in_channels = out_channels, out_channels = out_channels)
        self.leaky = nn.LeakyReLU(0.2)
        self.pn = PixNorm()
        self.use_pixlnorm = use_pixelnorm

    def forward(self, x):
        # The main structure is Conv -> Leaky Relu -> Pixel Norm
        x = self.conv1(x)
        x = self.leaky(x)
        if self.use_pixlnorm:
            x = self.pn(x)
        x = self.conv2(x)
        x = self.leaky(x)
        if self.use_pixlnorm:
            x = self.pn(x)
        return x

# Generator
class Generator(nn.Module):
    pass

# Discriminator
class Discriminator(nn.Module):
    pass
