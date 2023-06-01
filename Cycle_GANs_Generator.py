# In Generator they use 9 residual blocks
# In paper they use InstanceNorm2D instate of BatchNorm.

# Here we use ModuleList. Which holds a list of 'nn.Module' objects. Whether 'nn.Sequential' holds and ordered sequence of nn.Module

import torch
import torch.nn as nn

# Let's create the CNN block for Generator
# This is for begining and end block
class ConvBlock(nn.Module):
    # kwargs is basically kernel, padding, stride etc.
    def __init__(self, in_channel, out_channel, down = True, use_act = True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, padding_mode = 'reflect', **kwargs)
            if down
            else nn.ConvTranspose2d(in_channel, out_channel, **kwargs),
            nn.InstanceNorm2d(out_channel),
            nn.ReLU(inplace = True) if use_act else nn.Identity() # Just pass through. Do nothing
        )

    def forward(self, x):
        return self.conv(x)

# Let's create Residual Block for skip connection
# This is the middle portion
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            # Default stride = 1. So the output shape is same as input shape
            ConvBlock(channels, channels, kernel_size = 3, padding = 1, stride = 1),
            ConvBlock(channels, channels, use_act = False, kernel_size = 3, padding = 1, stride = 1),
        )
    
    def forward(self, x):
        return x + self.block(x)

# Let's create the Generator Model
# Initially it's Conv2D -> ReLU
class Generator(nn.Module):
    def __init__(self, image_channels, out_channels = 64, num_residual = 9):
        super().__init__()
        # This is the initial block
        self.initial = nn.Sequential(
            nn.Conv2d(image_channels, out_channels, kernel_size = 7, stride = 1, padding = 3, padding_mode = 'reflect'),
            nn.ReLU(inplace = True)
        )

        # This is the down block
        self.down_block = nn.ModuleList(
            [
                ConvBlock(out_channels, out_channels*2, down = True, kernel_size = 3, stride = 2, padding = 1),
                ConvBlock(out_channels*2, out_channels*4, down = True, kernel_size = 3, stride = 2, padding = 1)
            ]
        )

        # Next is Residual Blocks
        # We have 9 Residual Blocks
        self.residual_block = nn.Sequential(
            *[ResidualBlock(out_channels*4) for _ in range(num_residual)]
        )

        # Next we use Up sampling
        self.Up_Block = nn.ModuleList(
            [
                ConvBlock(out_channels*4, out_channels*2, down = False, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
                ConvBlock(out_channels*2, out_channels, down = False, kernel_size = 3, stride = 2, padding = 1, output_padding = 1)
            ]
        )

        # Last layer from high dimention to RGB 
        self.Last = nn.Conv2d(out_channels, image_channels, kernel_size = 7, stride = 1, padding = 3, padding_mode = 'reflect')

    def forward(self, x):
        # First Layer
        x = self.initial(x)
        
        # Then Down Blocks
        for layer in self.down_block:
            x = layer(x)
        
        # Now Residual Blocks
        x = self.residual_block(x)

        # Now Upsampling Blocks
        for layer in self.Up_Block:
            x = layer(x)
        
        # We use tanh at the end after the last layer
        return torch.tanh(self.Last(x))

# Let's test the model
def test():
    # Input image size is 256x256
    x = torch.randn((1, 3, 256, 256))
    model = Generator(image_channels = 3, out_channels = 9)
    pred = model(x)
    # We get the shape 256x256
    print(pred.shape)

if __name__ == '__main__':
    test()