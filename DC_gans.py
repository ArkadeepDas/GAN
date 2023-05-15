# Let's Implement DC GANs(Deep Convolution GANs)
# In paper it's generator takes a noise of 100 dimentional vector
# Apply Transpose operation of Convolution Neural Network for Upscale
# 4x4x1024 -> 8x8x512 -> 16x16x256 -> 32x32x128 -> 64x64x3

# Diccriminator network is also the same but opposite direction
# GANs are very sensitive in parameters
# Here no polling layers are used, only convolution layers are used
# In Generator ReLU is used in all layers except the output layers, which uses Tanh.
# In Discriminator LeakyRelu activation is used in all layers

# They use mini-batch stochastic gradient descent(SGD) with a mini-batch 128.

"""
Discriminator and Generator implementation for DCGAN
"""

import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, channels_image, features_d):
        """
        Input shape: N x channels_image x 64 x 64
        """
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # DCGANs don't use BatchNorms in the first layer in Disciminator
            nn.Conv2d(channels_image, features_d, kernel_size=4, stride=2, padding=1), # 32 x 32
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d*2, 4, 2, 1), #16x16
            self._block(features_d*2, features_d*4, 4, 2, 1), #8x8
            self._block(features_d*4, features_d*8, 4, 2, 1), #4x4
            # At the end there is a single channel which representing fake or real image
            nn.Conv2d(features_d*8, 1, 4, 2, 0), # Output is 1x1 (strides and padding converts 4x4 to 1x1 and we want 1 output)
            nn.Sigmoid() # Convert the output value in range of (0<= value <= 1)
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        # DCGAN structure is CNN -> BatchNorm -> LeakyReLU
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x):
        return self.disc(x)
    

class Generator(nn.Module):
    def __init__(self, z_dim, channels_image, features_g):
        super(Generator, self).__init__()
        """
        Input: N x z_dim x 1 x 1 
        """
        self.gene = nn.Sequential(
            # DCGANs don't use BatchNorms in the last layer in Generator
            self._block(z_dim, features_g*16, 4, 1, 0), # 4x4
            self._block(features_g*16, features_g*8, 4, 2, 1), # 8x8
            self._block(features_g*8, features_g*4, 4, 2, 1), # 16x16
            self._block(features_g*4, features_g*2, 4, 2, 1), # 32x32
            nn.ConvTranspose2d(features_g*2, channels_image, kernel_size=4, stride=2, padding=1), #64x64
            nn.Tanh() #[-1 <= value <= 1]
        )
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.gene(x)

# In the paper all weights are initialize with 0 mean and 0.02 standard daviation   
def initialize_weights(model):
    """
    Initialize the weights first for all the parameters in the model
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

# Now test the code to understand whether all the shapes are properly working or not
def test():
    N, in_channels, H, W = 8, 3, 64, 64
    z_dim = 100
    x = torch.randn((N, in_channels, H, W))
    # 8 is just for testing here
    disc = Discriminator(in_channels, 8)
    initialize_weights(disc)
    assert disc(x).shape == (N, 1, 1, 1)
    gen = Generator(z_dim, in_channels, 8)
    initialize_weights(gen)
    z = torch.randn((N, z_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W)
    print("!!Success!!")

test()