# W-gan provides better training stability 
# Loss of W-gan means something 
# Some termination criteria is there
# It's very different from the previous GANs we seen
# In previous DCGAN the problem is called "Mode Collapse". When model collapsed only output spacific classes.

##########################################################################################
# We want the genarated images and real images distribution to be very similar in order to
# generate realistic looking images
##########################################################################################

# DCGAN Loss was similar to JS divargence which leads to unstable training
# W-gan we use Wasserstein distance
# They don't use sigmoid activation function at the end of the discriminator which does not limit output to be between 0 and 1
# In this loss discriminator try to maximize the distance as much as possible but generator wants to minimize the distance
# Loss function is bit different from previous
# Activation function used here is RMSProp

# It takes longer training time
# Here we use exact same model as DCGAN but change in output activaion of Discriminator/Critic

"""
Discriminator and Generator implementation for DCGAN
"""

import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, channels_image, features_d):
        """
        Input shape: N x channels_image x 64 x 64
        """
        super(Critic, self).__init__()
        self.disc = nn.Sequential(
            # DCGANs don't use BatchNorms in the first layer in Disciminator
            nn.Conv2d(channels_image, features_d, kernel_size=4, stride=2, padding=1), # 32 x 32
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d*2, 4, 2, 1), #16x16
            self._block(features_d*2, features_d*4, 4, 2, 1), #8x8
            self._block(features_d*4, features_d*8, 4, 2, 1), #4x4
            # At the end there is a single channel which representing fake or real image
            nn.Conv2d(features_d*8, 1, 4, 2, 0), # Output is 1x1 (strides and padding converts 4x4 to 1x1 and we want 1 output)
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
    disc = Critic(in_channels, 8)
    initialize_weights(disc)
    assert disc(x).shape == (N, 1, 1, 1)
    gen = Generator(z_dim, in_channels, 8)
    initialize_weights(gen)
    z = torch.randn((N, z_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W)
    print("!!Success!!")

test()