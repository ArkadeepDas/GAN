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
            nn.Conv2d(features_d*8, 1, 4, 2, 0), # Output is 1x1
            nn.Sigmoid() # Convert the output value in range of (0<= value <= 1)
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        # DCGAN structure is CNN -> BatchNorm -> LeakyReLU
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x):
        return self.disc(x)