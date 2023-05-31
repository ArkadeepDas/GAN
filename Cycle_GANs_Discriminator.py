# For discriminator in Cycle gan we use 4 convolution layers with stride 2
# Here we use the patch gan concept as pix2pix gan used
# Each of these values in patch gan corresponds to seeing a patch in the original image
# In paper they use InstanceNorm2D instate of BatchNorm. In first step they don't use any Normalization
# For discriminator they use Leaky Relu
# At the end we use sigmoid activation function for the value in between 0 and 1

import torch
import torch.nn as nn

# Let's tru to create a block
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 4, stride = stride, padding = 1, bias = True, padding_mode = 'reflect'),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x):
        return self.conv(x)
    
# Let's create the Discriminator
class Discriminator(nn.Module):
    # We use conv block for all those features
    def __init__(self, in_channels = 3, out_channels = [64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, out_channels[0], kernel_size = 4, stride = 2, padding = 1, padding_mode = 'reflect'),
            nn.LeakyReLU(0.2)
        )

        layers = []
        in_channels = out_channels[0]
        for feature in out_channels[1:]:
            # In last layer we use stride = 1
            layers.append(CNNBlock(in_channels, feature, stride = 1 if feature == out_channels[-1] else 2))
            in_channels = feature
        # This is the last layer
        layers.append(nn.Conv2d(in_channels, 1, kernel_size = 4, stride = 1, padding = 1, padding_mode = 'reflect'))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial(x)
        x = self.model(x)
        return torch.sigmoid(x)
    
# Let's test the discriminator model
def test():
    # Input image size is 256x256
    x = torch.randn((1, 3, 256, 256))
    model = Discriminator()
    pred = model(x)
    # We get the shape 30x30
    print(pred.shape)

if __name__ == '__main__':
    test()