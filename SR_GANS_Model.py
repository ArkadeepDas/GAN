import torch
import torch.nn as nn

# Let's create the Convolution Block
# Generator: Conv -> BatchNorm -> PReLU
# Discriminator: Conv -> BatchNorm -> LeakyReLU
# Bias should be false if we use batch normalization
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, discriminator = False, use_activation = True, use_batchNorm = True, **kwargs):
        super().__init__()
        self.use_activation = use_activation
        self.conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, **kwargs, bias = not use_activation)
        # If no batch norm then pass through what it is
        self.batchnorm = nn.BatchNorm2d(out_channels) if use_batchNorm else nn.Identity()
        self.activation = nn.LeakyReLU(0.2) if discriminator else nn.PReLU(num_parameters = out_channels)
    
    def forward(self, x):
        if self.use_activation:
            return self.activation(self.batchnorm(self.conv(x)))
        else:
            return self.batchnorm(self.conv(x))

# Upsample image by 2 (eg. 4x4 -> 8x8)
class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, scale_factor):
        super().__init__()
        # We use the out_channels this way because we are going to use pixel shuffle
        self.conv = nn.Conv2d(in_channels = in_channels, out_channels = in_channels * scale_factor ** 2, kernel_size = 3, stride = 1, padding = 1)
        self.pixsuf = nn.PixelShuffle(scale_factor) # in_channels x 4, H, W -> in_channels, Hx2, Wx2
        self.activation = nn.PReLU(num_parameters = in_channels)
    
    def forward(self, x):
        return self.activation(self.pixsuf(self.conv(x)))

# Residual Block
# Here 16 Residual blocks are used
# Residual block is only for Generator
# Here we don't change number of channels
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block1 = ConvBlock(in_channels = in_channels, out_channels = in_channels, kernel_size = 3, stride = 1, padding = 1)
        # Block 2 don't use activation function
        self.block2 = ConvBlock(in_channels = in_channels, out_channels = in_channels, kernel_size = 3, stride = 1, padding = 1, use_activation = False)
    
    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        return out + x

# Create the generator
class Generator(nn.Module):
    def __init__(self, in_channels = 3, num_channels = 64, num_blocks = 16):
        super().__init__()
        self.initial = ConvBlock(in_channels = in_channels, out_channels = num_channels, kernel_size = 9, stride = 1, padding = 4, use_batchNorm = False)
        self.residuals = nn.Sequential(*[ResidualBlock(num_channels) for _ in range(num_blocks)])
        self.convblock = ConvBlock(in_channels = num_channels, out_channels = 64, kernel_size = 3, stride = 1, padding = 1, use_activation = False)
        self.upsample = nn.Sequential(
            UpSampleBlock(in_channels = num_channels, scale_factor = 2),
            UpSampleBlock(in_channels = num_channels, scale_factor = 2)
            )
        self.final = nn.Conv2d(in_channels = num_channels, out_channels = in_channels, kernel_size = 9, stride = 1, padding = 4)

    def forward(self, x):
        initial = self.initial(x)
        x = self.residuals(initial)
        x = self.convblock(x) + initial
        x = self.upsample(x)
        x = torch.tanh(self.final(x))
        return x

# Create the discriminator
class Discriminator(nn.Module):
    def __init__(self, in_channels = 3, out_channels = [64, 64, 128, 128, 256, 256, 512, 512]):
        super().__init__()
        blocks = []
        for idx, feature in enumerate(out_channels):
            blocks.append(
                ConvBlock(in_channels = in_channels, out_channels = feature, kernel_size = 3, stride = 1 + idx % 2, padding = 1, discriminator = True, use_activation = True, use_batchNorm = False if idx == 0 else True)
            )
            in_channels = feature
        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6,6)),
            nn.Flatten(),
            nn.Linear(512 * 6 * 6, 1024),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Linear(1024, 1)
        )
    # We don't use sigmoid here because we use BCE Loss with logit = True
    def forward(self, x):
        x = self.blocks(x)
        x = self.classifier(x)
        return x
    
def test():
    low_resolution = 24 # In paper low resolution is 24x24
    x = torch.randn((1, 3, low_resolution, low_resolution))
    gen = Generator()
    out_gen = gen(x)
    disc = Discriminator()
    out_disc = disc(x)

    print(out_gen.shape)
    print(out_disc.shape)

if __name__ == '__main__':
    test()