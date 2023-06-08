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
    def __init__(self, z_dim, in_channels, image_channels = 3):
        super().__init__()
        # First layer for the generator
        # Here we use Pixel Normalization in the begining
        # The architecture of the begining layer is PixelNorm -> ConvTr2D -> LeakyReLU -> WSConv2D -> LeakyReLU -> PixlNorm
        self.initial = nn.Sequential(
            PixNorm(),
            nn.ConvTranspose2d(in_channels = z_dim, out_channels = in_channels, kernel_size = 4, stride = 1, padding = 0), # 1x1 -> 4x4
            nn.LeakyReLU(0.2),
            WSConv2D(in_channels = in_channels, out_channels = in_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.LeakyReLU(0.2),
            PixNorm()
        )
        self.initial_RGB = WSConv2D(in_channels = in_channels, out_channels = image_channels, kernel_size = 1, stride = 1, padding = 0)
        # After each block we need a RGB
        # 1) Progress Block -> 2) RGB
        self.progress_block, self.rgb_layer = nn.ModuleList(), nn.ModuleList(self.initial_RGB)

        # Now using the factors = [1, 1, 1, 1, 1/2, 1/4, 1/8, 1/16, 1/32] we create blocks
        for i in range(len(factors)-1):
            # factors[i] -> factors[i+1]
            conv_in_channels = int(in_channels * factors[i])
            conv_out_channels = int(in_channels * factors[i+1])
            # One CNN Block -> RGB layer
            self.progress_block.append(ConvBlock(in_channels = conv_in_channels, out_channels = conv_out_channels))
            self.rgb_layer.append(WSConv2D(conv_out_channels, image_channels, kernel_size = 1, stride = 1))

    # We are going to have fade in layer which use alpha
    def fade_in(self, alpha, upscale_image, generated_image):
        return torch.tanh(alpha * generated_image + (1-alpha) * upscale_image)
    
    # We need alpha value and steps = 0 -> 4x4, steps = 1 -> 8x8, ...
    def forward(self, x, alpha, steps): 
        out = self.initial(x) # 4x4
        
        # If we only use one step then just the initial step will happen
        if steps == 0:
            self.initial_RGB(x)
        for step in range(steps):
            # Upsample -> CNN Block
            # The Upsampling is different. They upsample by using nearest neighbour. Not using ConvTranspose2d
            upscaled = F.interpolate(out, scale_factor = 2, model = 'nearest')
            out = self.progress_block[step](upscaled)
        
        # Before the last layer the RGB layer will create
        # Only before the output we use it
        # We want it at the end
        final_upscaled =self.rgb_layer[steps - 1](upscaled)
        final_out = self.rgb_layer[steps](out)
        # For incriment the structure of the image we need this
        return self.fade_in(alpha, final_upscaled, final_out)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, z_dim, in_channels, image_channels = 3):
        super().__init__()
        # The architecture is opposite of generator
        self.prog_blocks, self.rgb_layer = nn.ModuleList(), nn.ModuleList()
        self.leaky = nn.LeakyReLU(0.2)

        # In the factor list we move from backward to front 
        for i in range(len(factors)-1, 0, -1):
            conv_in_channels = int(in_channels * factors[i])
            conv_out_channels = int(in_channels * factors[i-1])
            self.prog_blocks.append(ConvBlock(in_channels = conv_in_channels, out_channels = conv_out_channels, use_pixelnorm = False))
            self.rgb_layer.append(WSConv2D(in_channels = image_channels, out_channels = conv_in_channels, kernel_size = 1, stride = 1))
        
        # Here we create end blocks
        # The initial RGB here is mirror of the generator initial RGB but it is in the last layer
        # This is 4x4 resolution 3 -> 512 mapping
        self.initial_rgb = WSConv2D(in_channels = image_channels, out_channels = in_channels, kernel_size = 1, stride = 1) 
        # This is the downsampling
        self.avg_pool = nn.AvgPool2d(kernel_size = 2, stride = 1)
        # This is the final block 4x4
        self.final_block = nn.Sequential(
            WSConv2D(in_channels = in_channels+1, out_channels = in_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.LeakyReLU(0.2),
            WSConv2D(in_channels = in_channels, out_channels = in_channels, kernel_size = 4, padding = 0, stride = 1),
            nn.LeakyReLU(0.2),
            # Final linear layer. They use neural network we use CNN same way
            WSConv2D(in_channels = in_channels, out_channels = 1, kernel_size = 1, padding = 0, stride = 1)
        )

    def fade_in(self, alpha, downscaled, out):
        # out = output from the conv layer
        # downscale = output from average pooling
        return (alpha * out + (1 - alpha) * downscaled)

    def minibatch_std(self, x):
        # It gives the std of every example of x
        batch_statistics = torch.std(x, dim = 0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3]) # N x C x W x H -> N (We get single scaler value). We repeat this in a single channel in entire image
        # This is the reason the dimention increased by 1
        return torch.cat([x, batch_statistics], dim = 1)

    def forward(self, x):
        pass