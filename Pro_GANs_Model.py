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
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, gain = 2):
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
        self.progress_block, self.rgb_layer = nn.ModuleList(), nn.ModuleList([self.initial_RGB])

        # Now using the factors = [1, 1, 1, 1, 1/2, 1/4, 1/8, 1/16, 1/32] we create blocks
        for i in range(len(factors)-1):
            # factors[i] -> factors[i+1]
            conv_in_channels = int(in_channels * factors[i])
            conv_out_channels = int(in_channels * factors[i+1])
            # One CNN Block -> RGB layer
            self.progress_block.append(ConvBlock(in_channels = conv_in_channels, out_channels = conv_out_channels))
            self.rgb_layer.append(WSConv2D(conv_out_channels, image_channels, kernel_size = 1, stride = 1, padding = 0))

    # We are going to have fade in layer which use alpha
    def fade_in(self, alpha, upscale_image, generated_image):
        return torch.tanh(alpha * generated_image + (1-alpha) * upscale_image)
    
    # We need alpha value and steps = 0 -> 4x4, steps = 1 -> 8x8, ...
    def forward(self, x, alpha, steps): 
        out = self.initial(x) # 4x4
        # If we only use one step then just the initial step will happen
        if steps == 0:
            return self.initial_RGB(out)
        for step in range(steps):
            # Upsample -> CNN Block
            # The Upsampling is different. They upsample by using nearest neighbour. Not using ConvTranspose2d
            upscaled = F.interpolate(out, scale_factor = 2, mode = 'nearest')
            out = self.progress_block[step](upscaled)
        # Before the last layer the RGB layer will create
        # Only before the output we use it
        # We want it at the end
        final_upscaled = self.rgb_layer[steps - 1](upscaled)
        final_out = self.rgb_layer[steps](out)
        # For incriment the structure of the image we need this
        return self.fade_in(alpha, final_upscaled, final_out)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, in_channels, image_channels = 3):
        super().__init__()
        # The architecture is opposite of generator
        self.prog_blocks, self.rgb_layer = nn.ModuleList(), nn.ModuleList()
        self.leaky = nn.LeakyReLU(0.2)

        # In the factor list we move from backward to front
        # RGB layer will work for input size 1024 x 1024 -> 512 x 512 -> 256 x256 ......
        for i in range(len(factors)-1, 0, -1):
            conv_in_channels = int(in_channels * factors[i])
            conv_out_channels = int(in_channels * factors[i-1])
            self.prog_blocks.append(ConvBlock(in_channels = conv_in_channels, out_channels = conv_out_channels, use_pixelnorm = False))
            self.rgb_layer.append(WSConv2D(in_channels = image_channels, out_channels = conv_in_channels, kernel_size = 1, stride = 1, padding = 0))
        
        # Here we create end blocks
        # The initial RGB here is mirror of the generator initial RGB but it is in the last layer
        # This is 4x4 resolution 3 -> 512 mapping
        self.initial_rgb = WSConv2D(in_channels = image_channels, out_channels = in_channels, kernel_size = 1, stride = 1, padding = 0) 
        # Add in the last
        self.rgb_layer.append(self.initial_rgb)
        # This is the downsampling
        self.avg_pool = nn.AvgPool2d(kernel_size = 2, stride = 1)
        # This is the final block 4x4
        self.final_block = nn.Sequential(
            WSConv2D(in_channels = in_channels+1, out_channels = in_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.LeakyReLU(0.2),
            WSConv2D(in_channels = in_channels, out_channels = in_channels, kernel_size = 4, stride = 1, padding = 0),
            nn.LeakyReLU(0.2),
            # Final linear layer. They use neural network we use CNN same way
            WSConv2D(in_channels = in_channels, out_channels = 1, kernel_size = 1, stride = 1, padding = 0)
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

    def forward(self, x, alpha, steps):
        # This is opposite of generator
        # The last is for 4x4
        current_step = len(self.prog_blocks) - steps

        out = self.leaky(self.rgb_layer[current_step](x))
        if steps == 0:
            out = self.minibatch_std(out)
            out = self.final_block(out).view(out.shape[0], -1) # 4x4 - > 1x1
            return out
        
        # Here we use Average Pooling
        # 1) RGB -> 2) Prog Block -> 3) Avg Pooling -> 4) Fade In
        downscaled = self.leaky(self.rgb_layer[current_step + 1](self.avg_pool(x)))
        out = self.avg_pool(self.prog_blocks[current_step](out))
        out = self.fade_in(alpha, downscaled, out)

        # +1 is used because we already done with the current step
        for step in range(current_step + 1, len(self.prog_blocks)):
            out = self.prog_blocks[step](out)
            out = self.avg_pool(out)
        
        out = self.minibatch_std(out)
        out = self.final_block(out).view(out.shape[0], -1) # 4x4 - > 1x1
        return out
    
# Let's do some test cases
if __name__ == '__main__':
    Z_DIM = 100
    IN_CHANNELS = 256
    gen = Generator(z_dim = Z_DIM, in_channels = IN_CHANNELS, image_channels = 3)
    disc = Discriminator(in_channels = IN_CHANNELS, image_channels = 3)

    for image_size in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        num_steps = int(log2(image_size / 4))
        print(num_steps)
        x = torch.randn((1, Z_DIM, 1, 1))
        img = gen(x, 0.5, num_steps)
        assert img.shape == (1, 3, image_size, image_size)
        out = disc(img, 0.5, num_steps)
        print(out.shape)
        # assert out.shape == (1, 1) 
        print(f'Success! At image size: {image_size}')

        if image_size == 64:
            break