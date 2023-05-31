import torch
import torch.nn as nn

# Let's create the block
class Block(nn.Module):
    # If we are in encoder we are in downward
    # If we are in decoder we are in upword 
    # In encoder they use leaky relu and in decoder they use relu
    def __init__(self, in_channels, out_channels, down = True, act = 'relu', use_dropout = False):
        super().__init__()
        self.conv = nn.Sequential(
            # bias = False because we are going to use batchnorm
            nn.Conv2d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1, bias = False, padding_mode='reflect')
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == 'relu' else nn.LeakyReLU(0.2)
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x

# In paper they use 0.5 dropout in first three layers in upward part of UNET

# Let's create Generator class
class Generator(nn.Module):
    def __init__(self, in_channels = 3, features = 64):
        super().__init__()
        # Initially we don't use BatchNorm
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size = 4, stride = 2, padding = 1, padding_mode = 'reflect'),
            nn.LeakyReLU(0.2),
        )
        # We down sample 6 times
        # Initially 256x256 -> 128x128
        self.down1 = Block(features, features*2, down = True, act = 'leaky', use_dropout = False) # 64x64
        self.down2 = Block(features*2, features*4, down = True, act = 'leaky', use_dropout = False) # 32x32
        self.down3 = Block(features*4, features*8, down = True, act = 'leaky', use_dropout = False) # 16x16
        self.down4 = Block(features*8, features*8, down = True, act = 'leaky', use_dropout = False) # 8x8
        self.down5 = Block(features*8, features*8, down = True, act = 'leaky', use_dropout = False) # 4x4
        self.down6 = Block(features*8, features*8, down = True, act = 'leaky', use_dropout = False) # 2x2
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*8, features*8, kernel_size = 4, stride = 2, padding = 1, padding_mode = 'reflect'), # 1x1
            nn.ReLU()
        )
        self.upword1 = Block(features*8, features*8, down = False, act = 'relu', use_dropout = True) # 2x2
        # All of them are concatenate for skip connection
        self.upword2 = Block(features*8*2, features*8, down = False, act = 'relu', use_dropout = True) # 4x4
        self.upword3 = Block(features*8*2, features*8, down = False, act = 'relu', use_dropout = True) # 8x8
        self.upword4 = Block(features*8*2, features*8, down = False, act = 'relu', use_dropout = False) # 16x16
        self.upword5 = Block(features*8*2, features*4, down = False, act = 'relu', use_dropout = False) # 32x32
        self.upword6 = Block(features*4*2, features*2, down = False, act = 'relu', use_dropout = False) # 64x64
        self.upword7 = Block(features*2*2, features, down = False, act = 'relu', use_dropout = False) # 128x128
        # In paper they use Tanh in final layer
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features*2, in_channels, kernel_size = 4, stride = 2, padding = 1),
            nn.Tanh() # 256x256
        ) 

    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottle_nake = self.bottleneck(d7)

        up1 = self.upword1(bottle_nake)
        up2 = self.upword2(torch.concat([up1, d7], dim=1))
        up3 = self.upword3(torch.concat([up2, d6], dim=1))
        up4 = self.upword4(torch.concat([up3, d5], dim=1))
        up5 = self.upword5(torch.concat([up4, d4], dim=1))
        up6 = self.upword6(torch.concat([up5, d3], dim=1))
        up7 = self.upword7(torch.concat([up6, d2], dim=1))
        final_up = self.final_up(torch.concat([up7, d1], dim=1))
        return final_up
    
# Let's test and check whether the model is working or not
def test():
    # Input image size is 256x256
    x = torch.randn((1, 3, 256, 256))
    model = Generator()
    pred = model(x)
    # We get the shape 256x256
    print(pred.shape)

if __name__ == '__main__':
    test()