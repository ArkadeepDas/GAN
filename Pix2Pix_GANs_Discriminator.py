import torch 
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 2):
        super().__init__()
        # Now we are trying to create a block
        # The Block structure is: 1)Conv -> 2)BatchNorm -> 3)LeakyReLU
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, padding=1, bias=False, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x):
        return self.conv(x)

# Now let's create discriminator class
# Here we pass both X and Y image and concatenate along these channels
# That's why we use in_channels = in_channels*2 in initial block
class Discriminator(nn.Module):
    # In channels are used the value 3 as a default because it use normal RGB image
    def __init__(self, in_channels=3, features = [64, 128, 256, 512]):
        super().__init__()
        # In initial block there is no BatchNorm
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels * 2, features[0], 4, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU()
        )

        # Now create some layers
        layers = []
        # First initial layer use 64 as a out_channels
        in_channels = features[0]
        # So we skip the first value because it was used in initial block
        for feature in features[1:]:
            layers.append(
                # Because they use stride=2 all of them but except the last layer 
                CNNBlock(in_channels, feature, stride=1 if feature == features[-1] else 2)
            )
            in_channels = feature
        layers.append(
            nn.Conv2d(in_channels, 1, kernel_size = 4, stride = 1, padding = 1, padding_mode = 'reflect')
        )
        self.model = nn.Sequential(*layers)
    
    def forward(self, x, y):
        x = torch.concat([x, y], dim=1)
        x = self.initial(x)
        x = self.model(x)
        return x

# Let's test and check whether the model is working or not
def test():
    # Input image size is 256x256
    x = torch.randn((1, 3, 256, 256))
    # Output image size is 256x256
    y = torch.randn((1, 3, 256, 256))
    model = Discriminator()
    pred = model(x, y)
    # We get the shape 30x30
    print(pred.shape)

if __name__ == '__main__':
    test()