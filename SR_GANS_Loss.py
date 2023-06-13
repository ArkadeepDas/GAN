import torch.nn as nn
from torchvision.models import vgg19
import SR_GANS_Config

# We use pretrained weight
class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Took the features upto 36 layers
        self.vgg = vgg19(pretrained = True).features[:36].eval().to(SR_GANS_Config.DEVICE)
        self.loss = nn.MSELoss()

        # We don't want to upgrade the parameters of vgg
        for parameter in self.vgg.parameters():
            parameter.requires_grad = False
    
    def forward(self, input, target):
        # Pass input image to vgg
        # Input is Generated image
        vgg_input_features = self.vgg(input)
        vgg_target_features = self.vgg(target)
        return self.loss(vgg_input_features, vgg_target_features)