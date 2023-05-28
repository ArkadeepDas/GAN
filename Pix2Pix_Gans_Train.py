import torch
import torch.nn as nn
import torch.optim as optim
import Pix2Pix_Gans_Config
from Pix2Pix_Gans_Dataset import MapDataset
from Pix2Pix_GANs_Generator_Unet import Generator
from Pix2Pix_GANs_Discriminator import Discriminator
from torch.utils.data import DataLoader

def main():
    disc = Discriminator(in_channels = 3).to(Pix2Pix_Gans_Config.DEVICE)
    gen = Generator(in_channels = 3).to(Pix2Pix_Gans_Config.DEVICE)
    # Beta values are specify in paper
    optimizer_disc = optim.Adam(disc.parameters(), lr = Pix2Pix_Gans_Config.LEARNING_RATE, betas = (0.5, 0.999))
    optimizer_gen = optim.Adam(gen.parameters(), lr = Pix2Pix_Gans_Config.LEARNING_RATE, betas = (0.5, 0.999))
    # Here they are using standard binary cross entropy loss
    BCE = nn.BCEWithLogitsLoss()
    # They also use L1_loss or mean absolute error (MAE) loss
    # Here they don't use W-GAN-GP loss because it doesn't work well
    L1_LOSS = nn.L1Loss() # Mean absolute error (MAE) loss
    