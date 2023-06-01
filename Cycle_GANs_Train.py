###################################################################################################
# Here we implement the main tricky part. The loss function and how the models are actually working
###################################################################################################

# We build two functions 1) train() 2) main()

import torch
import torch.nn as nn
from Cycle_GANs_Dataset import ZebraHorseDataset
import sys
from Cycle_GANs_Util import save_checkpoint, load_model
from torch.utils.data import DataLoader
import torch.optim as optim
import Cycle_GANs_Config
from tqdm import tqdm
from torchvision.utils import save_image
# Now import the models
from Cycle_GANs_Discriminator import Discriminator
from Cycle_GANs_Generator import Generator

def main():
    # Initializing discriminator model
    disc_H = Discriminator(in_channels = 3).to(Cycle_GANs_Config.DEVICE) # Classifying images of horses(Real horse or fake horse)
    disc_Z = Discriminator(in_channels = 3).to(Cycle_GANs_Config.DEVICE) # Classifying images of zebras(Real zebra or fake zebra)

    # Initializing genetor model
    gen_H = Generator(image_channels = 3).to(Cycle_GANs_Config.DEVICE) # Try to generate horse
    gen_Z = Generator(image_channels = 3).to(Cycle_GANs_Config.DEVICE) # Try to generate zebra

    # Now we initialize optimizer for both of the discriminator
    optimizer_disc = optim.Adam(
        list(disc_H.parameters()) + list(disc_Z.parameters()), lr = Cycle_GANs_Config.LEARNING_RATE, betas = (0.5, 0.999)
    )
    # Now we initialize optimizer for both of the generator
    optimizer_gen = optim.Adam(
        list(gen_H.parameters()) + list(gen_Z.parameters()), lr = Cycle_GANs_Config.LEARNING_RATE, betas = (0.5, 0.999)
    )

    # Now the loss function
    # L1 loss measure the pixel-wise absolute difference between the generator's output and the target
    L1 = nn.L1Loss()
    MSE = nn.MSELoss()

    # For testing purpose
    if Cycle_GANs_Config.LOAD_MODEL:
        # Load zebra generator
        load_model(Cycle_GANs_Config.CHECKPOINT_GEN_Z, gen_Z, optimizer_gen, Cycle_GANs_Config.LEARNING_RATE)
        # Load horse generator
        load_model(Cycle_GANs_Config.CHECKPOINT_GEN_H, gen_H, optimizer_gen, Cycle_GANs_Config.LEARNING_RATE)
        # Load zebra discriminator
        load_model(Cycle_GANs_Config.CHECKPOINT_CRITIC_Z, disc_Z, optimizer_disc, Cycle_GANs_Config.LEARNING_RATE)
        # Load horse discriminator
        load_model(Cycle_GANs_Config.CHECKPOINT_CRITIC_H, disc_H, optimizer_disc, Cycle_GANs_Config.LEARNING_RATE)
    
    train_dataset = ZebraHorseDataset(root_zebra = Cycle_GANs_Config.TRAIN_DIR + '/trainB', root_horse = Cycle_GANs_Config.TRAIN_DIR + '/trainA', transform = Cycle_GANs_Config.transforms)
    validation_dataset = ZebraHorseDataset(root_zebra = Cycle_GANs_Config.TRAIN_DIR + '/testB', root_horse = Cycle_GANs_Config.TRAIN_DIR + '/testA', transform = Cycle_GANs_Config.transforms)
    