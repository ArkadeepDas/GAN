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

# Let's create the 'train()' function
def train(disc_Z, disc_H, gen_Z, gen_H, optimizer_gen, optimizer_disc, train_loader, L1, MSE):
    loop = tqdm(train_loader, leave = True)
    for idx, (zebra, horse) in enumerate(loop):
        zebra = zebra.to(Cycle_GANs_Config.DEVICE)
        horse = horse.to(Cycle_GANs_Config.DEVICE)

        # The main part starts here. 
        # 1) train discriminator.
        # Create fake horse from generator of horse by passing zebra image in generator 
        fake_horse = gen_H(zebra) # Output shape = 30x30
        # Now discriminator of horse take real horse and the fake horse
        disc_H_real = disc_H(horse) # Output shape = 30x30
        disc_H_fake = disc_H(fake_horse.detach()) # Output shape = 30x30
        # Real is identify as 1
        disc_H_real_loss = MSE(disc_H_real, torch.ones_like(disc_H_real))
        # Fake is identify as 0
        disc_H_fake_loss = MSE(disc_H_fake, torch.zeros_like(disc_H_fake))
        disc_H_loss = (disc_H_real_loss + disc_H_fake_loss) / 2

        # Create fake zebra from generator of zebra by passing zebra image in generator
        fake_zebra = gen_Z(horse) # Output shape = 30x30
        # Now discriminator of zebra take real zebra and the fake zebra
        disc_Z_real = disc_H(zebra) # Output shape = 30x30
        disc_Z_fake = disc_H(fake_zebra.detach()) # Output shape = 30x30
         # Real is identify as 1
        disc_Z_real_loss = MSE(disc_Z_real, torch.ones_like(disc_Z_real))
        # Fake is identify as 0
        disc_Z_fake_loss = MSE(disc_Z_fake, torch.zeros_like(disc_Z_fake))
        disc_Z_loss = (disc_Z_real_loss + disc_Z_fake_loss) / 2

        # The total loss
        D_Loss = (disc_H_loss + disc_Z_loss) / 2

        optimizer_disc.zero_grad()
        D_Loss.backword()
        optimizer_disc.step()

        # 2) train generator
        disc_H_fake = disc_H(fake_horse)
        disc_Z_fake = disc_H(fake_zebra)
        # Loss to fool discriminator
        gen_H_loss = MSE(disc_H_fake, torch.ones_like(disc_H_fake))
        gen_Z_loss = MSE(disc_Z_fake, torch.ones_like(disc_Z_fake))

        # Now we have to calculate cycle loss
        ############################################################################
        ################################ Cycle part ################################
        # We take fake horse and try to generate original zebra
        cycle_zebra = gen_Z(fake_horse)
        # We take fake zebra and try to generate original horse
        cycle_horse = gen_Z(fake_zebra)
        # Recontruction image loss calculation
        cycle_zebra_loss = L1(zebra, cycle_zebra)
        cycle_horse_loss = L1(horse, cycle_horse)
        
        # Now let's create identity loss
        identity_zebra = gen_Z(zebra)
        identity_zebra_loss = L1(zebra, identity_zebra)
        identity_horse = gen_H(horse)
        identity_horse_loss = L1(horse, identity_horse)

        # Let's putt all the loss together
        G_loss = (
            gen_H_loss + gen_Z_loss + (cycle_zebra_loss * Cycle_GANs_Config.LAMBDA_CYCLE) +
            (cycle_horse_loss * Cycle_GANs_Config.LAMBDA_CYCLE) + (identity_zebra_loss * Cycle_GANs_Config.LAMBDA_IDENTITY) +
            (identity_horse_loss * Cycle_GANs_Config.LAMBDA_IDENTITY)
        )
        # But identity loss is actually unnecessary. We can ignore
        optimizer_gen.zero_grad()
        G_loss.backword()
        optimizer_gen.step()
        print('Working')

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

    train_loader = DataLoader(train_dataset, batch_size = Cycle_GANs_Config.BATCH_SIZE, shuffle = True, num_workers = Cycle_GANs_Config.NUM_WORKERS, pin_memory = True)
    validation_loader = DataLoader(validation_dataset, batch_size = 1, shuffle = True, pin_memory = True)

    for epoch in range(Cycle_GANs_Config.NUM_EPOCH):
        train(disc_Z, disc_H, gen_Z, gen_H, optimizer_gen, optimizer_disc, train_loader, L1, MSE)

        if Cycle_GANs_Config.SAVE_MODEL:
            save_checkpoint(gen_H, optimizer_gen, file_name = Cycle_GANs_Config.CHECKPOINT_GEN_H)
            save_checkpoint(gen_Z, optimizer_gen, file_name = Cycle_GANs_Config.CHECKPOINT_GEN_Z)
            save_checkpoint(disc_H, optimizer_disc, file_name = Cycle_GANs_Config.CHECKPOINT_CRITIC_H)
            save_checkpoint(disc_Z, optimizer_disc, file_name = Cycle_GANs_Config.CHECKPOINT_CRITIC_Z)

if __name__ == '__main__':
    main()