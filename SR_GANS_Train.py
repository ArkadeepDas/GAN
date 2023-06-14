import torch
import SR_GANS_Config
import torch.nn as nn
from torch import optim
from SR_GANS_Util import load_checkpoint, save_checkpoint, plot_examples
from SR_GANS_Loss import VGGLoss
from torch.utils.data import DataLoader
from SR_GANS_Model import Generator, Discriminator
from tqdm import tqdm
from SR_GANS_Dataset import MyImageFolder

def train(loader, gen, disc, opt_gen, opt_disc, mse, bce, vgg_loss):
    loop = tqdm(loader, leave = True)
    
    for idx, (low_res, high_res) in enumerate(loop):
        high_res = high_res.to(SR_GANS_Config.DEVICE)
        low_res = low_res.to(SR_GANS_Config.DEVICE)

        # Train Discriminator First
        fake = gen(low_res)
        disc_real = disc(high_res)
        disc_fake = disc(fake)
        # Binary Cross Entropy is used to calculate loss
        # They use label smoothing in real loss
        disc_loss_real = bce(disc_real, torch.ones_like(disc_real) - 0.1 * torch.rand_like(disc_real))
        disc_loss_fake = bce(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = disc_loss_fake + disc_loss_real

        opt_disc.zero_grad()
        loss_disc.backword()
        opt_disc.step()

        # Train Generator
        disc_fake = disc(fake)
        # Calculate MSE loss on generated and original high resolution image
        l2_loss = mse(fake, high_res)
        adversarial_loss = 1e-3 * (bce(disc_fake, torch.ones_like(disc_fake)))
        loss_for_VGG = 0.006 * VGGLoss(fake, high_res)
        # We can use any of the bellow losses
        loss_gen = adversarial_loss + loss_for_VGG
        # loss_gen = l2_loss
        
        opt_gen.zero_grad()
        loss_gen.backword()
        opt_gen.step()

        if idx % 10 == 1:
            # Passing generator and the path
            plot_examples(low_resolution_folder = 'test_image/', gen = gen)

def main():
    dataset = MyImageFolder(root_dir = 'dataset/')
    loader = DataLoader(dataset, batch_size = SR_GANS_Config.BATCH_SIZE, shuffle = True, pin_memory = True, num_workers = SR_GANS_Config.NUM_WORKERS)
    gen = Generator(in_channels = 3).to(SR_GANS_Config.DEVICE)
    disc = Discriminator(in_channels = 3).to(SR_GANS_Config.DEVICE)
    optimizer_gen = optim.Adam(gen.parameters(), lr = SR_GANS_Config.LEARNING_RATE, betas = (0.9, 0.999))
    optimizer_disc = optim.Adam(gen.parameters(), lr = SR_GANS_Config.LEARNING_RATE, betas = (0.9, 0.999))
    mse = nn.MSELoss()
    bce = nn.BCELoss()
    vgg_loss = VGGLoss()

    if SR_GANS_Config.LOAD_MODEL:
        load_checkpoint(
            checkpoint_file = SR_GANS_Config.CHECKPOINT_GEN,
            model = gen,
            optimizer = optimizer_gen,
            learning_rate = SR_GANS_Config.LEARNING_RATE
        )
        load_checkpoint(
            checkpoint_file = SR_GANS_Config.CHECKPOINT_DISC,
            model = disc,
            optimizer = optimizer_disc,
            learning_rate = SR_GANS_Config.LEARNING_RATE
        )
    
    for epoch in range(SR_GANS_Config.NUM_EPOCHS):
        train(loader, gen, disc, optimizer_gen, optimizer_disc, mse, bce, vgg_loss)

        if SR_GANS_Config.SAVE_MODEL:
            save_checkpoint(model = gen, optimizer = optimizer_gen, file_name = SR_GANS_Config.CHECKPOINT_GEN)
            save_checkpoint(model = disc, optimizer = optimizer_disc, file_name = SR_GANS_Config.CHECKPOINT_DISC)

if __name__ == '__main__':
    main()