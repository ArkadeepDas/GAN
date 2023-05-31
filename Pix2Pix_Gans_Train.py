import torch
import torch.nn as nn
import torch.optim as optim
import Pix2Pix_Gans_Config
from Pix2Pix_Gans_Dataset import MapDataset
from Pix2Pix_GANs_Generator_Unet import Generator
from Pix2Pix_GANs_Discriminator import Discriminator
from torch.utils.data import DataLoader
from Pix2Pix_GANs_Util import load_model, save_checkpoint, save_some_examples
from tqdm import tqdm

# Let's create train functuion
def train(disc, gen, optimizer_disc, optimizer_gen, BCE, L1, g_scaler, d_scaler, loader):
    loop = tqdm(loader, leave = True)
    for idx, (x, y) in enumerate(loop):
        x, y = x.to(Pix2Pix_Gans_Config.DEVICE), y.to(Pix2Pix_Gans_Config.DEVICE)

        # Train Discriminator
        with torch.cuda.amp.autocast_mode():
            # Create fake image from generator 256x256
            y_fake = gen(x)
            # Now discriminator discriminate between real and noise
            d_real = disc(x, y) # 30x30
            # Now same thing will happen wityh fake image produce by generator
            # use 'detach()' to break the computational graph when we do optimizer.step
            d_fake = disc(x, y_fake.detach()) # 30x30

            # Now the loss calculation with real output from discriminator
            D_real_loss = BCE(d_real, torch.ones_like(d_real))
            # Same way we calculate the fake loss
            D_fake_loss = BCE(d_fake, torch.zeros_like(d_fake))
            D_Loss = (D_real_loss + D_fake_loss) / 2
        
        # Now calculate the gradient
        disc.zero_grad()
        d_scaler.scale(D_Loss).backword()
        d_scaler.step(optimizer_disc)
        d_scaler.update()

        # Train Generator
        with torch.cuda.amp.autocast_mode():
            d_fake = disc(x, y_fake)
            # Try to fool the discriminator
            G_fake_loss = BCE(d_fake, torch.ones_like(d_fake))
            # In paper they also compute the L1 loss/Mean absolute error
            # So that the output of generator is much more real
            L1_LOSS = L1(y_fake, y) * Pix2Pix_Gans_Config.L1_LAMBDA
            G_Loss = (G_fake_loss + L1_LOSS)
            # Loss is same as previous. Just for image re-construction they use extra L1_loss 

        # Now calculate the gradient
        gen.zero_grad()
        g_scaler.scale(G_Loss).backward()
        g_scaler.step(optimizer_gen)
        g_scaler.update()
        
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
    
    # For testing we load model
    if Pix2Pix_Gans_Config.LOAD_MODE:
        # Load generator
        load_model(Pix2Pix_Gans_Config.CHECKPOINT_GEN, gen, optimizer_gen, Pix2Pix_Gans_Config.LEARNING_RATE)
        # Load discriminator
        load_model(Pix2Pix_Gans_Config.CHECKPOINT_DISC, disc, optimizer_disc, Pix2Pix_Gans_Config.LEARNING_RATE)
    
    # Load data
    train_dataset = MapDataset(root_dir='./pix2pix_dataset/maps/maps/train')
    train_loader = DataLoader(train_dataset, batch_size = Pix2Pix_Gans_Config.BATCH_SIZE, shuffle = True, num_workers = Pix2Pix_Gans_Config.NUM_WORKERS)
    
    # Here we try float 16 training, it runs faster
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    validation_dataset = MapDataset(root_dir='./pix2pix_dataset/maps/maps/val')
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=True)

    for epoch in range(Pix2Pix_Gans_Config.NUM_EPOCHS):
        ##############################################################################################
        ##############################################################################################
        ############################### Important things to understand ###############################
        ##############################################################################################
        ##############################################################################################

        # Input is a image and output is the layout
        train(disc, gen, optimizer_disc, optimizer_gen, BCE, L1_LOSS, g_scaler, d_scaler, train_loader)
        # Save the model
        if Pix2Pix_Gans_Config.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(gen, optimizer_gen, file_name = Pix2Pix_Gans_Config.CHECKPOINT_GEN)
            save_checkpoint(disc, optimizer_disc, file_name = Pix2Pix_Gans_Config.CHECKPOINT_DISC)
        
        # Save some examples
        save_some_examples(gen, validation_loader, epoch, folder = 'pix2pix_evaluation')

if __name__ == '__main__':
    main()