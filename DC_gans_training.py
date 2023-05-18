"""
Now in this script we train our DCGAN network on MINST dataset with Discriminator and Generator imported from DC_gans.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# We are importing both the models and the weight initialization 
from DC_gans import Discriminator, Generator, initialize_weights

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 2e-4
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNELS_IMG = 1
# Noise dimention
Z_DIM = 100
NUM_EPOCHS = 5
FEATURE_D = 64
FEATURE_G = 64

# In PyTorch, transforms.Compose is a class that allows you to chain together multiple image transformations to be applied sequentially to a dataset
transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        # Normalize images with 0.5 mean and 0.5 standard daviation
        transforms.Normalize([0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)])
    ]
)

# Now we load the data
dataset = datasets.MNIST(root='/dataset', train=True, transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

gen = Generator(Z_DIM, CHANNELS_IMG, FEATURE_G).to(device)
disc = Discriminator(CHANNELS_IMG, FEATURE_D).to(device)
initialize_weights(gen)
initialize_weights(disc)

opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
loss = nn.BCELoss()

# We create noise of 32x100x1x1 for testing purpose
fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)
print(fixed_noise.shape)

# This is for tensorboard
write_real = SummaryWriter(f"logs/real")
write_fake = SummaryWriter(f"logs/fake")
step = 0

# Set both of the network in the training mode
gen.train()
disc.train()

for epoch in range(NUM_EPOCHS):
    # We don't need target variable. GANs are unsupervised learning
    for batch_idx, (real_img, _) in enumerate(loader):
        real_img = real_img.to(device)
        # creating noise for training
        noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
        # Now we create Image from noise
        fake_img = gen(noise)

        ### Discriminator train first ###
        # Train discriminator to minimize the loss function
        disc_real = disc(real_img).reshape(-1) # Here we don't get N x 1 x 1 x 1. We get only N
        # Loss calculate for Real with label 1
        loss_disc_real = loss(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake_img).reshape(-1)
        # Loss calculate for fake with 0
        loss_disc_fake = loss(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        disc.zero_grad()
        loss_disc.backward(retain_graph=True)
        opt_disc.step()

        ### Generator train second ###
        # Train Generator to maximize the 1st part of the loss
        output = disc(fake_img).reshape(-1)
        # This loss is calculated with label 1 to maximize the loss
        loss_gen = loss(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward(retain_graph=True)
        opt_gen.step()

        # Print losses occationally and print to tensorboard
        # Here main part is we print it to tensorboard to show the output
        # This loss here is called non-saturated heuristic 
        if batch_idx % 100 == 0:
            print(f"EPOCH[{epoch/NUM_EPOCHS}] Batch: {batch_idx}/{len(loader)} Loss D: {loss_disc}, Loss G: {loss_gen}")
            # We disabling the gradient to test our result
            # No gradient will calculated here
            with torch.no_grad():
                fake = gen(fixed_noise)
                # We took 32 examples
                # Original images grid
                img_grid_real = torchvision.utils.make_grid(real_img[:32], normalize=True)
                # Fake images grid
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                write_real.add_image("Real", img_grid_real, global_step=step)
                write_fake.add_image("Fake", img_grid_fake, global_step=step)

# You can see the output images in Tensor Board