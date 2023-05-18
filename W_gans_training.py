"""
Now in this script we train our W_GAM network on MINST dataset with Discriminator and Generator imported from W_gans.py
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
from W_gans import Discriminator, Generator, initialize_weights

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 5e-5 # learning rate is different in W-gans
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 1
Z_DIM = 100
NUM_EPOCHS = 5
FEATURES_DISC = 64
FEATURES_GEN = 64
# The called discriminator "critic" in paper
CRITIC_ITERATION = 5
WEIGHT_CLIP = 0.01

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

# Initializing the Models
gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
critic = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)
initialize_weights(gen)
initialize_weights(critic)

# Optimizer is different here for W-GAN
opt_gen = optim.RMSprop(gen.parameters(), lr=LEARNING_RATE)
opt_critic = optim.RMSprop(critic.parameters(), lr=LEARNING_RATE)

# We create noise of 32x100x1x1 for testing purpose
fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)
print(fixed_noise.shape)

# This is for tensorboard
write_real = SummaryWriter(f"logs/real")
write_fake = SummaryWriter(f"logs/fake")
step = 0

# Set both of the network in the training mode
gen.train()
critic.train()

# In out training Discriminator/Critic will train more. So in training loop we increase number of train times of Discriminator/Critic
# Loss function is not simply binary cross entropy here. It's different. We want to maximize the loss here for better work for Discriminator/Critic
for epoch in range(NUM_EPOCHS):
    for batch_idx, (real_img, _) in enumerate(loader):
        real_img = real_img.to(device)
        for _ in range(CRITIC_ITERATION):
            noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1).to(device)
            fake_image = gen(noise)
            critic_real = critic(real_img).reshape(-1)
            critic_fake = critic(fake_image).rehsape(-1)
            # We want to maximize this loss. So we negate it for maximization 
            # The distance between original and fake is always high. That's the idea here
            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake))
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()
            
            # One thing we need to apply is clip the parameters
            for p in critic.parameters():
                p.data.clamp(-WEIGHT_CLIP, WEIGHT_CLIP)