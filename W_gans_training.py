"""
Now in this script we train our W_GAM network on MINST dataset with Discriminator and Generator imported from W_gans.py
"""

########################################### FOR W-GAN GP Training ###########################################
# LEARNING_RATE will update to 1e-4
# Change Discriminator/Critic Normalization part from BatchNorm2d to InstanceNorm2d or LayerNorm
# Remove WEIGHT_CLIP 
# import gradient_penalty from W_gans_gradient_penalty
# Add LAMBDA_GP = 10
# Change optimizer to Adam line 61 and 62
# Add gradient penalty in line 88
# Gradient penalty loss is added in line 93
#############################################################################################################

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# We are importing both the models and the weight initialization 
from W_gans import Critic, Generator, initialize_weights
from W_gans_gradient_penalty import gradient_penalty

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
critic = Critic(CHANNELS_IMG, FEATURES_DISC).to(device)
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
            critic_fake = critic(fake_image).reshape(-1)
            #############################################################################################
            
            # Gradient penalty is adding for W-Gan GP
            ### gp = gradient_penalty(critic, real_img, fake, device=device)
            # We want to maximize this loss. So we negate it for maximization 
            # The distance between original and fake is always high. That's the idea here
            # For W-GAN GP we add gradient penalty in loss
            ### loss_critic = -((torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp)

            #############################################################################################
            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake))
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()
            
            # One thing we need to apply is clip the parameters
            # Basically cliping/converting the parameters in range of -0.01 to 0.01(Vales from paper)
            for p in critic.parameters():
                # clamp convert the values in some range 
                p.data.clamp(-WEIGHT_CLIP, WEIGHT_CLIP)

        ################################################
        # Now train the Generator
        # Train generator: minimize the distance: min -E[critic(gen_fake)]
        output = critic(fake_image).reshape(-1)
        # Now loss calculation
        loss_gen = -torch.mean(output)
        gen.zero_grad()
        loss_gen.backward(retain_graph=True)
        opt_gen.step()

        # Print losses occationally and print to tensorboard
        # Here main part is we print it to tensorboard to show the output
        # This loss here is called non-saturated heuristic 
        if batch_idx % 100 == 0:
            print(f"EPOCH[{epoch/NUM_EPOCHS}] Batch: {batch_idx}/{len(loader)} Loss D: {loss_critic}, Loss G: {loss_gen}")
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