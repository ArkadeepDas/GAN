import torch
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from Pro_GANs_Util import gradient_penalty, plot_to_tensorborad, save_checkpoint, load_model, generate_examples
from Pro_GANs_Model import Discriminator, Generator
from math import log2
from tqdm import tqdm
import Pro_GANs_Config

# Dataset loader
def get_loader(image_size):
     transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(
                [0.5 for _ in range(Pro_GANs_Config.CHANNELS_IMG)],
                [0.5 for _ in range(Pro_GANs_Config.CHANNELS_IMG)],
            )
        ]
    )
     batch_size = Pro_GANs_Config.BATCH_SIZES[int(log2(image_size / 4))]
     dataset = datasets.ImageFolder(root=Pro_GANs_Config.DATASET, transform = transform)
     loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Pro_GANs_Config.NUM_WORKERS,
        pin_memory=True,
    )
     return loader, dataset

# Train function
def train(critic, gen, loader, dataset, step, alpha, opt_critic, opt_gen, tensorboard_step, writer):
    loop = tqdm(loader, leave = True)
    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(Pro_GANs_Config.DEVICE)
        current_batch_size = real.shape[0]

        # Train Critic
        # Similar loss of W-GAN-GP
        # Create the noise
        noise = torch.randn(current_batch_size, Pro_GANs_Config.Z_DIM, 1, 1).to(Pro_GANs_Config.DEVICE)
        
        # Train Discriminator
        fake = gen(noise, alpha, step)
        # Real image
        critic_real = critic(real, alpha, step)
        # Generated image
        critic_fake = critic(fake.detach(), alpha, step)
        gp = gradient_penalty(critic, real, fake, alpha, device = (Pro_GANs_Config.DEVICE))
        # We want to increase the distance(Maximize)
        loss_critic = -((torch.mean(critic_real)) - torch.mean(critic_fake)) + Pro_GANs_Config.LAMBDA_GP * gp + (0.001 * torch.mean(critic_real ** 2))
        opt_critic.zero_grad()
        loss_critic.backword()
        opt_critic.step()

        # Train Generator
        gen_fake = critic(fake, alpha, step)
        loss_gen = -torch.mean(gen_fake)
        opt_gen.zero_grad()
        loss_gen.bachward()
        opt_gen.step()

        # Alpha value update
        alpha = alpha + current_batch_size / (len(dataset) * Pro_GANs_Config.PROGRESSIVE_EPOCHS[step] * 0.5)
        # Alpha not grater than 1
        alpha = min(alpha, 1)

        #############################################################################################################
        # If we want to reuse the image generated from the model then we have to use .detach()

        # Testing outputs
        if batch_idx % 500 == 0:
            # Evaluation without tracking gradient
            with torch.no_grad():
                fixed_fake = gen(Pro_GANs_Config.FIXED_NOISE, alpha, step) * 0.5 + 0.5
            plot_to_tensorborad(
                writer, loss_critic.item(), loss_gen.item(), real.detach(), fixed_fake.detach(), tensorboard_step
            )
    
    return tensorboard_step, alpha
 
# Main Function
def main():
    # Initializing generator and discriminator
    gen = Generator(z_dim = Pro_GANs_Config.Z_DIM, in_channels = Pro_GANs_Config.IN_CHANNELS, image_channels = Pro_GANs_Config.CHANNELS_IMAGE).to(Pro_GANs_Config.DEVICE)
    critic = Discriminator(in_channels = Pro_GANs_Config.IN_CHANNELS, image_channels = Pro_GANs_Config.CHANNELS_IMAGE).to(Pro_GANs_Config.DEVICE)

    # Initializing optimizers
    opt_gen = optim.Adam(gen.parameters(), lr = Pro_GANs_Config.LEARNING_RATE, betas = (0.0, 0.99))
    opt_critic = optim.Adam(critic.parameters(), lr = Pro_GANs_Config.LEARNING_RATE, betas = (0.0, 0.99))
    
    # Writer for Tensorboard
    write = SummaryWriter(f'logs/GAN')

    if Pro_GANs_Config.LOAD_MODEL:
        load_model(
            Pro_GANs_Config.CHECKPOINT_GEN, gen, opt_gen, Pro_GANs_Config.LEARNING_RATE
        )
        load_model(
            Pro_GANs_Config.CHECKPOINT_CRITIC, critic, opt_critic, Pro_GANs_Config.LEARNING_RATE
        )

    # Set both of them to train mode
    gen.train()
    critic.train()
    tensorboard_step = 0
    # Start our image size
    step = int(log2(Pro_GANs_Config.START_TRAINING_AT_IMG_SIZE / 4))

    for num_epochs in Pro_GANs_Config.PROGRESSIVE_EPOCHS[step: ]:
       alpha = 1e-5
       loader, dataset = get_loader(4 * 2 ** step)
       print('Image Size: ', (4 * 2 ** step))
       for epoch in range(num_epochs):
           print(f'Epoch [{epoch+1}] / {num_epochs}')
           tensorboard_step, alpha = train(critic, gen, loader, dataset, step, alpha, opt_critic, opt_gen, tensorboard_step, write)
       # Progressively increase the step 
       step = step + 1

if __name__ == '__main__':
    main()