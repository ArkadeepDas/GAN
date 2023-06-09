import torch
import torchvision
import torch.nn as nn
import Pro_GANs_Config
from torchvision.utils import save_image
from scipy.stats import truncnorm

# Plot outputs to tensorboard
def plot_to_tensorborad(writer, loss_critic, loss_gen, real, fake, tensorboard_step):
    writer.add_scalar('Loss Critic', loss_critic, global_step = tensorboard_step)
    with torch.no_grad():
        # Take out 8 example to plot
        image_grid_real = torchvision.utils.make_grid(real[:8], normalize = True)
        image_grid_fake = torchvision.utils.make_grid(fake[:8], normalize = True)
        writer.add_image("Real", image_grid_real, global_step = tensorboard_step)
        writer.add_image("Fake", image_grid_fake, global_step = tensorboard_step)

# Pro GANs use W-GAN GP loss
def gradient_penalty(critic, real, fake, alpha, train_step, device = 'cpu'):
    BATCH_SIZE, C, H, W = real.shape
    # Create random value and set it to everywhere
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    # Now let's create interpolated image
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad(True)

    # Now calculate the discriminator output
    mixed_output = critic(interpolated_images, alpha, train_step)

    # Take the gradient of the scores with respect to the image
    # We calculate here with positive outcome
    gradient = torch.autograd.grad(inputs = interpolated_images, outputs = mixed_output, grad_outputs = torch.ones_like(mixed_output), create_graph = True, retain_graph = True)[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim = 1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

# Save Checkpoint
def save_checkpoint(model, optimizer, file_name = 'my_checkpoint.pth.tar'):
    print('=> !!!Saving Checkpoint!!! <=')
    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, file_name)

# Load model from saved data
def load_model(checkpoint_file, model, optimizer, learning_rate):
    print('=> !!!Load Checkpoint!!! <=')
    checkpoint = torch.load(checkpoint_file, map_location=Pro_GANs_Config.DEVICE)
    # Load model and optimizer
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    # Now load the learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

def generate_examples(gen, steps, truncation = 0.7, n = 100):
    gen.eval()
    alpha = 1.0
    for i in range(n):
        with torch.no_grad():
            noise = torch.tensor(truncnorm.rvs(-truncation, truncation, size=(1, Pro_GANs_Config.Z_DIM, 1, 1)), device=Pro_GANs_Config.DEVICE, dtype=torch.float32)
            img = gen(noise, alpha, steps)
            save_image(img*0.5+0.5, f"saved_examples/img_{i}.png")
    gen.train()