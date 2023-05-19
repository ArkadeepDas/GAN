# This is for gradient penalty which might occur in training due to weight clipping
# If weight is too low then there is a vanishing gradient problem
# If it is large then it is very difficult and take long time for model to to train the Discriminator/Critic till optimality.

import torch
import torch.nn as nn

def gradient_penalty(critic, real_image, fake_image, device="cpu"):
    # Number of images x Channels x Height x Width
    BATCH_SIZE, C, H, W = real_image.shape
    # epsilon value for each example
    epsilon = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_image = (real_image * epsilon) + (fake_image - (1 - epsilon))

    # Calculate critic scores
    mixed_scores = critic(interpolated_image)

    # Now we are computing the mixed scores with respect to the interpolated images
    # grad_outputs a tensor same shape of output represents the initial gradients for the 'outputs' 
    gradient = torch.autograd.grad(inputs=interpolated_image, outputs=mixed_scores, grad_outputs=torch.ones_like(mixed_scores),
                                   create_graph=True, retain_graph=True)[0]
    # flatten the dimentions
    gradient = gradient.view(gradient.shape[0], -1)
    # Now we calculate the norm
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm-  1) ** 2)
    return gradient_penalty