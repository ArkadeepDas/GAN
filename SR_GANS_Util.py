import torch
import os
import SR_GANS_Config
import numpy as np
from PIL import Image
from torchvision.utils import save_image

def save_checkpoint(model, optimizer, file_name = 'MyCheckPoint.pth.tar'):
    print('! Saveing Checkpoint !')
    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, file_name)

def load_checkpoint(checkpoint_file, model, optimizer, learning_rate):
    print('! Loading Checkpoint !')
    checkpoint = torch.load(checkpoint_file, map_location = SR_GANS_Config.DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    # Set the learning rate
    for params in optimizer.param_groups:
        params['lr'] = learning_rate

def plot_examples(low_resolution_folder, gen):
    files = os.listdir(low_resolution_folder)
    # Set generator in evaluation mode
    gen.eval()
    for file in files:
        image = Image.open('test_image/' + file)
        image = SR_GANS_Config.lower_transformation(image = np.asanyarray(image)).unsqueeze(0).to(SR_GANS_Config.DEVICE)
        with torch.no_grad():
            upscaled_image = gen(image)
        save_image(upscaled_image * 0.5 + 0.5, f'saved/{file}')
    gen.train()