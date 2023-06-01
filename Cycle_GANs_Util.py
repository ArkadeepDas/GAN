import torch
import Pix2Pix_Gans_Config
# PyTorch's torchvision library specifically designed for saving tensors or batches of images to disk as image files
from torchvision.utils import save_image

# Save some of the examples from Generator
def save_some_examples(gen, val_loader, epoch, folder):
    x, y = next(iter(val_loader))
    x, y = x.to(Pix2Pix_Gans_Config.DEVICE), y.to(Pix2Pix_Gans_Config.DEVICE)
    # Set the model as evaluation mode
    gen.eval()
    # We just predict/test here. No training is happening here
    with torch.no_grad():
        y_fake = gen(x)
        # Remove Normalization
        y_fake = (y_fake * 0.5) + 0.5
        x = (x * 0.5) + 0.5
        save_image(y_fake, folder + f"/output_{epoch}.png" )
        save_image(x, folder + f"/input_{epoch}.png")
        if epoch == 1:
            y = (y * 0.5) + 0.5
            save_image(y, folder + f"/label{epoch}.png")
    
    # Set the model to train again
    gen.train()

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
    checkpoint = torch.load(checkpoint_file, map_location=Pix2Pix_Gans_Config.DEVICE)
    # Load model and optimizer
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    # Now load the learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate