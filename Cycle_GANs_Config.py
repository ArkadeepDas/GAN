import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TRAIN_DIR = 'CycleGan_Dataset/horse2zebra/horse2zebra'
VALIDATION_DIR = 'CycleGan_Dataset/horse2zebra/horse2zebra'
# In paper they use batch size 1
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCH = 100
LOAD_MODEL = False
SAVE_MODEL = True
# There is two generator and two discriminator
CHECKPOINT_GEN_H = 'gen_h.pth.tar'
CHECKPOINT_GEN_Z = 'gen_z.pth.tar'
CHECKPOINT_CRITIC_H = 'critic_h.pth.tar'
CHECKPOINT_CRITIC_Z = 'critic_z.pth.tar'

transforms = A.Compose(
    [
        A.Resize(width = 256, height = 256),
        A.HorizontalFlip(p = 0.5),
        A.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5], max_pixel_value = 255),
        ToTensorV2()
    ],
    additional_targets = {'image0': 'image'}
)