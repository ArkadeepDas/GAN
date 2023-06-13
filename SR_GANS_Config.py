import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_GEN = 'gen.pth'
CHECKPOINT_DISC = 'disc.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
BATCH_SIZE = 16
LAMBDA_GP = 10
NUM_WORKERS = 4
HIGH_RES = 128
LOW_RES = HIGH_RES // 4
IMAGE_CHANNELS = 3

highres_transformation = A.Compose(
    [
        A.Normalize(mean = [0, 0, 0], std = [1, 1, 1]),
        ToTensorV2()
    ]
)

lower_transformation = A.Compose(
    [
        A.Resize(wide = LOW_RES, height = LOW_RES, interpolation = Image.BICUBIC),
        A.Normalize(mean = [0, 0, 0], std = [1, 1, 1])
    ]
)