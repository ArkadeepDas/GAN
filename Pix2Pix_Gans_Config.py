# Config file for Pix2Pix GANs
import torch

# IF gpu is present then we are going to use GPU else CPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# Dataset paths
TRAIN_DIR = 'pix2pix_dataset/maps/maps/train'
VALID_DIR = 'pix2pix_dataset/maps/maps/val'
# Fixed learning rate
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
NUM_WORKERS = 2
IMAGE_SIZE = 256
CHANNEL_IMAGE = 3
L1_LAMBDA = 100
NUM_EPOCHS = 100
LOAD_MODE = False
SAVE_MODEL = False
# Checkpoint for Discriminator and Generator
CHECKPOINT_DISC = 'disc.path.tar'
CHECKPOINT_GEN = 'gen.path.tar'