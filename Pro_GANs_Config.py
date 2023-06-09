import torch
from math import log2

START_TRAINING_AT_IMG_SIZE = 4
DATASET = 'ProGan_Dataset/celeba_hq'
CHECKPOINT_GEN = 'generator.pth'
CHECKPOINT_CRITIC = 'critic.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_MODEL = True
LOAD_MODEL = False
LEARNING_RATE = 1e-3
# Change depending upon memory
BATCH_SIZE = [16, 16, 16, 16, 16, 16, 16, 8, 4]
IMAGE_SIZE = 512 # Max image size we are going to produce
CHANNELS_IMAGE = 3
Z_DIM = 256 # Original paper 512
IN_CHANNELS = 256
CRITIC_ITERATIONS = 1
LAMBDA_GP = 10
# 30 for every size of images
PROGRESSIVE_EPOCHS = [30] * len(BATCH_SIZE)
FIXED_NOISE = torch.randn(8, Z_DIM, 1, 1).to(DEVICE)
NUM_WORKERS = 4