import torch
from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np

class ZebraHorseDataset(Dataset):
    def __init__(self, root_zebra, root_horse, transform = None):
        super().__init__()
        self.root_zebra = root_zebra
        self.root_horse = root_horse
        self.transform = transform

        self.zebra_images = os.listdir(root_zebra)
        self.horse_images = os.listdir(root_horse)
        # Length of the both data sets are not same. Usually we have x, y pairs
        # So we find the minimum length from both of the datasets
        self.length_dataset = min(len(self.zebra_images), len(self.horse_images))
    
    # It returns the length of the dataset. 'len()' is the function 
    def __len__(self):
        # Return minimum length of the dataset
        return self.length_dataset
    
    def __getitem__(self, index):
        zebra_image = self.zebra_images[index]
        horse_image = self.horse_images[index]
        
        zebra_path = os.path.join(self.root_zebra, zebra_image)
        horse_path = os.path.join(self.root_horse, horse_image)

        output_zebra_image = np.array(Image.open(zebra_path).convert('RGB'))
        output_horse_image = np.array(Image.open(horse_path).convert('RGB'))
        
        # If transform is require then we apply transformation
        if self.transform:
            augmentation = self.transform(iamge = output_zebra_image, image1 = output_horse_image)
            output_zebra_image = augmentation['image']
            output_horse_image = augmentation['image0']

        return output_zebra_image, output_horse_image