# Here we are taking only MAP data
# WE need to load the image and split it by 1/2. Then we can decide inpute and target.
import cv2
import numpy as np
import os
from torch.utils.data import Dataset

class MapDataset(Dataset):
    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)
        print(self.list_files)
    
    def __len__(self):
        return len(self.list_files)
    
    def __getitem__(self, index):
        image_file = self.list_files[index]
        image_path = os.path.join(self.root_dir, image_file)
        image = cv2.imread(image_path)
        # channels x width x height
        # We have to devide the image through width
        # Width size of image is 1200
        input_image = image[: , : 600, :]
        input_image = cv2.resize(input_image, (256, 256))
        target_image = image[: , 600: , :]
        target_image = cv2.resize(target_image, (256, 256))
        return input_image, target_image
