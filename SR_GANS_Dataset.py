import os
import numpy as np
import SR_GANS_Config
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class MyImageFolder(Dataset):
    def __init__(self, root_dir):
        super().__init__()
        self.data = []
        self.class_name = os.listdir(root_dir)

        for index, name in enumerate(self.class_name):
            files = os.listdir(os.path.join(root_dir, name))
            self.data += list(zip(files, [index] * len(files)))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image_file, label = self.data[index]
        root_and_dir = os.path.join(self.root_dir, self.class_name[label])
        image = np.array(Image.open(os.path.join(root_and_dir, image_file)))
        image_high_res = SR_GANS_Config.highres_transformation(image = image)['image']
        image_low_res = SR_GANS_Config.lower_transformation(image = image)['image']
        return image_low_res, image_high_res