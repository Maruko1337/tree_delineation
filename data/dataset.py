import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class ForestDepthDataset(Dataset):
    def __init__(self, image_paths, label_paths, transform=None):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        label = Image.open(self.label_paths[idx])
        
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label
