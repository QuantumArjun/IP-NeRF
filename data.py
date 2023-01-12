import torch
import torchvision
from torch.utils.data import Dataset

import json
import os

class NerfDataset(Dataset):
    def __init__(self, path, split_type="train"):
        self.path = path
        transforms_path = self._get_paths(path, split_type)
        f = open(transforms_path)
        transforms_data = json.load(f)
        self.camera_angle = transforms_data["camera_angle_x"]
        self.frames = transforms_data["frames"]
    
    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        rotation = frame['rotation']
        transform_matrix = frame['transform_matrix']

        img_path = os.path.join(self.path, frame['file_path'])
        print(img_path)
        return torchvision.io.read_image(f"{img_path}.png")

    def _get_paths(self, path, split_type):
        transforms_path = os.path.join(path, f"transforms_{split_type}.json")
        return transforms_path