import torch
from torch.utils.data import Dataset

import json

class NerfDataset(Dataset):
    def __init__(self, transforms_path, img_path):
        f = open(transforms_path)
        transforms_data = json.load(f)
        
    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass