import torch
import torchvision
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as tf


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
        img_path = os.path.join(os.getcwd(), self.path, f"{frame['file_path']}.png")
        image = Image.open(img_path).convert('RGB')

        transforms = tf.Compose([tf.Resize(400), 
                                tf.ToTensor()])
        img_tensor = transforms(image)

        
        transform_matrix = torch.Tensor(frame['transform_matrix'])
        assert transform_matrix.shape == (4,4)
        # Calculate position and rotation using https://math.stackexchange.com/questions/82602/how-to-find-camera-position-and-rotation-from-a-4x4-matrix
        R = transform_matrix[:3, :3]
        T = transform_matrix[:3, 3]
        cam_pos = - R.T @ T
        cam_angles = R.T @ torch.Tensor([0, 0, 1])
        # [H, W, 1], [x, y, z], 3d-vector where camera points
        return img_tensor, cam_pos, cam_angles

    def _get_paths(self, path, split_type):
        transforms_path = os.path.join(path, f"transforms_{split_type}.json")
        return transforms_path