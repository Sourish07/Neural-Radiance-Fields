import torch
from torch.utils.data import Dataset
import os
import numpy as np
import wget


class TinyCybertruckDataset(Dataset):
    def __init__(self, split="train"):
        self.file_name = 'cybertruck100_cycles_no_hdri'
        self.split = split

        data = np.load(f'{self.file_name}.npz')
        images = data["images"]
        poses = data["poses"]
        focal = data["focal"]
        
        self.images = images[:-1] if split == "train" else images[-1:]
        self.poses = poses[:-1] if split == "train" else poses[-1:]
        self.focal = focal

    def get_near_far(self):
        return 3, 7
    
    def get_image_size(self):
        first_img = self.images[0]
        return first_img.shape[0], first_img.shape[1]

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        return (torch.tensor(self.images[index], dtype=torch.float32), 
                torch.tensor(self.poses[index], dtype=torch.float32), 
                torch.tensor(self.focal, dtype=torch.float32),)


class TinyLegoDataset(Dataset):
    def __init__(self, split="train"):
        if not os.path.exists('tiny_nerf_data.npz'):
            url = 'http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz'
            wget.download(url, 'tiny_nerf_data.npz')

        self.split = split

        data = np.load('tiny_nerf_data.npz')
        images = data["images"]
        poses = data["poses"]
        focal = data["focal"]

        self.images = images[:-1] if split == "train" else images[-1:]
        self.poses = poses[:-1] if split == "train" else poses[-1:]
        self.focal = focal

    def get_near_far(self):
        return 2, 6
    
    def get_image_size(self):
        first_img = self.images[0]
        return first_img.shape[0], first_img.shape[1]

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        return (torch.tensor(self.images[index], dtype=torch.float32), 
                torch.tensor(self.poses[index], dtype=torch.float32), 
                torch.tensor(self.focal, dtype=torch.float32),)
