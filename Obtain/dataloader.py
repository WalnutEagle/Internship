import glob
import numpy as np
import torch
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import Dataset
import os
import json

class CarlaRunDataset(Dataset):
    def __init__(self, run_dir):
        self.img_list = []
        self.depth_list = []
        self.data_list = []
        self.run_dir = run_dir

        # Load RGB images
        rgb_dir = os.path.join(run_dir, 'rgb')
        self.img_list = glob.glob(os.path.join(rgb_dir, '*.jpg'))

        # Load Depth images
        depth_dir = os.path.join(run_dir, 'disparity')
        self.depth_list = glob.glob(os.path.join(depth_dir, '*.png'))

        # Load JSON files for actions
        json_dir = os.path.join(run_dir, 'json')
        self.data_list = glob.glob(os.path.join(json_dir, '*.json'))

        print(f'Loaded {len(self.img_list)} images, {len(self.depth_list)} depth images, and {len(self.data_list)} action files from run: {run_dir}')

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # Load the JSON file
        with open(self.data_list[idx], 'r') as f:
            data = json.load(f)

        # Extract actions (steering and throttle)
        actions = torch.Tensor([data["Steer"], data["Throttle"]])

        # Load the corresponding RGB image
        img_path = self.img_list[idx]
        img = read_image(img_path)
        img = img[:3, 120:600, 400:880]  # Crop the image
        normalized_image = img.float() / 255.0  # Normalize image to [0, 1]

        # Load the corresponding depth image
        depth_path = self.depth_list[idx]
        depth_img = read_image(depth_path)
        depth_img = depth_img.float() / 255.0  # Normalize depth if needed
        depth_img = depth_img.unsqueeze(0)  # Add channel dimension for depth

        # Concatenate RGB and depth images
        combined_image = torch.cat((normalized_image, depth_img), dim=0)

        return combined_image, actions  # Return combined image and actions

def get_run_dataloader(run_dir, batch_size, num_workers=4):
    return torch.utils.data.DataLoader(
        CarlaRunDataset(run_dir),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )
