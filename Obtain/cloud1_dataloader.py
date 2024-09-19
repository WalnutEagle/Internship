'''import glob
import numpy as np
import torch
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import Dataset
import os

class CarlaDataset(Dataset):
    def __init__(self, data_dir):
        self.img_list = []
        self.disparity_list = []
        self.data_list = []

        # Traverse through the rc_data folder structure
        for run_folder in os.listdir(data_dir):
            run_path = os.path.join(data_dir, run_folder)
            if os.path.isdir(run_path):
                rgb_folder = os.path.join(run_path, 'rgb')
                disparity_folder = os.path.join(run_path, 'disparity')
                action_folder = os.path.join(run_path, 'json')  # Assuming action data is in JSON files

                # Collect RGB images
                self.img_list += glob.glob(os.path.join(rgb_folder, '*.jpg'))

                # Collect disparity images
                self.disparity_list += glob.glob(os.path.join(disparity_folder, '*.png'))

                # Collect corresponding action data
                self.data_list += glob.glob(os.path.join(action_folder, '*.npy'))  # Assuming numpy files contain action data

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        print(f"Total RGB images: {len(self.img_list)}")
        print(f"Total disparity images: {len(self.disparity_list)}")
        print(f"Total action data files: {len(self.data_list)}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # Load actions from numpy file
        data = np.load(self.data_list[idx], allow_pickle=True)
        
        # Load corresponding RGB image
        img = read_image(self.img_list[idx])
        img = img[:3, 120:600, 400:880]  # Crop if necessary
        normalized_image = self.normalize(img.float() / 255.0)

        # Load corresponding disparity image
        disparity_img = read_image(self.disparity_list[idx])
        disparity_img = disparity_img[:3, 120:600, 400:880]  # Crop if necessary
        normalized_disparity = self.normalize(disparity_img.float() / 255.0)

        # Extract actions
        actions = torch.Tensor(data[:2])  # Assuming first two elements are actions
        
        return (normalized_image, normalized_disparity, actions)

def get_dataloader(data_dir, batch_size, num_workers=4):
    return torch.utils.data.DataLoader(
        CarlaDataset(data_dir),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )'''


# import glob
# import numpy as np
# import torch
# from torchvision import transforms
# from torchvision.io import read_image
# from torch.utils.data import Dataset
# import os
# import json

# class CarlaDataset(Dataset):
#     def __init__(self, data_dir):
#         self.img_list = []
#         self.depth_list = []
#         self.data_list = []
#         self.data_dir = data_dir

#         # Traverse through subdirectories
#         for run_dir in glob.glob(os.path.join(self.data_dir, 'rc_data', 'run_*')):
#             # Load RGB images and Depth images
#             rgb_dir = os.path.join(run_dir, 'rgb')
#             depth_dir = os.path.join(run_dir, 'disparity')  # Assuming depth images are in a 'disparity' folder
#             self.img_list += glob.glob(os.path.join(rgb_dir, '*.jpg'))
#             self.depth_list += glob.glob(os.path.join(depth_dir, '*.png'))

#             # Load JSON files for actions
#             json_dir = os.path.join(run_dir, 'json')
#             self.data_list += glob.glob(os.path.join(json_dir, '*.json'))

#         print(f'Loaded {len(self.img_list)} images, {len(self.depth_list)} depth images, and {len(self.data_list)} action files.')

#     def __len__(self):
#         return len(self.data_list)

#     def __getitem__(self, idx):
#         # Load the JSON file
#         with open(self.data_list[idx], 'r') as f:
#             data = json.load(f)

#         # Extract actions
#         actions = torch.Tensor([data["Steer"], data["Throttle"]])

#         # Load the corresponding RGB image
#         img_path = self.img_list[idx]
#         img = read_image(img_path)
#         img = img[:3, 120:600, 400:880]  # Crop the image
#         normalized_image = self.normalize(img.float() / 255.0)

#         # Load the corresponding depth image
#         depth_path = self.depth_list[idx]
#         depth_img = read_image(depth_path)
#         depth_img = depth_img.float() / 255.0  # Normalize depth if needed
#         depth_img = depth_img.unsqueeze(0)  # Add channel dimension if necessary

#         return normalized_image, depth_img, actions

# def get_dataloader(data_dir, batch_size, num_workers=4):
#     return torch.utils.data.DataLoader(
#         CarlaDataset(data_dir),
#         batch_size=batch_size,
#         num_workers=num_workers,
#         shuffle=True
#     )


'''import glob
import numpy as np
import torch
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import Dataset
import os
import json

class CarlaDataset(Dataset):
    def __init__(self, data_dir):
        self.img_list = []
        self.depth_list = []
        self.data_list = []
        self.data_dir = data_dir

        # Traverse through subdirectories
        for run_dir in glob.glob(os.path.join(self.data_dir, 'rc_data', 'run_*')):
            # Load RGB images
            rgb_dir = os.path.join(run_dir, 'rgb')
            self.img_list += glob.glob(os.path.join(rgb_dir, '*.jpg'))

            # Load Depth images
            depth_dir = os.path.join(run_dir, 'disparity')  # Adjust if depth images are in a different folder
            self.depth_list += glob.glob(os.path.join(depth_dir, '*.png'))

            # Load JSON files for actions
            json_dir = os.path.join(run_dir, 'json')
            self.data_list += glob.glob(os.path.join(json_dir, '*.json'))

        print(f'Loaded {len(self.img_list)} images, {len(self.depth_list)} depth images, and {len(self.data_list)} action files.')

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # Load the JSON file
        with open(self.data_list[idx], 'r') as f:
            data = json.load(f)

        # Extract actions
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
        depth_img = depth_img.unsqueeze(0)  # Add channel dimension if necessary

        return normalized_image, depth_img, actions

def get_dataloader(data_dir, batch_size, num_workers=4):
    return torch.utils.data.DataLoader(
        CarlaDataset(data_dir),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )
'''

import glob
import numpy as np
import torch
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import Dataset
import os
import json

class CarlaDataset(Dataset):
    def __init__(self, data_dir):
        self.img_list = []
        self.depth_list = []
        self.data_list = []
        self.data_dir = data_dir

        # Traverse through subdirectories
        for run_dir in glob.glob(os.path.join(self.data_dir, 'rc_data', 'run_*')):
            # Load RGB images
            rgb_dir = os.path.join(run_dir, 'rgb')
            self.img_list += glob.glob(os.path.join(rgb_dir, '*.jpg'))

            # Load Depth images
            depth_dir = os.path.join(run_dir, 'disparity')
            self.depth_list += glob.glob(os.path.join(depth_dir, '*.png'))

            # Load JSON files for actions
            json_dir = os.path.join(run_dir, 'json')
            self.data_list += glob.glob(os.path.join(json_dir, '*.json'))

        print(f'Loaded {len(self.img_list)} images, {len(self.depth_list)} depth images, and {len(self.data_list)} action files.')

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

        # Ensure depth image is squeezed to 2D if it has a channel dimension
        if depth_img.dim() == 3:
            depth_img = depth_img.squeeze(0)  # Remove channel dimension if exists

        # Concatenate RGB and depth images
        combined_image = torch.cat((normalized_image, depth_img.unsqueeze(0)), dim=0)

        return combined_image, actions  # Return combined image and actions


def get_dataloader(data_dir, batch_size, num_workers=4):
    return torch.utils.data.DataLoader(
        CarlaDataset(data_dir),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )

