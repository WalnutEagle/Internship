import glob
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
import os
import json
import time
import argparse
from cloud1_model import CustomRegNetY002
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the dataset class
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

        # Ensure all lists have the same length
        min_length = min(len(self.img_list), len(self.depth_list), len(self.data_list))
        self.img_list = self.img_list[:min_length]
        self.depth_list = self.depth_list[:min_length]
        self.data_list = self.data_list[:min_length]

        logging.info(f'Loaded {len(self.img_list)} images, {len(self.depth_list)} depth images, and {len(self.data_list)} action files.')

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        try:
            # Load the JSON file
            with open(self.data_list[idx], 'r') as f:
                data = json.load(f)

            # Extract actions (steering and throttle)
            actions = torch.Tensor([data["Steer"], data["Throttle"]])

            # Load the corresponding RGB image
            img_path = self.img_list[idx]
            img = read_image(img_path)
            
            # Ensure the RGB image is 300x300
            img = img[:3, :, :]  # Keep only the first 3 channels (RGB)
            img = transforms.Resize((300, 300))(img)  # Ensure the image is 300x300
            
            normalized_image = img.float() / 255.0  # Normalize image to [0, 1]

            # Load the corresponding depth image
            depth_path = self.depth_list[idx]
            depth_img = read_image(depth_path)
            depth_img = depth_img.float() / 255.0  # Normalize depth to [0, 1]

            # Resize the depth image to match the RGB image dimensions
            depth_img = transforms.Resize((300, 300))(depth_img)

            # Log image shapes for debugging
            # logging.info(f"RGB image shape: {normalized_image.shape}, Depth image shape: {depth_img.shape}")

            # Concatenate the images
            combined_image = torch.cat((normalized_image, depth_img), dim=0)

            return combined_image, actions  # Return combined image and actions
        except Exception as e:
            logging.error(f"Error processing item {idx}: {str(e)}")
            raise



def train(data_folder, save_path):
    device = torch.device('cuda')
    print(device)
    nr_epochs = 200
    batch_size = 16
    start_time = time.time()

    # Create the DataLoader
    try:
        full_dataset = CarlaDataset(data_folder)  # Create dataset instance
        full_size = len(full_dataset)

        # Split the dataset into training (70%), validation (15%), and testing (15%)
        train_size = int(0.7 * full_size)
        val_size = int(0.15 * full_size)
        test_size = full_size - train_size - val_size
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size])

        # Create DataLoaders for each dataset
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Initialize the model
        model = CustomRegNetY002()  # Ensure this matches your model definition
        model = nn.DataParallel(model)
        model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        loss_values = []
        val_loss_values = []  # To track validation loss
        for epoch in range(nr_epochs):
            total_loss = 0

            # Training phase
            model.train()  # Set the model to training mode
            for batch_idx, (batch_in, batch_gt) in enumerate(train_loader):
                batch_in = batch_in.to(device)
                batch_gt = batch_gt.to(device)

                # Forward pass
                optimizer.zero_grad()
                batch_out = model(batch_in)  # Pass only the combined RGB and depth images
                loss = criterion(batch_out, batch_gt)

                # Backward pass and optimization
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
                optimizer.step()
                total_loss += loss.item()

            average_loss = total_loss / (batch_idx + 1)
            loss_values.append(average_loss)
            scheduler.step()

            # Validation phase
            model.eval()  # Set the model to evaluation mode
            val_total_loss = 0
            with torch.no_grad():
                for val_batch in val_loader:
                    val_batch_in, val_batch_gt = val_batch
                    val_batch_in = val_batch_in.to(device)
                    val_batch_gt = val_batch_gt.to(device)

                    val_outputs = model(val_batch_in)
                    val_loss = criterion(val_outputs, val_batch_gt)
                    val_total_loss += val_loss.item()

            average_val_loss = val_total_loss / len(val_loader)
            val_loss_values.append(average_val_loss)

            time_per_epoch = (time.time() - start_time) / (epoch + 1)
            time_left = time_per_epoch * (nr_epochs - 1 - epoch)
            logging.info(f"Epoch {epoch + 1}\t[Train]\tloss: {average_loss:.6f} \t[Val] loss: {average_val_loss:.6f} \tETA: +{time_left:.2f}s")

        # Save the final model checkpoint
        final_checkpoint = {
            'epoch': nr_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': average_loss,
        }
        torch.save(final_checkpoint, save_path)

        # Plot loss values
        plt.figure()
        plt.title('Loss Plot for RegNet Model')
        plt.plot(loss_values, label='Training Loss')
        plt.plot(val_loss_values, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('Loss_Plot.jpg')
        plt.show()  # Display the plot

    except Exception as e:
        logging.error(f"An error occurred during training: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Imitation Learning Training Script')
    parser.add_argument('-d', '--data_folder', default="/home/h2x/Desktop/IL_DATA_COLLECTION_ADWAIT/Main_script/09-15-2024", type=str, help='Path to your dataset')
    parser.add_argument('-s', '--save_path', default="/home/h2x/Desktop/IL_DATA_COLLECTION_ADWAIT/Main_script/09-15-2024/model.pth", type=str, help='Path to save your model')
    args = parser.parse_args()
    
    train(args.data_folder, args.save_path)