'''import glob
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
    
    train(args.data_folder, args.save_path)'''


import glob
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import json
import time
import argparse
from cloud1_model import CustomRegNetY002
import logging
from cloud1_dataloader import CarlaDataset

def train(data_folder, save_path):
    device = torch.device('cuda')
    print(device)
    nr_epochs = 200
    batch_size = 16
    start_time = time.time()
    l1_lambda = 0.001 

    # Create the DataLoader
    try:
        full_dataset = CarlaDataset(data_folder) 
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
        model = CustomRegNetY002()  
        model = nn.DataParallel(model)
        model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)  # L2 regularization
        criterion = nn.MSELoss()

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        loss_values = []
        val_loss_values = []  # To track validation loss
        for epoch in range(nr_epochs):
            total_loss = 0

            # Training phase
            model.train()
            for batch_idx, (batch_in, batch_gt) in enumerate(train_loader):
                batch_in = batch_in.to(device)
                batch_gt = batch_gt.to(device)

                # Forward pass
                optimizer.zero_grad()
                batch_out = model(batch_in)
                loss = criterion(batch_out, batch_gt)

                # L1 Regularization
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                loss += l1_lambda * l1_norm 

                # Backward pass and optimization
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
                optimizer.step()
                total_loss += loss.item()

            average_loss = total_loss / (batch_idx + 1)
            loss_values.append(average_loss)
            scheduler.step()

            # Validation phase
            model.eval()  
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
        plt.show() 

    except Exception as e:
        logging.error(f"An error occurred during training: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Imitation Learning Training Script')
    parser.add_argument('-d', '--data_folder', default="/home/h2x/Desktop/IL_DATA_COLLECTION_ADWAIT/Main_script/09-15-2024", type=str, help='Path to your dataset')
    parser.add_argument('-s', '--save_path', default="/home/h2x/Desktop/IL_DATA_COLLECTION_ADWAIT/Main_script/09-15-2024/model.pth", type=str, help='Path to save your model')
    args = parser.parse_args()
    
    train(args.data_folder, args.save_path)
