import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
import glob
from torch.utils.data import DataLoader

from cloud1_model import CustomRegNetY002  # Adjust this to your actual model name
from dataloader import get_run_dataloader  # Import the new dataloader

def train(run_folder, save_path):
    device = torch.device('cuda')
    print(device)
    nr_epochs = 200
    batch_size = 16
    start_time = time.time()

    # Load the run-specific dataset
    train_loader = get_run_dataloader(run_folder, batch_size)

    # Initialize the model
    model = CustomRegNetY002()  # Ensure this matches your model definition
    model = nn.DataParallel(model)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    loss_values = []
    for epoch in range(nr_epochs):
        total_loss = 0

        # Training phase
        model.train()  # Set the model to training mode
        for batch_idx, (batch_in, batch_gt) in enumerate(train_loader):
            batch_in = batch_in.to(device)
            batch_gt = batch_gt.to(device)

            # Forward pass
            optimizer.zero_grad()
            batch_out = model(batch_in)  # Pass the combined RGB and depth images
            loss = criterion(batch_out, batch_gt)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / (batch_idx + 1)
        loss_values.append(average_loss)
        scheduler.step()

        time_per_epoch = (time.time() - start_time) / (epoch + 1)
        print(f"Epoch {epoch + 1}\tLoss: {average_loss:.6f} \tETA: +{time_per_epoch * (nr_epochs - epoch - 1):.2f}s")

    # Save the model for this run
    torch.save(model.state_dict(), save_path)

    # Plot loss values
    plt.figure()
    plt.title(f'Loss Plot for Run: {run_folder}')
    plt.plot(loss_values, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'Loss_Plot_{os.path.basename(run_folder)}.jpg')
    plt.show()  # Display the plot

def main(data_folder, save_base_path):
    # Iterate through each run folder
    for run_dir in glob.glob(os.path.join(data_folder, 'rc_data', 'run_*')):
        print(f"Training on {run_dir}")
        save_path = os.path.join(save_base_path, f'model_{os.path.basename(run_dir)}.pth')
        train(run_dir, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Imitation Learning Training Script for Individual Runs')
    parser.add_argument('-d', '--data_folder', default="/home/h2x/Desktop/IL_DATA_COLLECTION_ADWAIT/Main_script/09-15-2024", type=str, help='Path to your dataset')
    parser.add_argument('-s', '--save_base_path', default="/home/h2x/Desktop/IL_DATA_COLLECTION_ADWAIT/Main_script/09-15-2024", type=str, help='Base path to save your models')
    args = parser.parse_args()
    
    main(args.data_folder, args.save_base_path)
