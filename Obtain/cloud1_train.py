'''import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from cloud1_model import CustomResNet18
from cloud1_dataloader import get_dataloader
import time
import argparse
from torch.utils.tensorboard import SummaryWriter

def train(data_folder, save_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nr_epochs = 200
    batch_size = 16
    start_time = time.time()

    # Initialize the model
    model = CustomResNet18()
    model = nn.DataParallel(model)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    # Load the training data
    train_loader = get_dataloader(data_folder, batch_size)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # TensorBoard writer
    writer = SummaryWriter('runs/model_training')

    loss_values = []
    for epoch in range(nr_epochs):
        total_loss = 0

        for batch_idx, (batch_in, batch_gt1) in enumerate(train_loader):
            batch_in = batch_in.to(device)
            batch_gt1 = batch_gt1.to(device)

            # Forward pass
            optimizer.zero_grad()
            batch_out = model(batch_in)  # Only pass the images
            loss = criterion(batch_out, batch_gt1)

            # Backward pass and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / (batch_idx + 1)
        time_per_epoch = (time.time() - start_time) / (epoch + 1)
        time_left = time_per_epoch * (nr_epochs - 1 - epoch)
        print(f"Epoch {epoch + 1}\t[Train]\tloss: {average_loss:.6f} \tETA: +{time_left:.2f}s")

        # Log loss to TensorBoard
        writer.add_scalar('Loss/train', average_loss, epoch)
        loss_values.append(average_loss)
        scheduler.step()

        # Save model checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            torch.save(model.state_dict(), f"{save_path}/model_epoch_{epoch + 1}.pth")

    # Final model save
    torch.save(model.state_dict(), save_path)

    # Plot loss values
    plt.title('Loss Plot for Cloud Only Model')
    plt.plot(loss_values)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('Loss_Plot.jpg')

    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EC500 Homework1 Imitation Learning')
    parser.add_argument('-d', '--data_folder', default="./", type=str, help='Path to your dataset')
    parser.add_argument('-s', '--save_path', default="./model.pth", type=str, help='Path to save your model')
    args = parser.parse_args()
    
    train(args.data_folder, args.save_path)
'''





import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from cloud1_model import CustomResNet18
from cloud1_dataloader import get_dataloader
import time
import argparse
from torch.utils.tensorboard import SummaryWriter

def train(data_folder, save_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nr_epochs = 200
    batch_size = 16
    start_time = time.time()

    # Initialize the model
    model = CustomResNet18()
    model = nn.DataParallel(model)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    # Load the training data
    train_loader = get_dataloader(data_folder, batch_size)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # TensorBoard writer
    writer = SummaryWriter('runs/model_training')

    loss_values = []
    for epoch in range(nr_epochs):
        total_loss = 0

        for batch_idx, (batch_in, batch_gt1) in enumerate(train_loader):
            batch_in = batch_in.to(device)
            batch_gt1 = batch_gt1.to(device)

            # Forward pass
            optimizer.zero_grad()
            batch_out = model(batch_in)  # Only pass the images
            loss = criterion(batch_out, batch_gt1)

            # Backward pass and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / (batch_idx + 1)
        time_per_epoch = (time.time() - start_time) / (epoch + 1)
        time_left = time_per_epoch * (nr_epochs - 1 - epoch)
        print(f"Epoch {epoch + 1}\t[Train]\tloss: {average_loss:.6f} \tETA: +{time_left:.2f}s")

        # Log loss to TensorBoard
        writer.add_scalar('Loss/train', average_loss, epoch)
        loss_values.append(average_loss)
        scheduler.step()

    # Save everything in a single file
    final_checkpoint = {
        'epoch': nr_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': average_loss,
    }
    torch.save(final_checkpoint, save_path)

    # Plot loss values
    plt.title('Loss Plot for Cloud Only Model')
    plt.plot(loss_values)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('Loss_Plot.jpg')

    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EC500 Homework1 Imitation Learning')
    parser.add_argument('-d', '--data_folder', default="./", type=str, help='Path to your dataset')
    parser.add_argument('-s', '--save_path', default="./model.pth", type=str, help='Path to save your model')
    args = parser.parse_args()
    
    train(args.data_folder, args.save_path)
