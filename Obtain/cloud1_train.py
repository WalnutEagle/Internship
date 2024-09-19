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





# import torch
# import torch.nn as nn
# import numpy as np
# import matplotlib.pyplot as plt
# from cloud1_model import CustomResNet18
# from cloud1_dataloader import get_dataloader
# import time
# import argparse
# from torch.utils.data import random_split

# def train(data_folder, save_path):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     nr_epochs = 200
#     batch_size = 16
#     start_time = time.time()

#     # Load the full dataset
#     full_dataset = get_dataloader(data_folder, batch_size)  # Modify to return the entire dataset

#     # Split the dataset into training (70%), validation (15%), and testing (15%)
#     train_size = int(0.7 * len(full_dataset))
#     val_size = int(0.15 * len(full_dataset))
#     test_size = len(full_dataset) - train_size - val_size
#     train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

#     # Create DataLoaders
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
#     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#     # Initialize the model
#     model = CustomResNet18()
#     model = nn.DataParallel(model)
#     model.to(device)
    
#     optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
#     criterion = nn.MSELoss()
    
#     # Learning rate scheduler
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
#     loss_values = []
#     val_loss_values = []  # To track validation loss
#     for epoch in range(nr_epochs):
#         total_loss = 0

#         # Training phase
#         model.train()  # Set the model to training mode
#         for batch_idx, (batch_in, batch_gt1) in enumerate(train_loader):
#             batch_in = batch_in.to(device)
#             batch_gt1 = batch_gt1.to(device)

#             # Forward pass
#             optimizer.zero_grad()
#             batch_out = model(batch_in)  # Only pass the images
#             loss = criterion(batch_out, batch_gt1)

#             # Backward pass and optimization
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
#             optimizer.step()
#             total_loss += loss.item()

#         average_loss = total_loss / (batch_idx + 1)
#         loss_values.append(average_loss)
#         scheduler.step()

#         # Validation phase
#         model.eval()  # Set the model to evaluation mode
#         val_total_loss = 0
#         with torch.no_grad():
#             for val_batch in val_loader:
#                 val_batch_in, val_batch_gt1 = val_batch
#                 val_batch_in = val_batch_in.to(device)
#                 val_batch_gt1 = val_batch_gt1.to(device)

#                 val_outputs = model(val_batch_in)
#                 val_loss = criterion(val_outputs, val_batch_gt1)
#                 val_total_loss += val_loss.item()

#         average_val_loss = val_total_loss / len(val_loader)
#         val_loss_values.append(average_val_loss)

#         time_per_epoch = (time.time() - start_time) / (epoch + 1)
#         time_left = time_per_epoch * (nr_epochs - 1 - epoch)
#         print(f"Epoch {epoch + 1}\t[Train]\tloss: {average_loss:.6f} \t[Val] loss: {average_val_loss:.6f} \tETA: +{time_left:.2f}s")

#     # Save everything in a single file
#     final_checkpoint = {
#         'epoch': nr_epochs,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'loss': average_loss,
#     }
#     torch.save(final_checkpoint, save_path)

#     # Plot loss values
#     plt.figure()
#     plt.title('Loss Plot for Cloud Only Model')
#     plt.plot(loss_values, label='Training Loss')
#     plt.plot(val_loss_values, label='Validation Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.savefig('Loss_Plot.jpg')
#     plt.show()  # Display the plot

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='EC500 Homework1 Imitation Learning')
#     parser.add_argument('-d', '--data_folder', default="/home/h2x/Desktop/IL_DATA_COLLECTION_ADWAIT/Main_script/09-15-2024", type=str, help='Path to your dataset')
#     parser.add_argument('-s', '--save_path', default="/home/h2x/Desktop/IL_DATA_COLLECTION_ADWAIT/Main_script/09-15-2024/model.pth", type=str, help='Path to save your model')
#     args = parser.parse_args()
    
#     train(args.data_folder, args.save_path)

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from cloud1_model import CustomRegNetY002  # Ensure this matches your model class name
from cloud1_dataloader import get_dataloader  # Import the DataLoader function
import time
import argparse
from torch.utils.data import random_split

def train(data_folder, save_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nr_epochs = 200
    batch_size = 16
    start_time = time.time()

    # Create the DataLoader for the full dataset
    full_dataset = get_dataloader(data_folder, batch_size)

    # Split the dataset into training (70%), validation (15%), and testing (15%)
    full_size = len(full_dataset.dataset)  # Access the dataset length directly
    train_size = int(0.7 * full_size)
    val_size = int(0.15 * full_size)
    test_size = full_size - train_size - val_size

    # Perform the split
    train_dataset, val_dataset, test_dataset = random_split(full_dataset.dataset, [train_size, val_size, test_size])

    # Create DataLoaders for each dataset
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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
            batch_out = model(batch_in)  # Pass combined RGB and depth images
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
        print(f"Epoch {epoch + 1}\t[Train]\tloss: {average_loss:.6f} \t[Val] loss: {average_val_loss:.6f} \tETA: +{time_left:.2f}s")

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Imitation Learning Training Script')
    parser.add_argument('-d', '--data_folder', default="/home/h2x/Desktop/IL_DATA_COLLECTION_ADWAIT/Main_script/09-15-2024", type=str, help='Path to your dataset')
    parser.add_argument('-s', '--save_path', default="/home/h2x/Desktop/IL_DATA_COLLECTION_ADWAIT/Main_script/09-15-2024/model.pth", type=str, help='Path to save your model')
    args = parser.parse_args()
    
    train(args.data_folder, args.save_path)




'''import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from cloud1_model import CustomResNet18
from cloud1_dataloader import get_dataloader
import time
import argparse
from torch.utils.data import random_split

def train(data_folder, save_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nr_epochs = 200
    batch_size = 16
    start_time = time.time()

    # Load the full dataset (without creating a DataLoader yet)
    full_dataset = get_dataloader(data_folder, batch_size)

    # Split the dataset into training (70%), validation (15%), and testing (15%)
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    # Create DataLoaders for each dataset
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    model = CustomResNet18()
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
        for batch_idx, (batch_in, batch_depth, batch_gt) in enumerate(train_loader):
            batch_in = batch_in.to(device)
            batch_depth = batch_depth.to(device)
            batch_gt = batch_gt.to(device)

            # Forward pass
            optimizer.zero_grad()
            batch_out = model(batch_in, batch_depth)  # Pass both RGB and depth images
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
                val_batch_in, val_batch_depth, val_batch_gt = val_batch
                val_batch_in = val_batch_in.to(device)
                val_batch_depth = val_batch_depth.to(device)
                val_batch_gt = val_batch_gt.to(device)

                val_outputs = model(val_batch_in, val_batch_depth)
                val_loss = criterion(val_outputs, val_batch_gt)
                val_total_loss += val_loss.item()

        average_val_loss = val_total_loss / len(val_loader)
        val_loss_values.append(average_val_loss)

        time_per_epoch = (time.time() - start_time) / (epoch + 1)
        time_left = time_per_epoch * (nr_epochs - 1 - epoch)
        print(f"Epoch {epoch + 1}\t[Train]\tloss: {average_loss:.6f} \t[Val] loss: {average_val_loss:.6f} \tETA: +{time_left:.2f}s")

    # Save everything in a single file
    final_checkpoint = {
        'epoch': nr_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': average_loss,
    }
    torch.save(final_checkpoint, save_path)

    # Plot loss values
    plt.figure()
    plt.title('Loss Plot for Cloud Only Model')
    plt.plot(loss_values, label='Training Loss')
    plt.plot(val_loss_values, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('Loss_Plot.jpg')
    plt.show()  # Display the plot

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EC500 Homework1 Imitation Learning')
    parser.add_argument('-d', '--data_folder', default="/home/h2x/Desktop/IL_DATA_COLLECTION_ADWAIT/Main_script/09-15-2024", type=str, help='Path to your dataset')
    parser.add_argument('-s', '--save_path', default="/home/h2x/Desktop/IL_DATA_COLLECTION_ADWAIT/Main_script/09-15-2024/model.pth", type=str, help='Path to save your model')
    args = parser.parse_args()
    
    train(args.data_folder, args.save_path)
'''