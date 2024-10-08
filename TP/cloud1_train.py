import torch
#import math
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from cloud1_model import CustomResNet18
from cloud1_dataloader import get_dataloader

import time
import random
import argparse
def train(data_folder, save_path):
    """
    Function for training the network. You can make changes (e.g., add validation dataloader, change batch_size and #of epoch) accordingly.
    """
    
    device = torch.device('cuda')
    nr_epochs = 200
    batch_size =16
    #nr_of_classes = 4  # needs to be changed
    start_time = time.time()

    infer_action = CustomResNet18()
    infer_action1=nn.DataParallel(infer_action)
    infer_action1.to(device)
    infer_action1.requires_grad = True
    optimizer = torch.optim.AdamW(infer_action.parameters(), lr=1e-4)
    criterion=nn.L1Loss()
    #optimizer = torch.optim.SGD(infer_action.parameters(), lr=0.01)
    

    train_loader = get_dataloader(data_folder, batch_size)
    #print(train_loader.shape)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)#Change
    loss_values=[]
    for epoch in range(nr_epochs):
        total_loss = 0
        batch_in = []
        batch_gt = []
        #print("Hi")

        for batch_idx, batch in enumerate(train_loader):
            batch_in, batch_gt1, batch_gt2 = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            #print(batch_gt.shape)
            #print("Hello")
            #print(batch_in)
            batch_out = infer_action1(batch_in, batch_gt2)
            loss = criterion(batch_out, batch_gt1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        #average_loss=total_loss/(batch_idx+1)
        time_per_epoch = (time.time() - start_time) / (epoch + 1)
        time_left = (1.0 * time_per_epoch) * (nr_epochs - 1 - epoch)
        print("Epoch %5d\t[Train]\tloss: %.6f \tETA: +%fs" % (
            epoch + 1, total_loss, time_left))
        loss_values.append(total_loss)
        scheduler.step()#Change
    torch.save(infer_action, save_path)
    #torch.save(infer_action1,"Hi.pth")
    plt.title('Loss Plot for Cloud Only Model')
    plt.plot(loss_values)#plotting loss values
    plt.savefig('/data2/kathakoli/carla/PythonAPI/vi_experiment_env_latency_07122023/Loss_Plot_14_11.jpg')#saving the plot



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EC500 Homework1 Imitation Learning')
    parser.add_argument('-d', '--data_folder', default="./", type=str, help='path to where you save the dataset you collect')
    parser.add_argument('-s', '--save_path', default="./", type=str, help='path where to save your model in .pth format')
    args = parser.parse_args()
    
    train(args.data_folder, args.save_path)