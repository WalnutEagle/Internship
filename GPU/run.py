import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time  # Import time module for measuring time

# Check if CUDA is available, if not fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define a simple CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Define training function
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # Move data and target to the dedicated GPU
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]\tLoss: {loss.item():.6f}')

# Define test function
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # Move data and target to the dedicated GPU
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)')

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('../data', train=False, transform=transform)

# Ensure the data is loaded onto the dedicated GPU
pin_memory = True if torch.cuda.is_available() else False  # Use pinned memory if using CUDA
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=pin_memory)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, pin_memory=pin_memory)

# Instantiate model and optimizer
model = CNN().to(device)
optimizer = optim.Adam(model.parameters())

# Measure total training and testing time
total_train_time = 0
total_test_time = 0

# Train and test the model
for epoch in range(1, 5):  # Train for 4 epochs
    start_train_time = time.time()  # Start training time
    train(model, device, train_loader, optimizer, epoch)
    end_train_time = time.time()  # End training time
    total_train_time += (end_train_time - start_train_time)

    start_test_time = time.time()  # Start testing time
    test(model, device, test_loader)
    end_test_time = time.time()  # End testing time
    total_test_time += (end_test_time - start_test_time)

# Print total times
print(f"\nTotal training time: {total_train_time:.2f} seconds")
print(f"Total testing time: {total_test_time:.2f} seconds")
