import torch

# Define the model architecture (this is the CustomResNet18 class you provided)
model = CustomResNet18()

# Load the model weights
model_path = 'path/to/your/model.pth'  # Replace with your actual path
model.load_state_dict(torch.load(model_path))

# Set the model to evaluation mode
model.eval()
from PIL import Image
import torchvision.transforms as transforms

# Load the image
image_path = 'path/to/your/image.jpg'  # Replace with your actual image path
image = Image.open(image_path)

# Define the preprocessing transformation
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match model's expected input size
    transforms.ToTensor(),          # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet mean and std
])

# Preprocess the image
input_image = preprocess(image).unsqueeze(0)  # Add batch dimension
# Example location data (replace with your actual location data)
location_data = torch.tensor([[0.5, -0.2]])  # Shape: [batch_size, 2]

# Make sure the location data is a tensor and matches the expected shape
location_data = location_data.float()  # Ensure it's a float tensor
# Move the model and inputs to the appropriate device (e.g., GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
input_image = input_image.to(device)
location_data = location_data.to(device)

# Run the model
with torch.no_grad():  # Disable gradient calculation for inference
    predicted_actions = model(input_image, location_data)

# Print the predicted actions
print(predicted_actions)
