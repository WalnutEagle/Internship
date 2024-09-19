'''import torch
import torchvision.transforms as transforms
from torchvision.io import read_image

class CustomRegNetY002(nn.Module):
    def __init__(self):
        super(CustomRegNetY002, self).__init__()
        self.model = timm.create_model('regnety_002', pretrained=False)  # Change to pretrained=False

        # Modify the first convolution layer to accept 4 channels (RGB + Depth)
        self.model.stem.conv = nn.Conv2d(
            4,
            self.model.stem.conv.out_channels, 
            kernel_size=self.model.stem.conv.kernel_size, 
            stride=self.model.stem.conv.stride, 
            padding=self.model.stem.conv.padding,
            bias=False
        )

        # Define the final layers for output (steering and throttle)
        self.lin = nn.Sequential(
            nn.Linear(self.model.head.fc.in_features, 512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.5), 
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.5),  
            nn.Linear(256, 2)  # 2 outputs: steering and throttle
        )

    def forward(self, x):
        x = self.model.stem(x)
        x = self.model.s1(x)
        x = self.model.s2(x)
        x = self.model.s3(x)
        x = self.model.s4(x)
        x = self.model.final_conv(x)
        x = self.model.head.global_pool(x)
        x = self.lin(x)
        return x

def preprocess_and_predict(model, rgb_image_path, depth_image_path):
    model.eval()  # Set the model to evaluation mode
    
    # Load and preprocess the RGB image
    img = read_image(rgb_image_path)
    img = img[:3, :, :]  # Keep only the RGB channels
    img = transforms.Resize((300, 300))(img)  # Resize to match model input
    normalized_image = img.float() / 255.0  # Normalize to [0, 1]

    # Load and preprocess the depth image
    depth_img = read_image(depth_image_path)
    depth_img = depth_img.float() / 255.0  # Normalize depth to [0, 1]
    depth_img = transforms.Resize((300, 300))(depth_img)

    # Combine RGB and depth images
    combined_image = torch.cat((normalized_image, depth_img), dim=0)

    # Add batch dimension
    combined_image = combined_image.unsqueeze(0)  # Shape: [1, 4, 300, 300]

    # Make a prediction
    with torch.no_grad():
        output = model(combined_image)
    
    return output

# Example usage
if __name__ == "__main__":
    model = CustomRegNetY002()  # Initialize the model

    rgb_image_path = 'path/to/your/image.jpg'
    depth_image_path = 'path/to/your/depth_image.png'

    prediction = preprocess_and_predict(model, rgb_image_path, depth_image_path)
    print("Predicted output (steering, throttle):", prediction)'''


import torch
import torchvision.transforms as transforms
from torchvision.io import read_image
import timm
import torch.nn as nn

class CustomRegNetY002(nn.Module):
    def __init__(self):
        super(CustomRegNetY002, self).__init__()
        self.model = timm.create_model('regnety_002', pretrained=False)

        # Modify the first convolution layer to accept 4 channels (RGB + Depth)
        self.model.stem.conv = nn.Conv2d(
            4,
            self.model.stem.conv.out_channels, 
            kernel_size=self.model.stem.conv.kernel_size, 
            stride=self.model.stem.conv.stride, 
            padding=self.model.stem.conv.padding,
            bias=False
        )

    def forward(self, x):
        outputs = {}
        x = self.model.stem(x)
        outputs['stem'] = x
        x = self.model.s1(x)
        outputs['s1'] = x
        x = self.model.s2(x)
        outputs['s2'] = x
        x = self.model.s3(x)
        outputs['s3'] = x
        x = self.model.s4(x)
        outputs['s4'] = x
        x = self.model.final_conv(x)
        outputs['final_conv'] = x
        x = self.model.head.global_pool(x)
        outputs['global_pool'] = x
        return outputs

def preprocess_image(rgb_image_path, depth_image_path):
    # Load and preprocess the RGB image
    img = read_image(rgb_image_path)
    img = img[:3, :, :]  # Keep only the RGB channels
    img = transforms.Resize((300, 300))(img)  # Resize to match model input
    normalized_image = img.float() / 255.0  # Normalize to [0, 1]

    # Load and preprocess the depth image
    depth_img = read_image(depth_image_path)
    depth_img = depth_img.float() / 255.0  # Normalize depth to [0, 1]
    depth_img = transforms.Resize((300, 300))(depth_img)

    # Combine RGB and depth images
    combined_image = torch.cat((normalized_image, depth_img), dim=0)

    # Add batch dimension
    combined_image = combined_image.unsqueeze(0)  # Shape: [1, 4, 300, 300]

    return combined_image

# Example usage
if __name__ == "__main__":
    model = CustomRegNetY002()  # Initialize the model

    rgb_image_path = '/home/h2x/Desktop/IL_DATA_COLLECTION_ADWAIT/Main_script/09-15-2024/rc_data/run_001/rgb/000000197_rgb.jpg'
    depth_image_path = '/home/h2x/Desktop/IL_DATA_COLLECTION_ADWAIT/Main_script/09-15-2024/rc_data/run_001/disparity/000000197_disparity.png'

    combined_image = preprocess_image(rgb_image_path, depth_image_path) 
    outputs = model(combined_image) 

    # Print the outputs from each layer
    for layer, output in outputs.items():
        print(f"{layer} output shape: {output.shape}")
