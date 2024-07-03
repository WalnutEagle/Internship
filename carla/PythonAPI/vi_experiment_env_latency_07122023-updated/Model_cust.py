import torch
import torch.nn.functional as F
import torch.nn as nn

class ResidualTransposedConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualTransposedConv, self).__init__()
        self.conv_transpose1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_transpose2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_transpose3 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)  # Added for matching dimensions in residual connection

    def forward(self, x):
        residual = self.conv_transpose3(x)
        x = F.relu(self.conv_transpose1(x))
        x = F.relu(self.conv_transpose2(x))
        return x + residual

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.final_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)  # Added based on your diagram
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, width, height = x.size()
        query = self.query_conv(x).view(batch_size, -1, width*height).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, width*height)
        value = self.value_conv(x).view(batch_size, -1, width*height)

        energy = torch.bmm(query, key)
        attention = self.softmax(energy)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.final_conv(out)  # Apply the final convolution to the attention output

        return out

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=4)
        # Quantization can be applied during post-processing

    def forward(self, x):
        x = self.conv(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels):
        super(Decoder, self).__init__()
        self.self_attention1 = SelfAttention(in_channels)  # First self-attention layer
        self.residual_transposed_conv1 = ResidualTransposedConv(in_channels, 64)
        
        # Assuming the output from the first set of operations feeds into the second self-attention layer
        self.self_attention2 = SelfAttention(64)  # Second self-attention layer
        self.residual_transposed_conv2 = ResidualTransposedConv(64, 32)
        
        # Final layer to adjust channel dimensions to match the desired output (e.g., RGB image)
        self.final_conv = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding = 1)

    def forward(self, x):
        x = self.self_attention1(x)
        x = self.residual_transposed_conv1(x)
        
        x = self.self_attention2(x)  # Second self-attention layer's output
        x = self.residual_transposed_conv2(x)
        
        x = self.final_conv(x)
        return x


class EdgeModel(nn.Module):
    def __init__(self):
        super(EdgeModel, self).__init__()
        self.encoder = Encoder(in_channels=3, out_channels=64)  
        self.decoder = Decoder(in_channels=64)  

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    