import torch
from torch import nn

# feature learning
class Layer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1):
        super(Layer, self).__init__()
        
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding)
        self.bn   = nn.BatchNorm2d(num_features=out_channel)
        self.relu = nn.ReLU()
        
    def forward(self,input):
        output = self.conv(input)
        output = self.relu(output)
        output = self.bn(output)
        return output
    
class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        
        self.features = nn.Sequential(
            Layer(3,  32,  kernel_size=3, stride=1, padding=1),
            Layer(32, 32, kernel_size=3, stride=1, padding=1),
            Layer(32, 32, kernel_size=3, stride=1, padding=1),

            nn.MaxPool2d(kernel_size=2, stride=2),

            Layer(32, 64, kernel_size=3, stride=1, padding=1),
            Layer(64, 64, kernel_size=3, stride=1, padding=1),
            Layer(64, 64, kernel_size=3, stride=1, padding=1),

            nn.MaxPool2d(kernel_size=2, stride=2),

            Layer(64,  128, kernel_size=3, stride=1, padding=1),
            Layer(128, 128, kernel_size=3, stride=1, padding=1),
            Layer(128, 128, kernel_size=3, stride=1, padding=1),

            nn.MaxPool2d(kernel_size=2, stride=2),

            Layer(128, 256, kernel_size=3, stride=1, padding=1),
            Layer(256, 256, kernel_size=3, stride=1, padding=1),
            Layer(256, 256, kernel_size=3, stride=1, padding=1),
            Layer(256, 256, kernel_size=3, stride=1, padding=1),

            nn.MaxPool2d(kernel_size=2, stride=2),

            Layer(256, 512, kernel_size=3, stride=1, padding=1),
            Layer(512, 512, kernel_size=3, stride=1, padding=1),
            Layer(512, 512, kernel_size=3, stride=1, padding=1),
            Layer(512, 512, kernel_size=3, stride=1, padding=1),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    def forward(self,input):
        return self.features(input)    
    