import torch
import torch.nn as nn
import torch.nn.functional as F

class IntervalCNN(nn.Module):
    """CNN model for interval recognition."""
    
    def __init__(self, n_mels: int = 128, n_intervals: int = 12):
        """
        Args:
            n_mels: Number of mel frequency bands (input height)
            n_intervals: Number of interval classes to predict
        """
        super().__init__()
        
        # First Convolution Block
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # Second Convolution Block
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # Third Convolution Block
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # Calculate the features shape after convolutions
        self.n_mels = n_mels
        
        # Adaptive pooling to get fixed size output regardless of input size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Calculate final flattened size: 64 channels * 4 * 4
        self.flat_features = 64 * 4 * 4
        
        # Fully Connected Layers
        self.fc = nn.Sequential(
            nn.Linear(self.flat_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_intervals)
        )
    
    def forward(self, x):
        # Add channel dimension if not present
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # Convolution blocks
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Adaptive pooling to get fixed size
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc(x)
        
        return F.log_softmax(x, dim=1)