import torch
import torch.nn as nn

class SimpleSpectrogramCNN(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(15360, 128), 
            nn.ReLU(),
            nn.Linear(128, n_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)  
        return x
