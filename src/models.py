# models.py
import torch
import torch.nn as nn

class Model1(nn.Module):
    """Model1: BN=True, MaxPool, Drop=0.25"""
    def __init__(self, num_classes=12):
        super().__init__()
        self.conv = nn.Sequential(
            # Block 1: BN=True, MaxPool, Drop=0.25
            nn.Conv2d(1, 32, 3, padding=1), 
            nn.BatchNorm2d(32), 
            nn.ReLU(), 
            nn.MaxPool2d(2), 
            nn.Dropout2d(0.25),
            
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(), 
            nn.MaxPool2d(2), 
            nn.Dropout2d(0.25),
            
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1), 
            nn.BatchNorm2d(128), 
            nn.ReLU(), 
            nn.MaxPool2d(2), 
            nn.Dropout2d(0.25),
            
            # Block 4
            nn.Conv2d(128, 256, 3, padding=1), 
            nn.ReLU()
        )
        
        # Auto-calcolo dimensioni FC
        dummy = torch.randn(1, 1, 128, 313)
        size = self.conv(dummy).view(-1).size(0)
        
        self.fc = nn.Sequential(
            nn.Linear(size, 128), 
            nn.ReLU(), 
            nn.Dropout(0.3), 
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.fc(self.conv(x).view(x.size(0), -1))


class Model1b(nn.Module):
    """Model1b: BN=True, Stride, Drop=0.20"""
    def __init__(self, num_classes=12):
        super().__init__()
        self.conv = nn.Sequential(
            # Block 1: BN=True, Stride, Drop=0.20
            nn.Conv2d(1, 32, 3, padding=1), 
            nn.BatchNorm2d(32), 
            nn.ReLU(), 
            nn.Dropout2d(0.20),
            
            # Block 2 (stride=2 per downsampling)
            nn.Conv2d(32, 64, 3, stride=2, padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(), 
            nn.Dropout2d(0.20),
            
            # Block 3 (stride=2 per downsampling)
            nn.Conv2d(64, 128, 3, stride=2, padding=1), 
            nn.BatchNorm2d(128), 
            nn.ReLU(), 
            nn.Dropout2d(0.20),
            
            # Block 4 (stride=2 per downsampling)
            nn.Conv2d(128, 256, 3, stride=2, padding=1), 
            nn.ReLU()
        )
        
        # Auto-calcolo dimensioni FC
        dummy = torch.randn(1, 1, 128, 313)
        size = self.conv(dummy).view(-1).size(0)
        
        self.fc = nn.Sequential(
            nn.Linear(size, 128), 
            nn.ReLU(), 
            nn.Dropout(0.3), 
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.fc(self.conv(x).view(x.size(0), -1))


class Model1c(nn.Module):
    """Model1c: BN=False, Stride, Drop=0.15"""
    def __init__(self, num_classes=12):
        super().__init__()
        self.conv = nn.Sequential(
            # Block 1: BN=False, Stride, Drop=0.15
            nn.Conv2d(1, 32, 3, padding=1), 
            nn.ReLU(), 
            nn.Dropout2d(0.15),
            
            # Block 2 (stride=2, no BatchNorm)
            nn.Conv2d(32, 64, 3, stride=2, padding=1), 
            nn.ReLU(), 
            nn.Dropout2d(0.15),
            
            # Block 3 (stride=2, no BatchNorm)
            nn.Conv2d(64, 128, 3, stride=2, padding=1), 
            nn.ReLU(), 
            nn.Dropout2d(0.15),
            
            # Block 4 (stride=2, no BatchNorm)
            nn.Conv2d(128, 256, 3, stride=2, padding=1), 
            nn.ReLU()
        )
        
        # Auto-calcolo dimensioni FC
        dummy = torch.randn(1, 1, 128, 313)
        size = self.conv(dummy).view(-1).size(0)
        
        self.fc = nn.Sequential(
            nn.Linear(size, 128), 
            nn.ReLU(), 
            nn.Dropout(0.3), 
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.fc(self.conv(x).view(x.size(0), -1))


class Model2(nn.Module):
    """Model2: BN=False, MaxPool, Drop=0.10"""
    def __init__(self, num_classes=12):
        super().__init__()
        self.conv = nn.Sequential(
            # Block 1: BN=False, MaxPool, Drop=0.10
            nn.Conv2d(1, 32, 3, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(2), 
            nn.Dropout2d(0.10),
            
            # Block 2 (MaxPool, no BatchNorm)
            nn.Conv2d(32, 64, 3, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(2), 
            nn.Dropout2d(0.10),
            
            # Block 3 (MaxPool, no BatchNorm)
            nn.Conv2d(64, 128, 3, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(2), 
            nn.Dropout2d(0.10),
            
            # Block 4 (no pooling)
            nn.Conv2d(128, 256, 3, padding=1), 
            nn.ReLU()
        )
        
        # Auto-calcolo dimensioni FC
        dummy = torch.randn(1, 1, 128, 313)
        size = self.conv(dummy).view(-1).size(0)
        
        self.fc = nn.Sequential(
            nn.Linear(size, 128), 
            nn.ReLU(), 
            nn.Dropout(0.3), 
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.fc(self.conv(x).view(x.size(0), -1))


def get_model(name="model1", **kwargs):

    if name == "model1":
        return Model1(**kwargs)
    elif name == "model1b":
        return Model1b(**kwargs)
    elif name == "model1c":
        return Model1c(**kwargs)
    elif name == "model2":
        return Model2(**kwargs)
    else:
        raise ValueError(f"Modello '{name}' non riconosciuto. "
                        f"Modelli disponibili: model1, model1b, model1c, model2")

