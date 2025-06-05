import torch
import torch.nn as nn
from torchvision import models

class CracksRecognitionModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'CracksRecognitionModel'
        self.model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
    
    def forward(self, x):
        return self.model(x) 