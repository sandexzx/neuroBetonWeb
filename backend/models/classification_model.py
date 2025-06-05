import torch
import torch.nn as nn
from torchvision import models

class ClassificationModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet18(weights=None)  # Отключаем предобученные веса
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x) 