import torch
import torch.nn as nn
import torch.nn.functional as F

class CracksRecognitionModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'CracksRecognitionModel'
        # first layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), padding='same')
        self.relu = nn.ReLU(inplace=False)
        self.batchnorm1 = nn.BatchNorm2d(num_features=32, )
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        # second layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding='same')
        self.dropout1 = nn.Dropout(0.3)
        self.pool2 = self.pool1
        # third layer
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding='same')
        self.dropout2 = self.dropout1
        self.pool3 = self.pool1
        # forth layer
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding='same')
        self.dropout4 = self.dropout1
        self.pool4 = self.pool1

        self.linear1 = nn.Linear(in_features=50176, out_features=1024)
        self.dropout = nn.Dropout(0.5)

        self.linear2 = nn.Linear(in_features=1024, out_features=2)
    
    def forward(self, x):
        # 1st layer
        x = self.pool1(self.batchnorm1(self.relu(self.conv1(x))))
        # 2nd layer
        x = self.pool2(self.dropout1(self.relu(self.conv2(x))))
        # 3rd layer
        x = self.pool3(self.dropout2(self.relu(self.conv3(x))))
        # 4th layer
        x = self.pool4(self.dropout4(self.relu(self.conv4(x))))
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        x = F.log_softmax(x, dim=1)
        return x 