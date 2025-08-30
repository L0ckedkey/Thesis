import torch
import torch.nn as nn
import torch.nn.functional as F

class ECGCNN(nn.Module):
    def __init__(self, input_channels, num_classes, input_length):
        super(ECGCNN, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(2)
        self.bn1 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(2)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv5 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(2)
        self.bn3 = nn.BatchNorm1d(128)

        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(self._get_flattened_size(input_length, input_channels), 64)
        self.fc2 = nn.Linear(64, num_classes)

    def _get_flattened_size(self, input_length, input_channels):
        x = torch.zeros(1, input_channels, input_length)
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = self.pool3(F.relu(self.conv6(F.relu(self.conv5(x)))))
        return x.view(1, -1).size(1)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.bn1(self.dropout(x))

        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = self.bn2(self.dropout(x))

        x = self.pool3(F.relu(self.conv6(F.relu(self.conv5(x)))))
        x = self.bn3(self.dropout(x))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
