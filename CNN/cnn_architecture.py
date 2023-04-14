from torch import nn
import torch
from torch import functional as F


class CNN(nn.Module):
    def __init__(self, hyperparams:dict):
        super(CNN, self).__init__()
        self.relu = nn.ReLU() # Activation
        
        self.conv1 = nn.Conv1d(in_channels=6, out_channels=16, kernel_size=hyperparams['cnn']['kernel_size'], stride=1,
                               padding='same')
        self.avgpool1 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.dropout1 = nn.Dropout(0.2)
        
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=hyperparams['cnn']['kernel_size'], stride=1,
                               padding='same')
        self.avgpool2 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.2)
        
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=hyperparams['cnn']['kernel_size'], stride=1,
                               padding='same')
        self.upsample1 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(0.2)
        
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=hyperparams['cnn']['kernel_size'], stride=1,
                               padding='same')
        self.upsample2 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.bn4 = nn.BatchNorm1d(16)
        self.dropout4 = nn.Dropout(0.2)
        
        self.conv5 = nn.Conv1d(in_channels=16, out_channels=hyperparams['num_classes'],
                               kernel_size=hyperparams['cnn']['kernel_size'], stride=1, padding='same')
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(hyperparams['num_classes']*hyperparams['cnn']['window_size'], hyperparams['num_classes'])
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.avgpool1(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.avgpool2(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        
        x = self.conv3(x)
        x = self.relu(x)
        x = self.upsample1(x)
        x = self.bn3(x)
        x = self.dropout3(x)
        
        x = self.conv4(x)
        x = self.relu(x)
        x = self.upsample2(x)
        x = self.bn4(x)
        x = self.dropout4(x)
        
        x = self.conv5(x)
        x = self.relu(x)
        
        x = self.flatten(x)
        x = self.fc(x)
        
        x = self.softmax(x)
        return x


class CNNDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, device, hyperparams:dict):
        self.device = device
        self.window_size = hyperparams['cnn']['window_size']
        self.len = len(X) - self.window_size
        self.X = X # Transpose for conv imput to yield shape (batch_size, in_channels, input_size)
        self.y = y
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx:idx+self.window_size], device=self.device).float().T, #.float maybe?
            torch.tensor(self.y[idx+self.window_size], device=self.device)
        )

if __name__ == '__main__':
    hyperparams = {
        'cnn': {
            'kernel_size': 12,
        }
    }
    model = CNN(hyperparams)
    print(model)
