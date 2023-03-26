from torch import nn
import torch


class CNNModel(nn.Module):
    def __init__(self, hyperparams:dict):
        super(CNNModel, self).__init__()
        # self.network = nn.Sequential(
        #     nn.Conv1d(6, 32, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool1d(1), # output: 64 x 125

        #     nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool1d(2), # output: 128 x 62

        #     nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool1d(2), # output: 256 x 31

        #     nn.Flatten(), 
        #     nn.Linear(256*31, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, num_classes),
        #     nn.Softmax(dim=1)
        # )
        self.conv1 = nn.Conv1d(6, 128, kernel_size=3, stride=1, padding=1)
        self.dense1 = nn.Linear(128*hyperparams['window_size'], 128)
        self.dense2 = nn.Linear(128, hyperparams['num_classes'])
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1) # flatten the output of the convolutional layer
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        return self.softmax(x)


class CNNDataset(torch.utils.data.Dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def __init__(self, X, y, hyperparams:dict):
        self.X = torch.tensor(X, device=CNNDataset.device).float()
        self.y = torch.tensor(y, device=CNNDataset.device)
        self.window_size = hyperparams['window_size']
    
    def __len__(self):
        return len(self.X) - self.window_size
    
    def __getitem__(self, idx):
        # self.X[idx:idx+self.window_size].unsqueeze(0).T,
        return self.X[idx:idx+self.window_size].T, self.y[idx+self.window_size]
