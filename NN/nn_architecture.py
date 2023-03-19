from torch import nn
import torch


class NNModel(nn.Module):
    def __init__(self, num_classes):
        super(NNModel, self).__init__()
        self.fc1 = nn.Linear(6, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.output = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return self.output(x)


class NNDataset(torch.utils.data.Dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def __init__(self, X, y):
        self.X = torch.tensor(X, device=NNDataset.device).float()
        self.y = torch.tensor(y, device=NNDataset.device)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
