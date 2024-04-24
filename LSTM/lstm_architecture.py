from torch import nn
import scipy
import torch


class LSTMModel(nn.Module):
    def __init__(self, hyperparams:dict):
        super(LSTMModel, self).__init__()
        self.hidden_size = hyperparams['lstm']['hidden_size'] #hidden state

        self.lstm = nn.LSTM(input_size=6, hidden_size=self.hidden_size, num_layers=hyperparams['lstm']['num_layers'],
                            batch_first=True)  # shape batch, seq, feature
        self.fc1 =  nn.Linear(self.hidden_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, hyperparams['num_classes'])
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Propagate input through LSTM
        # print(x.shape)
        x, (h, c) = self.lstm(x) #lstm with input, hidden, and internal state
        # print(h.shape)
        x = x[:, -1, :] # Use last time step's output
        out = self.relu(h.view(-1, self.hidden_size))
        out = self.fc1(h)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.softmax(out)
        return out
    
    
class LSTM_111(nn.Module):
    def __init__(self, hyperparams:dict):
        self.hidden_size = hyperparams['lstm']['hidden_size']
        self.lstm_1 = nn.LSTM(input_size=6, hidden_size=128, num_layers=hyperparams['lstm']['num_layers'],
                              batch_first=True)
        self.dropout_1 = nn.Dropout(0.2)
        self.lstm_1 = nn.LSTM(input_size=6, hidden_size=128, num_layers=hyperparams['lstm']['num_layers'],
                              batch_first=True)
        



class LSTMDataset(torch.utils.data.Dataset):
    
    def __init__(self, X, y, device, hyperparams:dict):
        self.device = device
        self.window_size = hyperparams['lstm']['window_size']
        self.X = torch.tensor(X, device=device).float()
        self.y = torch.tensor(y, device=device)
    
    def __len__(self):
        return len(self.X) - self.window_size
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx:idx+self.window_size], device=self.device),
            torch.tensor(scipy.stats.mode(self.y[idx+self.window_size]), device=self.device)
        )
        
        
# class LSTMDataset(torch.utils.data.Dataset):
    
#     def __init__(self, X, y, device, hyperparams:dict):
#         self.device = device
#         self.window_size = hyperparams['lstm']['window_size']
#         self.len = len(X) - self.window_size
#         self.X = X # Transpose for conv imput to yield shape (batch_size, in_channels, input_size)
#         self.y = y
    
#     def __len__(self):
#         return self.len
    
#     def __getitem__(self, idx):
#         return (
#             torch.tensor(self.X[idx:idx+self.window_size], device=self.device).float().T, #.float maybe?
#             torch.tensor(self.y[idx+self.window_size], device=self.device)
#         )
