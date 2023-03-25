from torch import nn
import torch
from torch.autograd import Variable


class LSTMModel(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTMModel, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm
        self.fc_1 =  nn.Linear(hidden_size, 128) #fully connected 1
        self.fc = nn.Linear(128, num_classes) #fully connected last layer

        self.relu = nn.ReLU()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def forward(self, x):
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device=self.device) #hidden state
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device=self.device) #internal state
        
        # Propagate input through LSTM
        output, (h, c) = self.lstm(x, (h0, c0)) #lstm with input, hidden, and internal state

        out = h[-1].view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.relu(out)
        out = self.fc_1(out) #first Dense
        out = self.relu(out) #relu
        out = self.fc(out) #Final Output
        return out


class LSTMDataset(torch.utils.data.Dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def __init__(self, X, y, window_size):
        self.X = torch.tensor(X, device=LSTMDataset.device).float()
        self.y = torch.tensor(y, device=LSTMDataset.device)
        self.window_size = window_size
    
    def __len__(self):
        return len(self.X) - self.window_size
    
    def __getitem__(self, idx):
        # self.X[idx:idx+self.window_size].unsqueeze(0).T,
        return self.X[idx:idx+self.window_size].T, self.y[idx+self.window_size]
