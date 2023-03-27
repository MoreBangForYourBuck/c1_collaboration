from torch import nn
import torch
from torch.autograd import Variable


class LSTMModel(nn.Module):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def __init__(self, hyperparams):
        super(LSTMModel, self).__init__()
        num_classes = hyperparams['num_classes'] #number of classes
        input_size = hyperparams['lstm']['window_size'] #input size
        self.num_layers = hyperparams['lstm']['num_layers'] #number of layers
        self.hidden_size = hyperparams['lstm']['hidden_size'] #hidden state

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                            batch_first=True) #lstm
        self.fc1 =  nn.Linear(self.hidden_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(LSTMModel.device) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(LSTMModel.device) #internal state
        # Propagate input through LSTM
        output, (h, c) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state

        out = h[-1].view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.softmax(out)
        return out


class LSTMDataset(torch.utils.data.Dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def __init__(self, X, y, hyperparams):
        self.X = torch.tensor(X, device=LSTMDataset.device).float()
        self.y = torch.tensor(y, device=LSTMDataset.device)
        self.window_size = hyperparams['lstm']['window_size']
    
    def __len__(self):
        return len(self.X) - self.window_size
    
    def __getitem__(self, idx):
        # self.X[idx:idx+self.window_size].unsqueeze(0).T,
        return self.X[idx:idx+self.window_size].T, self.y[idx+self.window_size]
