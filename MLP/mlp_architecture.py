from torch import nn
import torch


class MLPModel(nn.Module):    
    def __init__(self, hyperparams:dict):
        super(MLPModel, self).__init__()
        self.hidden_layer_sizes = hyperparams['mlp']['hidden_layers']
        
        self.relu = nn.ReLU() # Activation function
        self.input = nn.Linear(6, self.hidden_layer_sizes[0])

        for i in range(len(self.hidden_layer_sizes)):
            if i < len(self.hidden_layer_sizes) - 1:
                setattr(self, f'hidden_layer_{i}', nn.Linear(self.hidden_layer_sizes[i], self.hidden_layer_sizes[i+1]))
            else:
                setattr(self, f'hidden_layer_{i}', nn.Linear(self.hidden_layer_sizes[i], hyperparams['num_classes']))

        self.output = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.input(x)
        
        for i in range(len(self.hidden_layer_sizes)):
            x = self.relu(x)
            x = getattr(self, f'hidden_layer_{i}')(x)

        return self.output(x)


class MLPDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, device, *args, **kwargs): \
        # *args and **kwargs allow hyperparams to be passed for other Dataset classes initialized by TrainingLoop
        self.X = torch.tensor(X, device=device).float()
        self.y = torch.tensor(y, device=device)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
