import torch
from torch.utils.data import DataLoader
from helpers.preprocessing import cross_entropy_weights, get_distribution, normalize_data
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import List
from copy import deepcopy
import numpy as np
import yaml
import joblib


class TrainingLoop:
    def __init__(self, ModelArchitecture:torch.nn.Module, Dataset:torch.utils.data.Dataset, hyperparams:dict,
                 imu:np.ndarray, ann:np.ndarray, device, model_name:str):
        print(f'Using device: {device}')
        
        self.model_name = model_name
        
        # Set instance methods
        self.plot_loss = self._plot_loss
        self.save_model = self._save_model
        
        self.hyperparams = deepcopy(hyperparams)
        self.Dataset = Dataset
        self.model = ModelArchitecture(self.hyperparams).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hyperparams['learning_rate'])
        self.criterion = None # Set in training_loop()
        
        self.train_generator = None
        self.val_generator = None
        
        self.train_loss_history = []
        self.val_loss_history = []
        self.val_acc = None
        self.train_acc = None
        
        # Split into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(imu, ann, test_size=self.hyperparams['val_size'],
                                                          shuffle=self.hyperparams['shuffle_split'],
                                                          random_state=42)
        
        # Loss (optionally weighted)
        weight = cross_entropy_weights(get_distribution(ann.tolist())['fracs']).to(device) if \
            self.hyperparams['weighted_loss'] else None
        self.criterion = torch.nn.CrossEntropyLoss(weight=weight) # one-hot encoding taken care of by pytorch
        
        # Normalization (optionally)
        if self.hyperparams['normalize']['run']:
            X_train, self.scaler = normalize_data(X_train, method=self.hyperparams['normalize']['method'])
            if self.scaler: # None if method is 'mean'
                X_val = self.scaler.transform(X_val)
                
        # Dataloaders
        self.train_generator = TrainingLoop.dataloader(self.Dataset, X_train, y_train, self.hyperparams, device)
        self.val_generator = TrainingLoop.dataloader(self.Dataset, X_val, y_val, self.hyperparams, device)
    
    @staticmethod
    def dataloader(Dataset:torch.utils.data.Dataset, X:np.ndarray, y:np.ndarray, hyperparams:dict, device) -> DataLoader:
        return DataLoader(Dataset(X, y, device, hyperparams), batch_size=hyperparams['batch_size'])

    def training_loop(self):    
        for epoch in range(1, self.hyperparams['epochs'] + 1):
            print(f'Epoch {epoch}')
            
            # Batch train
            batch_train_loss_history = []
            for (X, y) in tqdm(self.train_generator):
                self.optimizer.zero_grad()
                self.model.train()
                
                y_p = self.model(X)
                loss = self.criterion(y_p, y)

                loss.backward()
                self.optimizer.step()
                batch_train_loss_history.append(loss.item())
            
            # Batch validation
            batch_val_loss_history = []
            for (X, y) in tqdm(self.val_generator):
                self.model.eval()
                with torch.no_grad():
                    y_p = self.model(X)
                
                loss = self.criterion(y_p, y)
                batch_val_loss_history.append(loss.item())
            
            # Batch average loss
            epoch_train_loss = sum(batch_train_loss_history) / len(batch_train_loss_history)
            epoch_val_loss = sum(batch_val_loss_history) / len(batch_val_loss_history)
            print(f'Train loss: {epoch_train_loss:.4f}\nVal loss: {epoch_val_loss:.4f}')
            
            # Append batch loss to epoch loss list
            self.train_loss_history.append(epoch_train_loss)
            self.val_loss_history.append(epoch_val_loss)
        
        # Calculate accuracy
        self.train_acc = TrainingLoop.eval_acc(self.model, self.train_generator)
        self.val_acc = TrainingLoop.eval_acc(self.model, self.val_generator)
        
        self.save_model(f'{self.model_name}.torch')
        self.save_hyperparams(f'{self.model_name}.yaml')
        if self.scaler:
            joblib.dump(self.scaler, f'{self.model_name}_scaler.joblib')
        self.plot_loss()
  
        return self.model

    @staticmethod
    def plot_loss(train_loss_history:List[float], val_loss_history:List[float], hyperparams:dict, save_name:str) -> None:
        plt.figure()
        plt.title('Loss curve')
        plt.plot(range(hyperparams['epochs']), train_loss_history, label='train loss')
        plt.plot(range(hyperparams['epochs']), val_loss_history, label='val loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(save_name)
        plt.show()
        
    def _plot_loss(self) -> None:
        TrainingLoop.plot_loss(self.train_loss_history, self.val_loss_history, self.hyperparams, f'{self.model_name}.png')

    @staticmethod
    def save_model(model, path:str) -> None:
        torch.save(model.state_dict(), path)
        
    def _save_model(self, path:str) -> None:
        TrainingLoop.save_model(self.model, path)
        
    def save_hyperparams(self, path:str) -> None:
        with open(path, 'w') as f:
            yaml.dump(self.hyperparams, f)
    
    @staticmethod
    def model_output_to_classes(model_output:torch.Tensor) -> torch.Tensor:
        return torch.max(model_output, 1)[1] # Indices of max values

    @staticmethod
    def eval_acc(model:torch.nn.Module, dataloader:torch.utils.data.DataLoader) -> float:
        sum = 0
        length = 0
        for (X, y) in tqdm(dataloader):
            model.eval()
            with torch.no_grad():
                y_p = TrainingLoop.model_output_to_classes(model(X))
                sum += torch.sum(y == y_p).item()
                length += len(y_p)
        return sum/length
