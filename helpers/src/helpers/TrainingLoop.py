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
import torchmetrics as tm


class TrainingLoop:
    def __init__(self, ModelArchitecture:torch.nn.Module, Dataset:torch.utils.data.Dataset, hyperparams:dict,
                 imu:np.ndarray, ann:np.ndarray, device, model_name:str):
        print(f'Using device: {device}')
        
        self.device = device
        
        self.model_name = model_name
        
        # Set instance methods
        self.plot_loss = self._plot_loss
        self.save_model = self._save_model
        
        self.hyperparams = deepcopy(hyperparams)
        self.Dataset = Dataset
        self.model = ModelArchitecture(self.hyperparams).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hyperparams['learning_rate'])
        
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
        print(get_distribution(ann.tolist(), num_classes=hyperparams['num_classes']))
        weight = cross_entropy_weights(get_distribution(
            ann.tolist(), num_classes=hyperparams['num_classes'])['fracs']).to(device) \
                if self.hyperparams['weighted_loss'] else None
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
                # if len(batch_train_loss_history) % 1000 == 0:
                #     print(f'Batch {len(batch_train_loss_history)} loss: {loss.item():.4f}')
            
            # Batch validation
            batch_val_loss_history = []
            val_precision = []
            val_recall = []
            val_f1 = []
            val_acc = []
            for (X, y) in tqdm(self.val_generator):
                self.model.eval()
                with torch.no_grad():
                    y_p = self.model(X)
                
                loss = self.criterion(y_p, y)
                batch_val_loss_history.append(loss.item())
                
                stats = TrainingLoop.stats(y_p, y, self.hyperparams['num_classes'], device=self.device)
                val_precision.append(stats['precision'])
                val_recall.append(stats['recall'])
                val_f1.append(stats['f1'])
                val_acc.append(stats['accuracy'])
            
            # Batch average loss
            epoch_train_loss = sum(batch_train_loss_history) / len(batch_train_loss_history)
            epoch_val_loss = sum(batch_val_loss_history) / len(batch_val_loss_history)
            print(f'Train loss: {epoch_train_loss:.4f}\nVal loss: {epoch_val_loss:.4f}')
            
            self.save_model(f'{self.model_name}_epoch{epoch}.torch')
            if epoch == 1:
                self.save_hyperparams(f'{self.model_name}.yaml')
                if self.hyperparams['normalize']['run']:
                    joblib.dump(self.scaler, f'{self.model_name}_scaler.joblib')
                
            # Append batch loss to epoch loss list
            self.train_loss_history.append(epoch_train_loss)
            self.val_loss_history.append(epoch_val_loss)
            
            joblib.dump(self.train_loss_history, f'{self.model_name}_train_loss_hist.joblib')
            joblib.dump(self.val_loss_history, f'{self.model_name}_val_loss_hist.joblib')
            
            print(f'Val precision: {sum(val_precision) / len(val_precision):.4f}')
            print(f'Val recall: {sum(val_recall) / len(val_recall):.4f}')
            print(f'Val f1: {sum(val_f1) / len(val_f1):.4f}')
            print(f'Val acc: {sum(val_acc) / len(val_acc):.4f}')
            
        self.plot_loss()
        
        # Calculate accuracy
        self.train_acc = TrainingLoop.eval_acc(self.model, self.train_generator)
        self.val_acc = TrainingLoop.eval_acc(self.model, self.val_generator)
          
        return self.model
    
    @staticmethod
    def stats(y_p:torch.Tensor, y:torch.Tensor, num_classes:int, device) -> dict:
        precision = tm.Precision(task='multiclass', average='macro', num_classes=num_classes).to(device)
        recall = tm.Recall(task='multiclass', average='macro', num_classes=num_classes).to(device)
        f_one = tm.F1Score(task='multiclass', average='macro', num_classes=num_classes).to(device)
        acc = torch.sum(y == TrainingLoop.model_output_to_classes(y_p)).item() / len(y)
        
        
        return {
            'precision': precision(y_p, y).item(),
            'recall': recall(y_p, y).item(),
            'f1': f_one(y_p, y).item(),
            'accuracy': acc
        }
        

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
        sumy = 0
        length = 0
        for (X, y) in tqdm(dataloader):
            model.eval()
            with torch.no_grad():
                y_p = TrainingLoop.model_output_to_classes(model(X))
                sumy += torch.sum(y == y_p).item()
                length += len(y_p)
        print('yp', torch.sum(y_p))
        print(torch.sum(y))
        return sumy/length