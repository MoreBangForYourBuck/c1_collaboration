import torch
from torch.utils.data import DataLoader
from nn_architecture import NNDataset, NNModel
from helpers.preprocessing import expand_ann
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def training_loop(imu, ann, hyperparams:dict):
    model = NNModel(num_classes=hyperparams['num_classes'])
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss() # one-hot encoding taken care of by pytorch

    train_generator = DataLoader(NNDataset(imu, ann), batch_size=hyperparams['batch_size'], shuffle=True) # DONT SHUFFLE (TIME SERIES)


    train_loss_history = []
    for epoch in range(1, hyperparams['epochs'] + 1):
        
        batch_train_loss_history = []
        for i, (X, y) in enumerate(train_generator):
            print(i)
            optimizer.zero_grad()
            model.train()
            
            y_p = model(X)
            loss = criterion(y_p, y)
            # print(y, y_p)

            loss.backward()
            optimizer.step()
            batch_train_loss_history.append(loss.item())
        
        # Append average loss across batches
        train_loss_history.append(sum(batch_train_loss_history) / len(batch_train_loss_history))
        
    plt.figure()
    plt.title('MSE Loss')
    plt.plot(range(hyperparams['epochs']), train_loss_history, label='train loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    

if __name__ == '__main__':
    imu = pd.read_csv('training_data/subject_001_01__x.csv',
                  names=['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'])
    imu_t = pd.read_csv('training_data/subject_001_01__x_time.csv', names=['time'])
    ann = pd.read_csv('training_data/subject_001_01__y.csv', names=['labels'])
    ann_t = pd.read_csv('training_data/subject_001_01__y_time.csv', names=['time'])
    
    expanded_ann = expand_ann(
        imu_t['time'].tolist(),
        ann['labels'].tolist(),
        ann_t['time'].tolist()
    )
    
    ann = expanded_ann['ann']
    ann_t = expanded_ann['ann_time']
    
    with open('NN/nn_hyperparams.yaml', 'r') as f:
        hyperparams = yaml.safe_load(f)
        
    training_loop(imu.to_numpy(), np.array(ann), hyperparams)
    