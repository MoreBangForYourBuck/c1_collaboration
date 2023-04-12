from helpers import TrainingLoop
import yaml
from helpers.preprocessing import read_all_data
from cnn_architecture import CNNModel, CNNDataset, SimpleCNN
import torch
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    data_dict = read_all_data()
    imu = data_dict['imu'].to_numpy()
    ann = data_dict['ann'].to_numpy().flatten()
    del data_dict # Remove to free memory
    
    with open('CNN/cnn_hyperparams.yaml', 'r') as f:
        hyperparams = yaml.safe_load(f)
        
    # plt.plot(range(len(ann[:10000])), ann[:10000])
    # plt.show()
    
    print(imu[np.where(ann == 1)])
    for i in range(4):
        x = imu[np.where(ann == i)] # 2, 5
        plt.figure()
        plt.title(f'class {i}')
        plt.scatter(range(len(x)), x)
    plt.show()