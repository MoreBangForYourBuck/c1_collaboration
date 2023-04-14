from helpers.preprocessing import read_all_data, read_all_data_no_zeros
import yaml
from mlp_architecture import MLPModel, MLPDataset
from helpers import TrainingLoop
import torch


if __name__ == '__main__':
    # data_dict = read_all_data()
    # imu = data_dict['imu'].to_numpy()
    # ann = data_dict['ann'].to_numpy().flatten()
    # del data_dict # Remove to free memory
    
    data_dict = read_all_data_no_zeros()
    imu = data_dict['imu']
    ann = data_dict['ann'].flatten() - 1
    del data_dict
    
    with open('MLP/mlp_hyperparams.yaml', 'r') as f:
        hyperparams = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    training_loop = TrainingLoop(MLPModel, MLPDataset, hyperparams, imu, ann, device, '/home/jacob/ece542_repos/c1_collaboration/c2_final_models/tuned3')
    training_loop.training_loop()
