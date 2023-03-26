from helpers.preprocessing import read_all_data
import yaml
from lstm_architecture import LSTMModel, LSTMDataset
from helpers import TrainingLoop
import joblib


if __name__ == '__main__':
    data_dict = read_all_data()
    imu = data_dict['imu'].to_numpy()
    ann = data_dict['ann'].to_numpy().flatten()
    del data_dict # Remove to free memory
    
    with open('LSTM/lstm_hyperparams.yaml', 'r') as f:
        hyperparams = yaml.safe_load(f)
        
    training_loop = TrainingLoop(LSTMModel, LSTMDataset, hyperparams)
    training_loop.training_loop(imu, ann)
    
    joblib.dump(training_loop, 'lstm_training_loop.joblib')
