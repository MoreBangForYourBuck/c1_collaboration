import torch
from torch.utils.data import DataLoader
from nn_architecture import NNDataset, NNModel
from helpers.preprocessing import read_all_data
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def training_loop(imu, ann, hyperparams:dict):
    model = NNModel(num_classes=hyperparams['num_classes'])
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss() # one-hot encoding taken care of by pytorch

    # Time series, but still shuffling because no window component
    train_generator = DataLoader(NNDataset(imu, ann), batch_size=hyperparams['batch_size'], shuffle=True)

    train_loss_history = []
    for epoch in range(1, hyperparams['epochs'] + 1):
        print(f'Epoch {epoch}')
        
        batch_train_loss_history = []
        for (X, y) in tqdm(train_generator):
            optimizer.zero_grad()
            model.train()
            
            y_p = model(X)
            loss = criterion(y_p, y)

            loss.backward()
            optimizer.step()
            # batch_train_loss_history.append(loss.item())
            train_loss_history.append(loss.item())
        
        # Append average loss across batches
        # train_loss_history.append(sum(batch_train_loss_history) / len(batch_train_loss_history))
        
    plt.figure()
    plt.title('MSE Loss')
    # plt.plot(range(hyperparams['epochs']), train_loss_history, label='train loss')
    plt.plot(range(len(train_loss_history)), train_loss_history, label='train loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    

if __name__ == '__main__':
    data_dict = read_all_data()
    imu = data_dict['imu'].to_numpy()
    ann = data_dict['ann'].to_numpy().flatten()
    del data_dict # Remove to free memory
    
    with open('NN/nn_hyperparams.yaml', 'r') as f:
        hyperparams = yaml.safe_load(f)
    
    training_loop(imu, ann, hyperparams)
