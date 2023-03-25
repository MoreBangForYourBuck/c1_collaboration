import torch
from torch.utils.data import DataLoader
from lstm_architecture import LSTMDataset, LSTMModel
from helpers.preprocessing import read_all_data
import matplotlib.pyplot as plt
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def training_loop(imu, ann, hyperparams:dict):

    model = LSTMModel(num_classes=hyperparams['num_classes'], input_size=hyperparams['input_size'], 
                    hidden_size=hyperparams['hidden_size'], num_layers=hyperparams['num_layers'],
                    seq_length=hyperparams['batch_size'])
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.0437158469945356,0.409836065573771,0.364298724888227,0.182149362477231])) # one-hot encoding taken care of by pytorch

    X_train, X_val, y_train, y_val = train_test_split(imu, ann, test_size=0.2, random_state=42)
    train_generator = DataLoader(LSTMDataset(X_train, y_train, hyperparams['input_size']), batch_size=hyperparams['batch_size'], shuffle=False)
    val_generator = DataLoader(LSTMDataset(X_val, y_val, hyperparams['input_size']), batch_size=hyperparams['batch_size'],shuffle=False)

    train_loss_history = []
    val_loss_history = []
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
            batch_train_loss_history.append(loss.item())
        
        batch_val_loss_history = []
        for (X, y) in tqdm(val_generator):
            model.eval()
            with torch.no_grad():
                y_p = model(X)
            
            loss = criterion(y_p, y)
            batch_val_loss_history.append(loss.item())
        
        # Append average loss across batches
        train_loss_history.append(sum(batch_train_loss_history) / len(batch_train_loss_history))
        val_loss_history.append(sum(batch_val_loss_history) / len(batch_val_loss_history))
        
    plt.figure()
    plt.title('Loss curve')
    plt.plot(range(hyperparams['epochs']), train_loss_history, label='train loss')
    plt.plot(range(hyperparams['epochs']), val_loss_history, label='val loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    return model
    
def labels_to_classes(labels):
    class_labels =[]
    for c in range(int(list(labels[0].shape)[0])):
        class_labels.append([float(labels[i][c]) for i in range(len(labels))])
    return class_labels

def evaluate(model,dataloader,plot=True):
    model_output = []
    for (X, y) in tqdm(dataloader):
        model.eval()
        with torch.no_grad():
            y_p = model(X)
        model_output.extend(y_p)
    classes = labels_to_classes(model_output)

    if plot:
        plt.figure()
        plt.title('Evaluation - Labels')
        for i,c in enumerate(classes):
            plt.plot(range(len(c)), c, '.',label=str(i) +' Labels')
        plt.xlabel('Time')
        plt.ylabel('Label')
        plt.legend()
        plt.show()

    return classes

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(path_to_saved_model, new_model):
    new_model.load_state_dict(torch.load(path_to_saved_model))
    new_model.eval()
    return new_model

if __name__ == '__main__':
    data_dict = read_all_data()
    imu = data_dict['imu'].to_numpy()
    ann = data_dict['ann'].to_numpy().flatten()
    del data_dict # Remove to free memory
    
    with open('LSTM/lstm_hyperparams.yaml', 'r') as f:
        hyperparams = yaml.safe_load(f)
    
    model = training_loop(imu, ann, hyperparams)
    save_model(model,"./lstm.model")
    

    # model = load_model('./lstm.model', LSTMModel(num_classes=hyperparams['num_classes'], input_size=hyperparams['input_size'], 
    #                 hidden_size=hyperparams['hidden_size'], num_layers=hyperparams['num_layers'],
    #                 seq_length=hyperparams['batch_size']))
    
    X_train, X_val, y_train, y_val = train_test_split(imu, ann, test_size=0.2, random_state=42)
    val_generator = DataLoader(LSTMDataset(X_val, y_val, hyperparams['input_size']), batch_size=hyperparams['batch_size'],shuffle=False)
    class_labels = evaluate(model,val_generator,plot=True)

    # result = np.asarray(labels)
    # np.savetxt("output_mlp.csv", result, delimiter=",")