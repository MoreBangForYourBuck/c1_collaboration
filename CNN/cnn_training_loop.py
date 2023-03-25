import torch
from torch.utils.data import DataLoader
from cnn_architecture import CNNDataset, CNNModel
from helpers.preprocessing import read_all_data,cross_entropy_weights
import matplotlib.pyplot as plt
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def training_loop(imu, ann, hyperparams:dict):
    model = CNNModel(num_classes=hyperparams['num_classes'], window_size=hyperparams['window_size'])
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
    #weights from inverse of fractional amount of each class
    criterion = torch.nn.CrossEntropyLoss(weight=(cross_entropy_weights([7.5,0.8,0.9,1.8])).to(device)) # one-hot encoding taken care of by pytorch

    X_train, X_val, y_train, y_val = train_test_split(imu, ann, test_size=0.2, random_state=42)
    train_generator = DataLoader(CNNDataset(X_train, y_train, hyperparams['window_size']), batch_size=hyperparams['batch_size'], shuffle=False)
    val_generator = DataLoader(CNNDataset(X_val, y_val, hyperparams['window_size']), batch_size=hyperparams['batch_size'],shuffle=False)

    train_loss_history = []
    val_loss_history = []
    for epoch in range(1, hyperparams['epochs'] + 1):
        print(f'Epoch {epoch}')
        
        batch_train_loss_history = []
        for (X, y) in tqdm(train_generator):
            optimizer.zero_grad()
            model.train()
            y_p = model(X)

            loss = criterion(y_p,y)
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
        
        epoch_train_loss = sum(batch_train_loss_history) / len(batch_train_loss_history)
        epoch_val_loss = sum(batch_val_loss_history) / len(batch_val_loss_history)
        
        print(f'Train loss: {epoch_train_loss:.4f}\nVal loss: {epoch_val_loss:.4f}')
        
        # Append average loss across batches
        train_loss_history.append(epoch_train_loss)
        val_loss_history.append(epoch_val_loss)
        
        
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
    
    with open('CNN/cnn_hyperparams.yaml', 'r') as f:
        hyperparams = yaml.safe_load(f)
    
    model = training_loop(imu, ann, hyperparams)
    save_model(model,"./cnn.model")
    

    # model = load_model('./cnn.model',CNNModel(num_classes=hyperparams['num_classes'], window_size=hyperparams['window_size']))
    
    # X_train, X_val, y_train, y_val = train_test_split(imu, ann, test_size=0.2, shuffle=False, random_state=42)
    # val_generator = DataLoader(CNNDataset(X_val, y_val, hyperparams['window_size']), batch_size=hyperparams['batch_size'],shuffle=False)
    # class_labels = evaluate(model,val_generator,plot=True)

    # result = np.asarray(labels)
    # np.savetxt("output_mlp.csv", result, delimiter=",")