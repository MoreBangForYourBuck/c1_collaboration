import torch
from torch.utils.data import DataLoader
from mlp_architecture import MLPDataset, MLPModel
from helpers.preprocessing import read_all_data
from helpers import eval
import matplotlib.pyplot as plt
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import List

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def training_loop(imu, ann, hyperparams:dict):
    model = MLPModel(num_classes=hyperparams['num_classes'])
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss() # one-hot encoding taken care of by pytorch

    # Time series, but still shuffling because no window component
    X_train, X_val, y_train, y_val = train_test_split(imu, ann, test_size=0.2, shuffle=True, random_state=42)
    train_generator = DataLoader(MLPDataset(X_train, y_train), batch_size=hyperparams['batch_size'])
    val_generator = DataLoader(MLPDataset(X_val, y_val), batch_size=hyperparams['batch_size'])

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

def labels_to_classes(labels) -> List[List[float]]:
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
    return new_model.to(device)

if __name__ == '__main__':
    data_dict = read_all_data()
    imu = data_dict['imu'].to_numpy()
    ann = data_dict['ann'].to_numpy().flatten()
    del data_dict # Remove to free memory
    
    with open('MLP/mlp_hyperparams.yaml', 'r') as f:
        hyperparams = yaml.safe_load(f)
    
    # model = training_loop(imu, ann, hyperparams)
    # save_model(model,"./mlp.model")
    print(torch.cuda.is_available())
    model = load_model('./mlp.model',MLPModel(num_classes=hyperparams['num_classes']))
    
    X_train, X_val, y_train, y_val = train_test_split(imu, ann, test_size=0.2, shuffle=False, random_state=42)
    train_generator = DataLoader(MLPDataset(X_train, y_train), batch_size=hyperparams['batch_size'])
    val_generator = DataLoader(MLPDataset(X_val, y_val), batch_size=hyperparams['batch_size'])
    # class_labels = evaluate(model,val_generator,plot=True)
    
    
    train_acc = eval.eval_acc(model, train_generator)
    val_acc = eval.eval_acc(model, val_generator)
    
    print(f'Train accuracy: {train_acc}%\nVal accuracy: {val_acc}%')
    

    # result = np.asarray(labels)
    # np.savetxt("output_mlp.csv", result, delimiter=",")
