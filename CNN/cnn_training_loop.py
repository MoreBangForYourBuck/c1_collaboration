from helpers import TrainingLoop
import yaml
from helpers.preprocessing import read_all_data
from cnn_architecture import CNNDataset, CNN
import torch
from helpers.eval import load_model
import time


if __name__ == '__main__':
    NAME = 'cnn_4_9_{0}'.format(str(time.time()).split('.')[0])
    print(NAME)
    
    data_dict = read_all_data()
    imu = data_dict['imu'].to_numpy()
    ann = data_dict['ann'].to_numpy().flatten()
    del data_dict # Remove to free memory
    
    with open('CNN/cnn_hyperparams.yaml', 'r') as f:
        hyperparams = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    training_loop = TrainingLoop(CNN, CNNDataset, hyperparams, imu, ann, device,
                                 f'/home/jacob/ece542_repos/c1_collaboration/{NAME}')
    training_loop.training_loop()
    # model = load_model(f'{NAME}.torch', GPTCNN(hyperparams).to(torch.device('cuda')))
    # print(training_loop.eval_loop(model))
    
    
    # model = load_model('./cnn.model',CNNModel(num_classes=hyperparams['num_classes'], window_size=hyperparams['window_size']))
    
    # X_train, X_val, y_train, y_val = train_test_split(imu, ann, test_size=0.2, shuffle=False, random_state=42)
    # val_generator = DataLoader(CNNDataset(X_val, y_val, hyperparams['window_size']), batch_size=hyperparams['batch_size'],shuffle=False)
    # class_labels = evaluate(model,val_generator,plot=True)

    # result = np.asarray(labels)
    # np.savetxt("output_mlp.csv", result, delimiter=",")