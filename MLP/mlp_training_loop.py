from helpers.preprocessing import read_all_data
import yaml
from mlp_architecture import MLPModel, MLPDataset
from helpers import TrainingLoop
import joblib


if __name__ == '__main__':
    data_dict = read_all_data()
    imu = data_dict['imu'].to_numpy()
    ann = data_dict['ann'].to_numpy().flatten()
    del data_dict # Remove to free memory
    
    with open('MLP/mlp_hyperparams.yaml', 'r') as f:
        hyperparams = yaml.safe_load(f)
        
    training_loop = TrainingLoop(MLPModel, MLPDataset, hyperparams)
    training_loop.training_loop(imu, ann)
    
    print('Train accuracy: {0}%\nVal accuracy: {1}%'.format(training_loop.train_acc, training_loop.val_acc))
    training_loop.plot_loss()
    
    joblib.dump(training_loop, 'mlp_training_loop_yes_weight_yes_norm_2.joblib')
    
    # model = training_loop(imu, ann, hyperparams)
    # save_model(model,"./mlp.model")
    # print(torch.cuda.is_available())
    # model = load_model('./mlp.model',MLPModel(num_classes=hyperparams['num_classes']))
    
    # X_train, X_val, y_train, y_val = train_test_split(imu, ann, test_size=0.2, shuffle=False, random_state=42)
    
    # # Scale on training set, then apply to validation set
    # X_train, scaler = normalize_data(X_train, method='standard')
    # X_val = scaler.fit(X_val)
    
    # train_generator = DataLoader(MLPDataset(X_train, y_train), batch_size=hyperparams['batch_size'])
    # val_generator = DataLoader(MLPDataset(X_val, y_val), batch_size=hyperparams['batch_size'])
    # # class_labels = evaluate(model,val_generator,plot=True)
    
    
    # train_acc = eval.eval_acc(model, train_generator)
    # val_acc = eval.eval_acc(model, val_generator)
    
    # print(f'Train accuracy: {train_acc}%\nVal accuracy: {val_acc}%')
    

    # result = np.asarray(labels)
    # np.savetxt("output_mlp.csv", result, delimiter=",")
