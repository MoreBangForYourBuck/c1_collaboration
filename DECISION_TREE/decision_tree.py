from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from helpers.preprocessing import read_all_data
import torch
from torch.nn import functional as F
import pickle


dtc = DecisionTreeClassifier()


params_dict = {
 'criterion': ['gini', 'entropy'],
 'max_depth': [None, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#  'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],
#  'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#  'min_weight_fraction_leaf': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
#  'max_features': [None, 'auto', 'sqrt', 'log2'],
#  'random_state': [None, 42],
#  'max_leaf_nodes': [None, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#  'min_impurity_decrease': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
#  'class_weight': [None, 'balanced'],
#  'cpp_alpha': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
}

if __name__ == '__main__':
    data_dict = read_all_data()
    X = data_dict['imu'].to_numpy()
    y = data_dict['ann'].to_numpy().flatten()
    del data_dict # Remove to free memory
    
    y = F.one_hot(torch.tensor(y, dtype=torch.long), num_classes=4).detach().numpy()
    
    # Train, val, test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, shuffle=False)
    # X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, shuffle=False)
    
    grid_search = GridSearchCV(dtc, params_dict, cv=5)
    grid_search.fit(X_train, y_train)
    tuned_dtc = grid_search.best_estimator_
    print(tuned_dtc.get_params()) # gini, none
    
    print('Train accuracy:', tuned_dtc.score(X_train, y_train))
    # print(cross_val_score(tuned_dtc, X, y, cv=5))
    print('Val accuracy:', tuned_dtc.score(X_val, y_val))

    filename = 'DECISION_TREE/dtc_model.sav'
    pickle.dump(tuned_dtc, open(filename, 'wb'))
