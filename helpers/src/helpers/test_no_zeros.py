from helpers.preprocessing import read_all_data
import pandas as pd
import numpy as np

def read_all_data_no_zeros(dir_path:str='processed_training_data'):
    data_dict = read_all_data(dir_path)
    mask = np.array(data_dict['ann']) != 0
    
    for key in data_dict.keys():
        data_dict[key] = np.array(data_dict[key][mask])
        print(key, data_dict[key].shape)
    return data_dict
    

read_all_data_no_zeros()