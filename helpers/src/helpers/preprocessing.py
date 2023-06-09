from typing import Dict, List, Union, Tuple, Optional
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch


def expand_ann(imu_t:List[float], ann:List[int], ann_t:List[float]) -> Dict[str, List[Union[int, float]]]:
    class MemIter():
        def __init__(self, v:list):
            self.v = v
            self.i = 0
            self.val = v[self.i]
            
        def increment(self):
            if self.i < len(self.v) - 1: # If not at end of list
                self.i += 1
                self.val = self.v[self.i]


    mem_ann = MemIter(ann)
    mem_ann_t = MemIter(ann_t)

    ann_out = []
    for curr_imu_time in imu_t:
        
        if curr_imu_time < mem_ann_t.val:
            ann_out.append(mem_ann.val)
            
        else:
            ann_out.append(mem_ann.val)
            mem_ann.increment()
            mem_ann_t.increment()
                
    return {
        'ann': ann_out,
        'ann_time': imu_t
    }
    
def nearest_neighbors_ann(imu_t:List[float], ann:List[int], ann_t:List[float]) -> Dict[str, List[Union[int, float]]]:
    ann_t_arr = np.array(ann_t)
    
    out = []
    for x_t in imu_t:
        diff = np.absolute(ann_t_arr - x_t)
        index = diff.argmin()
        out.append(ann[index])
        
    return {
        'ann': out,
        'ann_time': imu_t
    }

def read_all_data(dir_path:str='processed_training_data') -> Dict[str, pd.DataFrame]:
    imu = pd.DataFrame()
    imu_t = pd.DataFrame()
    ann = pd.DataFrame()
    ann_t = pd.DataFrame()
    
    imu_i = []
    imu_t_i = []
    ann_i = []
    ann_t_i = []
    
    dirpaths = os.listdir(dir_path)
    dirpaths.sort()
    for f in dirpaths:
        print(f)
        x = pd.read_csv(f'{dir_path}/{f}')
        if f[-7:] == '__x.csv':
            imu_i.append(f[-13:-7])
            imu = pd.concat([imu, x], axis=0)
        
        elif f[-12:] == '__x_time.csv':
            imu_t_i.append(f[-18:-12])
            imu_t = pd.concat([imu_t, x], axis=0)
            
        elif f[-7:] == '__y.csv':
            ann_i.append(f[-13:-7])
            ann = pd.concat([ann, x], axis=0)
            
        elif f[-12:] == '__y_time.csv':
            ann_t_i.append(f[-18:-12])
            ann_t = pd.concat([ann_t, x], axis=0)
            
    assert imu_i == imu_t_i == ann_i == ann_t_i, 'File names do not match'
    print(imu_i, ann_i)

    return {
        'imu': imu.reset_index(drop=True),
        'imu_t': imu_t.reset_index(drop=True),
        'ann': ann.reset_index(drop=True),
        'ann_t': ann_t.reset_index(drop=True)
    }

def read_all_data_no_zeros(dir_path:str='processed_training_data'):
    data_dict = read_all_data(dir_path)
    mask = (np.array(data_dict['ann']) != 0) & (np.array(data_dict['ann']) != 3)
    
    for key in data_dict.keys():
        data_dict[key] = np.array(data_dict[key][mask])
        print(key, data_dict[key].shape)
    return data_dict

def get_distribution(all_y:List[float], num_classes:int=4, class_offset:int=0) -> Dict[str, List[float]]:
    return {
        'counts': [all_y.count(x) for x in range(class_offset, num_classes+class_offset)],
        'fracs': [all_y.count(x)/len(all_y) for x in range(class_offset, num_classes+class_offset)]
    }

def normalize_data(arr:np.ndarray, method:str) -> Tuple[np.ndarray, Optional[Union[StandardScaler, MinMaxScaler]]]:
    if method == 'mean':
        # for col in df:
        #     df[col] = df[col] / df[col].mean()
        return (
            np.mean(axis=0),
            None
        )
    else:
        if method == 'standard':
            scaler = StandardScaler()
            scaler.fit(arr)
        elif method == 'minmax':
            scaler = MinMaxScaler()
            scaler.fit(arr)
        return (
            scaler.transform(arr),
            scaler
        )

def cross_entropy_weights(percent_vector) -> torch.Tensor:
    inverse_fraction = [1/c for c in percent_vector]
    s = sum(inverse_fraction)
    weights = [f/s for f in inverse_fraction]
    return torch.tensor(weights)
