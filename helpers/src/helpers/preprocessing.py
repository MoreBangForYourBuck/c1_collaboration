from typing import Dict
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch


def expand_ann(imu_t:list, ann:list, ann_t:list) -> Dict[str, list]:
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
    for i, curr_imu_time in enumerate(imu_t):
        
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

def read_all_data(dir_path:str='processed_training_data') -> Dict[str, pd.DataFrame]:
    imu = pd.DataFrame()
    imu_t = pd.DataFrame()
    ann = pd.DataFrame()
    ann_t = pd.DataFrame()

    for f in os.listdir(dir_path):
        x = pd.read_csv(f'{dir_path}/{f}')
        if f[-7:] == '__x.csv':
            imu = pd.concat([imu, x], axis=0)
        
        elif f[-12:] == '__x_time.csv':
            imu_t = pd.concat([imu_t, x], axis=0)
            
        elif f[-7:] == '__y.csv':
            ann = pd.concat([ann, x], axis=0)
            
        elif f[-12:] == '__y_time.csv':
            ann_t = pd.concat([ann_t, x], axis=0)

    return {
        'imu': imu.reset_index(drop=True),
        'imu_t': imu_t.reset_index(drop=True),
        'ann': ann.reset_index(drop=True),
        'ann_t': ann_t.reset_index(drop=True)
    }

def normalize_data(df:pd.DataFrame, method:str) -> pd.DataFrame:
    if method == 'divide_mean':
        for col in df:
            df[col] = df[col] / df[col].mean()
    else:
        if method == 'standard_scale':
            scaler = StandardScaler()
            scaler.fit(df)
        elif method == 'min_max_scale':
            scaler = MinMaxScaler()
            scaler.fit(df)
        return pd.DataFrame(scaler.transform(df), columns=df.columns), scaler

def cross_entropy_weights(sample_vector):
    '''Takes vector of amounts of samples, returns vector of weights'''
    s = sum(sample_vector)
    inverse_fraction = [s/c for c in sample_vector]
    s = sum(inverse_fraction)
    weights = [f/s for f in inverse_fraction]
    return torch.tensor(weights)