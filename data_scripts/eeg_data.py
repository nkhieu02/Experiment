
import torch 
import pandas as pd
from torch.utils.data import Dataset
import os

class EegDataset(Dataset):
    def __init__(self,tgt_transform = None, transform = None, train = True, channel = 'S1obj'):
        self.channel = channel
        self.train = train
        self.label_path = 'data/eeg/train_label.txt' if train else 'data/eeg/test_label.txt'
        self.labels = pd.read_csv(self.label_path, sep=' ', header= None)
        self.labels = self.labels[self.labels[0] == self.channel]
        self.feature_dir = 'data/eeg/train/' if train else 'data/eeg/test/'
        self.transform = transform
        self.tgt_transform = tgt_transform

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        
        feature_path = os.path.join(self.feature_dir, self.labels.iloc[index, 2])
        label = self.labels.iloc[index, 1]
        feature_matrix = pd.read_pickle(feature_path)
        if self.transform:
            feature_matrix = self.transform(feature_matrix)
        if self.tgt_transform:
            label = self.tgt_transform(label)
        
        return feature_matrix, label
        
        
def transform(x):
    assert(x.shape == (256 * 64, 3))
    assert(list(x.columns) == ['Sensor', 'Position', 'Value'])
    value = x['Value']
    value = torch.from_numpy(value.to_numpy()).reshape(256, 64)
    return value   