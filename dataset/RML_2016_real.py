import h5py
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import pickle
import numpy as np

class RML_2016_real(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'rb') as f:
            self.file = pickle.load(f)
            self.data = self.file['X']
            self.labels = self.file['Y']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = torch.tensor(self.data[idx],dtype=torch.float32)
        label_item = torch.tensor(self.labels[idx],dtype=torch.long)
        return data_item, label_item

def build_RML_2016_real(path,batch_size):

    dataset = RML_2016_real(path)

    train_indices, test_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader,test_dataloader


def get_noise_data(path):
    dataset = RML_2016_real(path)
    data,label = dataset[-1]
    return data