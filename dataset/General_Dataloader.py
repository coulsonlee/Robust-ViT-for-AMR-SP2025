import h5py
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
import random

class RML_2016_real(Dataset):
    def __init__(self, file_path, sub_size=1.0):
        with open(file_path, 'rb') as f:
            self.file = pickle.load(f)

        self.n_cls = (np.max(self.file['Y']) - np.min(self.file['Y'])) + 1

        self.ttl_sample = self.file['Y'].shape[0]
        self.sub_size = int(sub_size*self.ttl_sample)

        subset_idx = self.get_subset_idx()
        print(f"{self.ttl_sample = }, {len(subset_idx) = }")

        self.data = self.file['X'][subset_idx, :, :]
        self.labels = self.file['Y'][subset_idx]

        if 'Z' in self.file:
            self.power = self.file['Z'][subset_idx]
        else:
            self.power = np.zeros(self.labels.shape)

        print(self.labels.shape)

    def get_subset_idx(self):
        ret = random.sample(range(self.ttl_sample), self.sub_size)
        return ret

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = torch.from_numpy(self.data[idx])
        label_item = torch.tensor(self.labels[idx])
        power_item = torch.tensor(self.power[idx])
        return data_item, label_item, power_item

def build_general_loader(path, batch_size, sub_size=1.0):

    dataset = RML_2016_real(path, sub_size=sub_size)

    train_indices, test_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    n_cls = dataset.n_cls
    in_chans = dataset.data.shape[1]

    return train_dataloader, test_dataloader, n_cls, in_chans
