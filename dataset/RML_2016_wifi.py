import h5py
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import pickle
import numpy as np

class RML_2016_real(Dataset):
    def __init__(self, file_path,crop_type='center'):
        with open(file_path, 'rb') as f:
            self.all_data = pickle.load(f)
        self.data_keys = list(self.all_data.keys())
        self.data_keys = list(self.all_data.keys())
        self.samples = []
        self.crop_type = crop_type

        for key in self.data_keys:
            X = self.all_data[key]['X']
            Y = self.all_data[key]['Y']
            for i in range(len(X)):
                self.samples.append((X[i], Y[i]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X, Y = self.samples[idx]
        X_processed = self.process_X(X)
        X_tensor = torch.tensor(X_processed, dtype=torch.float32)
        Y_tensor = torch.tensor(Y,dtype=torch.long)
        return X_tensor, Y_tensor

    def process_X(self, X):
        if self.crop_type == 'center':
            return self.center_crop(X)
        elif self.crop_type == 'random':
            return self.random_crop(X)
        else:
            raise ValueError("Invalid crop type. Choose either 'center' or 'random'.")

    def center_crop(self, X):
        if X.shape[1] > 128:
            start_idx = (X.shape[1] - 128) // 2
            end_idx = start_idx + 128
            cropped_sample = X[:, start_idx:end_idx]
            return cropped_sample
        else:
            return X

    def random_crop(self, X):
        if X.shape[1] > 128:
            max_start_idx = X.shape[1] - 128
            start_idx = np.random.randint(0, max_start_idx)
            end_idx = start_idx + 128
            cropped_sample = X[:, start_idx:end_idx]
            return cropped_sample
        else:
            return X


def build_RML_2016_wifi(path='/Users/lee/Desktop/WirelessProtocol_325_1m.pkl',batch_size=128,crop_type='center'):

    dataset = RML_2016_real(path,crop_type=crop_type)

    train_indices, test_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader,test_dataloader

