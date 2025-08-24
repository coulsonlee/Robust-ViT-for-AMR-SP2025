import h5py
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

class RML_2018(Dataset):
    def __init__(self, file_path):
        self.file = h5py.File(file_path, 'r')
        self.data = self.file['X']
        self.labels = self.file['Y']
        self.snr = self.file['Z']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = torch.from_numpy(self.data[idx])
        label_item = torch.from_numpy(self.labels[idx])
        snr_item = torch.from_numpy(self.snr[idx])
        return data_item, label_item, snr_item

def build_RML_2018(path,batch_size):

    dataset = RML_2018(path)

    train_indices, test_indices = train_test_split(range(len(dataset)), test_size=0.1, random_state=42)

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader,test_dataloader