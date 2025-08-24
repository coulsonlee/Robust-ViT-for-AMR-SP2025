# import h5py
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
import random
import os
import itertools

class Multi_File_Dataloader(Dataset):
    def __init__(self, file_path, loader_dict):
        self.SAMPLE_PRE_MOD = 20_000

        self.source_folder = loader_dict["path"]
        self.prefix = loader_dict["prefix"]
        self.tx_pwr = loader_dict["tx_pwr"]
        self.distance = loader_dict["distance"]
        self.samp_date = loader_dict["samp_date"]
        self.inter = loader_dict["inter"]
        self.dataset_size = loader_dict["dataset_size"]

        check_success = self.check_all_files()
        if not check_success:
            print("Error")
            exit()

        self.init_X_Y()

    def init_X_Y(self):
        print("."*100)
        print("Initializing data...")

        X, Y = None, None
        for filename in self.pickle_filenames:
            with open(filename, 'rb') as f:
                sub_data = pickle.load(f)

            subset_idx = self.get_subset_idx( \
                ttl_samp=sub_data['Y'].shape[0], \
                sub_size=self.dataset_size)

            _X = sub_data['X'][subset_idx, :, :]
            _Y = sub_data['Y'][subset_idx]

            if X is None:
                X = _X.copy()
                Y = _Y.copy()
                self.n_cls = (np.max(Y) - np.min(Y)) + 1
            else:
                X = np.concatenate((X, _X), axis=0)
                Y = np.concatenate((Y, _Y), axis=0)

        print(f"{X.shape = }")
        print(f"{Y.shape = }")
        print(f"{self.n_cls = }")
        self.data = X
        self.labels = Y
        print("Done.")
        print("."*100)
        pass

    def check_all_files(self):
        if not os.path.isdir(self.source_folder):
            print(f"{self.source_folder} is not a legitimate folder.")
            return False
        
        for prefix in self.prefix:
            if not os.path.exists(f"{self.source_folder}{prefix}"):
                print(f"Can not find file for prefix: {self.source_folder}{prefix}")
                return False

        combinations = itertools.product( \
                            self.prefix, \
                            self.tx_pwr, \
                            self.distance, \
                            self.samp_date, \
                            self.inter)
        "Dataset_EIB_Outdoor_TP0_D10_SR20_CF2360_I0.pkl"
        pickle_filenames = []
        print("All pickles name:")
        for comb in combinations:
            filename = f"{comb[0]}_TP{comb[1]}_D{comb[2]}_SR{comb[3]}_CF2360_I{comb[4]}.pkl"
            
            full_filename = f"{self.source_folder}{comb[0]}/{filename}"
            if not os.path.exists(full_filename):
                print(f"Can not find file: {full_filename}")
            else:
                print(f"{full_filename}. Success.")
                pickle_filenames.append(full_filename)
        self.pickle_filenames = pickle_filenames
        return True

    def get_subset_idx(self, ttl_samp, idx_start=0, sample_mod=20_000, sub_size=1.0):
        n_mod = int(ttl_samp/sample_mod)
        ret = []
        for mod_i in range(n_mod):
            idx_start = mod_i*sample_mod
            ret += random.sample(range(idx_start, idx_start+sample_mod), int(sub_size*sample_mod))
        return ret

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = torch.from_numpy(self.data[idx])
        label_item = torch.tensor(self.labels[idx])
        # power_item = torch.tensor(self.power[idx])
        return data_item, label_item

def build_general_loader(path, loader_dict, batch_size):

    dataset = Multi_File_Dataloader(path, loader_dict=loader_dict)

    train_indices, test_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    n_cls = dataset.n_cls
    in_chans = dataset.data.shape[1]

    return train_dataloader, test_dataloader, n_cls, in_chans
