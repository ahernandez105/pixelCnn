import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle

class Data(Dataset):
    def __init__(self, array, device='cpu', mean=None, std=None):
        self.N, self.H, self.W, self.C  = array.shape
        self.ttl_dims = self.C * self.H * self.W
        self.array = array
        self.device = device
        if mean is None and std is None:
            self.mean = np.mean(self.array, axis=(0,1,2))
            self.std = np.std(self.array, axis=(0,1,2))
        else:
            self.mean = mean
            self.std = std

    def __len__(self):
        return self.N
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        
        x = self.array[idx].copy().astype(float)
        for i in range(self.C):
            x[:,:,i] = (x[:,:,i] - self.mean[i])/self.std[i]

        return {
            'x': torch.tensor(x, dtype=torch.float).to(self.device).permute(2, 0, 1),
            'y': torch.tensor(self.array[idx], dtype=torch.long).to(self.device).permute(2, 0, 1)}
    
    @staticmethod
    def collate_fn(batch):
        bsize = len(batch)

        return {
            'x': torch.stack([batch[i]['x'] for i in range(bsize)], dim=0).contiguous(),
            'y': torch.stack([batch[i]['y'] for i in range(bsize)], dim=0).contiguous()}
    
    @staticmethod
    def read_pickle(fn, dset):
        assert dset=='train' or dset=='test'
        with open(fn, 'rb') as file:
            data = pickle.load(file)[dset]
            if 'mnist.pkl' in fn:
                data = (data > 127.5).astype('uint8')

            return data