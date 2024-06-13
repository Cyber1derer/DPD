import numpy as np
import torch
from torch.utils.data import Dataset


class SignalDataset(Dataset):
    def __init__(self, xo, ref, samples_num=1024, step=32):  # x_e, x_o, ref - numpy arrays or lists of shape (n)
        if not len(xo) == len(xe) == len(ref):
            self.n = min(len(xe, xo, ref))
        else:
            self.n = len(xo)

        self.X = torch.tensor( xo[:self.n])  # shape (2, n)
        self.y = torch.tensor(ref[:self.n]).reshape((1, -1))  # shape (1, n)
        self.samples_num = samples_num
        self.step = step

    def __len__(self):
        return (self.n - self.samples_num) // self.step + 1

    def __getitem__(self, index):
        return self.X[:, index * self.step:index * self.step + self.samples_num], \
               self.y[:, index * self.step:index * self.step + self.samples_num]
