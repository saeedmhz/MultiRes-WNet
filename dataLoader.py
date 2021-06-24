from torch.utils import data
import torch
import numpy as np
import py7zr

def _loader(file_name):
    data_numpy = np.loadtxt(file_name)

    return data_numpy.reshape(1,len(data_numpy),len(data_numpy))


class _dataset(data.Dataset):
    def __init__(self, path_input, path_output, batch_size, dataset_size):
        self.batch_size = batch_size
        self.path_input = path_input
        self.path_output = path_output
        self.loader = _loader
        self.dataset_size = dataset_size

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        file_name_input = self.path_input + str(index) + '.npy'
        data_input = self.loader(file_name_input)
        tensor_input = torch.tensor(data_input, dtype=float)
        file_name_output = self.path_output + str(index) + '.npy'
        data_output = self.loader(file_name_output)
        tensor_output = torch.tensor(data_output, dtype=float)
        inputs = tensor_input
        outputs = tensor_output

        return inputs, outputs


def data_load(path_input: str, path_output: str, batch_size: int, dataset_size: int, shuffle: bool):
    dataSet = _dataset(path_input, path_output, batch_size, dataset_size)
    return data.DataLoader(dataset=dataSet, batch_size=batch_size, shuffle=shuffle)

