import os
from typing import List
import scipy.io as spio
import torch
import numpy as np
import torchvision.transforms.functional as transforms
from numpy.random import choice, random


def _make_path_list(dir_name):
    return sorted([os.path.join(dir_name, file_name) for file_name in os.listdir(dir_name)])


def coin_flip(p_true=0.5):
    return choice([True, False], [p_true, 1. - p_true])


class OsaDataset(torch.utils.data.Dataset):
    def __init__(self, path: str,
                 input_labels: List[str],
                 output_label: str,
                 max_rotation_angle: float = 90.,
                 rotation_p: float = .7,
                 flip_p: float = .5,
                 ):
        super(OsaDataset, self).__init__()
        assert 0. <= flip_p <= 1.
        assert 0. <= rotation_p <= 1.
        self.paths = _make_path_list(path)
        self.input_labels = input_labels
        self.output_label = output_label
        self.max_rotation_angle = max_rotation_angle
        self.flip_p = flip_p
        self.rotation_p = rotation_p

    def __getitem__(self, item):
        mat_file = spio.loadmat(self.paths[item // len(self.input_labels)], squeeze_me=True)
        input_label = self.input_labels[item % len(self.input_labels)]
        x = torch.log(torch.from_numpy(mat_file[input_label].astype(np.float32) + 1.)).unsqueeze(0)
        y = torch.log(torch.from_numpy(mat_file[self.output_label].astype(np.float32) + 1.)).unsqueeze(0)
        if coin_flip(self.rotation_p):
            rotation_angle = random() * 2 * self.max_rotation_angle - self.max_rotation_angle
            x, y = transforms.rotate(x, rotation_angle), transforms.rotate(y, rotation_angle)
        if coin_flip(self.flip_p):
            if coin_flip():
                x, y = transforms.hflip(x), transforms.hflip(y)
            else:
                x, y = transforms.vflip(x), transforms.vflip(y)
        return x, y

    def __len__(self):
        return len(self.paths)
