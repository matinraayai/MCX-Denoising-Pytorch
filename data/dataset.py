import scipy.io as spio
import torch
import numpy as np
import torchvision.transforms.functional as transforms
from random import random


class OsaDataset(torch.utils.data.Dataset):
    def __init__(self, path: str,
                 input_label: str,
                 output_label: str,
                 num_samples: int,
                 start_idx: int = 0,
                 max_rotation_angle: float = 90.,
                 rotation_p: float = .7,
                 flip_p: float = .5,
                 ):
        super(OsaDataset, self).__init__()
        assert 0. <= flip_p <= 1.
        assert 0. <= rotation_p <= 1.
        self.path = path
        self.input_label = input_label
        self.output_label = output_label
        self.num_samples = num_samples
        self.start_idx = start_idx
        self.max_rotation_angle = max_rotation_angle
        self.flip_p = flip_p
        self.rotation_p = rotation_p

    def __getitem__(self, item):
        mat_file = spio.loadmat(self.path % (item + self.start_idx), squeeze_me=True)
        x = torch.log(torch.from_numpy(mat_file[self.input_label].astype(np.float32) + 1.)).unsqueeze(0)
        y = torch.log(torch.from_numpy(mat_file[self.output_label].astype(np.float32) + 1.)).unsqueeze(0)
        if random() > self.rotation_p:
            rotation_angle = random() * 2 * self.max_rotation_angle - self.max_rotation_angle
            x, y = transforms.rotate(x, rotation_angle), transforms.rotate(y, rotation_angle)
        if random() > self.flip_p:
            if random() >= 0.5:
                x, y = transforms.hflip(x), transforms.hflip(y)
            else:
                x, y = transforms.vflip(x), transforms.vflip(y)
        return x, y

    def __len__(self):
        return self.num_samples
