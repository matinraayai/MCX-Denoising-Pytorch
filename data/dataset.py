import os
from typing import List
import scipy.io as spio
import torch
import numpy as np
import torchvision.transforms.functional as transforms
from numpy.random import choice, random, randint


def _make_path_list(dir_name):
    return sorted([os.path.join(dir_name, file_name) for file_name in os.listdir(dir_name)])


def coin_flip(p_true=0.5):
    return choice([True, False], p=[p_true, 1. - p_true])


def read_norm_sqz_from_mat_file(mat_file, label):
    return torch.log(torch.from_numpy(mat_file[label].astype(np.float32)) + 1.).unsqueeze(0)


def get_random_crop_starting_point(t_shape, crop_size):
    assert crop_size[0] < t_shape[1] and crop_size[1] < t_shape[2]
    return randint(0, t_shape[1] - crop_size[0]), randint(0, t_shape[2] - crop_size[1])


class OsaDataset(torch.utils.data.Dataset):
    def __init__(self, path: str,
                 input_labels: List[str],
                 output_label: str,
                 is_train: bool,
                 crop_size: tuple = None,
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
        self.crop_size = crop_size
        self.is_train = is_train
        if self.is_train:
            self.max_rotation_angle = max_rotation_angle
            self.flip_p = flip_p
            self.rotation_p = rotation_p

    def __getitem_test(self, item):
        mat_file = spio.loadmat(self.paths[item], squeeze_me=True)
        x = {input_label: read_norm_sqz_from_mat_file(mat_file, input_label) for input_label in self.input_labels}
        y = read_norm_sqz_from_mat_file(mat_file, self.output_label)
        if self.crop_size is not None:
            assert self.crop_size[0] < y.shape[1] and self.crop_size[1] < y.shape[2]
            for input_label in self.input_labels:
                x[input_label] = transforms.crop(x[input_label],
                                                 0,
                                                 0,
                                                 self.crop_size[0],
                                                 self.crop_size[1])
            y = transforms.crop(y, 0, 0, self.crop_size[0], self.crop_size[1])
        return x, y

    def __getitem_train(self, item):
        mat_file = spio.loadmat(self.paths[item // len(self.input_labels)], squeeze_me=True)
        input_label = self.input_labels[item % len(self.input_labels)]
        x = read_norm_sqz_from_mat_file(mat_file, input_label)
        y = read_norm_sqz_from_mat_file(mat_file, self.output_label)
        # Crop
        if self.crop_size is not None:
            assert self.crop_size[0] < x.shape[1] and self.crop_size[1] < x.shape[2]
            starting_pos = get_random_crop_starting_point(x.shape, self.crop_size)
            x = transforms.crop(x, starting_pos[0], starting_pos[1], self.crop_size[0], self.crop_size[1])
            y = transforms.crop(y, starting_pos[1], starting_pos[1], self.crop_size[0], self.crop_size[1])
        # Rotation
        if coin_flip(self.rotation_p):
            rotation_angle = random() * 2 * self.max_rotation_angle - self.max_rotation_angle
            x, y = transforms.rotate(x, rotation_angle), transforms.rotate(y, rotation_angle)
        # Flip
        if coin_flip(self.flip_p):
            if coin_flip():
                x, y = transforms.hflip(x), transforms.hflip(y)
            else:
                x, y = transforms.vflip(x), transforms.vflip(y)

        return x, y

    def __getitem__(self, item):
        if self.is_train:
            return self.__getitem_train(item)
        else:
            return self.__getitem_test(item)

    def __len__(self):
        return len(self.paths) * len(self.input_labels)
