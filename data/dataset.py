import os
from typing import List
import scipy.io as spio
import torch
from numpy.random import randint
from .augmentation import Compose
from torch.nn.functional import pad


def make_path_list(dir_name):
    """
    Makes a list of paths to files present in the directory. Used by the dataset to then read the files.
    :param dir_name: the directory where the file are present
    :return: A sorted list of paths to all the files present in the directory
    """
    return sorted([os.path.join(dir_name, file_name) for file_name in os.listdir(dir_name)])


def read_norm_sqz_from_mat_file(mat_file, label):
    return torch.log1p(torch.from_numpy(mat_file[label])).unsqueeze(0).float()


def random_crop(x, y, crop_size):
    t_shape = x.shape
    crop_pos = tuple(randint(0, t_shape[i + 1] - crop_size[0]) for i in range(len(t_shape) - 1))
    return crop_volume(x, crop_pos, crop_size), crop_volume(y, crop_pos, crop_size)


def crop_volume(vol, crop_pos, crop_size):
    crop_slice = (slice(1),) + tuple(slice(crop_pos[i], crop_pos[i] + crop_size[i - 1]) for i in range(len(crop_pos)))
    return vol[crop_slice]


def pad_volume_to_nearest_4(vol):
    """
    Pads the input volume (both 2D and 3D) so that each dimension is a multiple of 4, to ensure that it can be
    fed into all CNN models
    :param vol:
    :return: Padded volume
    """
    vol_shape = vol.shape[1:]
    vol_padding = (4 - (s % 4) for s in vol_shape)
    padding_input = ()
    for p in vol_padding:
        padding_input += (0, p)
    return pad(vol, padding_input, 'constant', 0)


class OsaDataset(torch.utils.data.Dataset):
    def __init__(self, path: str,
                 input_labels: List[str],
                 output_label: str,
                 is_train: bool,
                 crop_size: tuple = None,
                 augmentor: Compose = None):
        super(OsaDataset, self).__init__()
        self.paths = make_path_list(path)
        self.input_labels = input_labels
        self.output_label = output_label
        self.crop_size = crop_size
        self.is_train = is_train
        self.augmentor = augmentor
        self.mat_files = [spio.loadmat(path, squeeze_me=True) for path in self.paths]
        sample_output_tensor_shape = read_norm_sqz_from_mat_file(self.mat_files[0], self.output_label).shape
        self.unpaded_volume_slice = (slice(1),) + tuple(slice(0, s) for s in sample_output_tensor_shape)

    def __getitem_test(self, item):
        mat_file = self.mat_files[item]
        x = {input_label: read_norm_sqz_from_mat_file(mat_file, input_label) for input_label in self.input_labels}
        y = read_norm_sqz_from_mat_file(mat_file, self.output_label)
        if self.crop_size is not None:
            assert self.crop_size[0] < y.shape[1] and self.crop_size[1] < y.shape[2]
            crop_pos = (0, ) * (len(y.shape) - 1)
            for input_label in self.input_labels:
                x[input_label] = crop_volume(x[input_label], crop_pos, self.crop_size)
            y = crop_volume(y, crop_pos, self.crop_size)
        for input_label in self.input_labels:
            x[input_label] = pad_volume_to_nearest_4(x[input_label])
        y = pad_volume_to_nearest_4(y)
        return x, y

    def __getitem_train(self, item):
        mat_file = self.mat_files[item // len(self.input_labels)]
        input_label = self.input_labels[item % len(self.input_labels)]
        x = read_norm_sqz_from_mat_file(mat_file, input_label)
        y = read_norm_sqz_from_mat_file(mat_file, self.output_label)
        # Crop
        if self.crop_size is not None:
            assert self.crop_size[0] < x.shape[1] and self.crop_size[1] < x.shape[2]
            x, y = random_crop(x, y, self.crop_size)
        # Augment
        if self.augmentor is not None:
            augmented = self.augmentor({'image': x, 'label': y})
            x, y = augmented['image'], augmented['label']
        return input_label, x, y

    def __getitem__(self, item):
        if self.is_train:
            return self.__getitem_train(item)
        else:
            return self.__getitem_test(item)

    def __len__(self):
        if self.is_train:
            return len(self.paths) * len(self.input_labels)
        else:
            return len(self.paths)
