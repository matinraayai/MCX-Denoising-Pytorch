from abc import ABC

import tensorflow as tf
import scipy.io as spio
import numpy as np


class OsaDataset(tf.data.Dataset, ABC):
    @staticmethod
    def _generator(root: str, input_file_name: str, label_file_name: str, num_samples: int):
        for i in range(1, num_samples + 1):
            x = spio.loadmat(root + input_file_name % i, squeeze_me=True)['data']
            y = spio.loadmat(root + label_file_name % i, squeeze_me=True)['data']
            for z in range(x.shape[2]):
                yield np.expand_dims(np.array([x[z], y[z]]), axis=-1)

    def __new__(cls, root: str, input_file_name: str, label_file_name: str, num_samples: int):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=tf.dtypes.float32,
            output_shapes=(2, 100, 100, 1),
            args=(root, input_file_name, label_file_name, num_samples)
        )
