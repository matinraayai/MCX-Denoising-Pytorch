import os
import matplotlib
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import tqdm

def make_path_list(dir_name):
    return sorted([(file_name, os.path.join(dir_name, file_name)) for file_name in os.listdir(dir_name)])


def visualize(x, y, prediction, output_path, matplotlib_backend='Agg'):
    matplotlib.use(matplotlib_backend)
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(x)
    axs[0].set_title('Input')
    axs[1].imshow(y)
    axs[1].set_title('Label')
    axs[2].imshow(prediction)
    axs[2].set_title('Prediction')
    fig.savefig(output_path)
    plt.close(fig)


def all_labels_in_mat(path, only_mats=True):
    """
    Returns all the labels in a matlab mat file. It can filter out only the matrix data types if required
    :param path: path to the mat file
    :param only_mats: if True, only returns labels that point to matrix data; Returns all the labels in the file if
    False
    :return: A set with all the labels as strings
    """
    info = sio.whosmat(path)
    return set([i[0] for i in info if i[2] == 'single' or i[2] == 'double'])


def read_mat_files(directory, mapping='label to filename'):
    path_list = make_path_list(directory)
    output = {}
    iterator = tqdm.tqdm(enumerate(path_list))
    for i, (file_name, path) in iterator:
        iterator.set_description(f'Reading from {i}-th path: {path}')
        mat_file = sio.loadmat(path)
        labels = all_labels_in_mat(path)
        iterator.set_postfix({'labels': labels})
        for label in labels:
            if mapping == 'label to filename':
                if label not in output:
                    output[label] = {}
                output[label][file_name] = mat_file[label]
            if mapping == 'filename to label':
                if file_name not in output:
                    output[file_name] = {}
                output[file_name][label] = mat_file[label]
    return output


def prepare_label_to_idx_mapping_for_analysis(mapping):
    for label, sims in mapping.items():
        sorted_file_names = sorted(sims.keys())
        sims_as_array = np.array([sims[file_name] for file_name in sorted_file_names])
        mapping[label] = sims_as_array
    return mapping
