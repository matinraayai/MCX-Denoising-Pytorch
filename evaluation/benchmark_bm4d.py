import os
import argparse
import tqdm
from time import time
import scipy.io as scio
import bm3d
from skimage.restoration import denoise_nl_means, estimate_sigma
from utils import read_mat_files
import numpy as np


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Script for applying BM4D/3D on test datasets")
    parser.add_argument('--simulation-path', type=str, help='path to test simulation dataset')
    parser.add_argument('--output-path', type=str, help='path to save the results')
    return parser.parse_args()


def main():
    r"""Main function."""
    # arguments
    args = get_args()
    print("Command line arguments:")
    print(args)

    simulation_files_mapping = read_mat_files(args.simulation_path, mapping='filename to label')

    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    iterator = tqdm.tqdm(simulation_files_mapping.keys())

    for file_name in iterator:
        predictions = {}
        current_file = simulation_files_mapping[file_name]
        for label, simulation in current_file.items():
            start = time()
            sigma_est = estimate_sigma(simulation, multichannel=False)
            prediction = bm3d.bm3d(simulation, sigma_psd=sigma_est)
            predictions[label] = prediction
            end = time()
            iterator.set_postfix({"Inf. time": "{:.5f}".format(end - start),
                                  'Sigma est BM3D': sigma_est,
                                  'BM3D Mean': prediction.mean()})
        scio.savemat(os.path.join(output_path, file_name), predictions)


if __name__ == '__main__':
    main()
