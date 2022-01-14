import os
import argparse
import tqdm
from time import time
import scipy.io as scio
from skimage.restoration import denoise_nl_means, estimate_sigma
from utils import read_mat_files
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description="Script for applying NLM on test datasets")
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
            patch_kw = dict(patch_size=3,
                            patch_distance=5,
                            multichannel=False)
            log_simulation = np.log1p(simulation.copy())
            sigma_est_nlm = estimate_sigma(log_simulation, multichannel=False)
            # NLM is very unstable so depending on the stability either do it in log scale or normal scale
            if sigma_est_nlm != sigma_est_nlm:
                prediction = denoise_nl_means(simulation.copy(), sigma=sigma_est, fast_mode=False,
                                              **patch_kw)
                if prediction.mean() != prediction.mean():
                    prediction = denoise_nl_means(simulation.copy(), sigma=0.0001, fast_mode=False,
                                                  **patch_kw)
            else:
                prediction = np.exp(denoise_nl_means(log_simulation, sigma=sigma_est_nlm, fast_mode=False,
                                                     **patch_kw))
            predictions[label] = prediction
            end = time()
            iterator.set_postfix({"Inf. time": "{:.5f}".format(end - start),
                                  'Sigma est': sigma_est,
                                  'Mean': prediction.mean()})
        scio.savemat(os.path.join(output_path, file_name), predictions)


if __name__ == '__main__':
    main()
