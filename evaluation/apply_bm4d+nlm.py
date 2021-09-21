import os
import argparse
import tqdm
from time import time
import scipy.io as scio
import bm3d
from skimage.restoration import denoise_nl_means, estimate_sigma
from utils import read_mat_files


def get_args():
    parser = argparse.ArgumentParser(description="Script for applying NLM and BM4D/3D on test datasets")
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

    bm4d_output_path = os.path.join(args.output_path, 'bm4d')
    nlm_output_path = os.path.join(args.output_path, 'nlm')
    os.makedirs(bm4d_output_path, exist_ok=True)
    os.makedirs(nlm_output_path, exist_ok=True)
    iterator = tqdm.tqdm(simulation_files_mapping.keys())

    for file_name in iterator:
        bm3d_predictions = {}
        nlm_predictions = {}
        current_file = simulation_files_mapping[file_name]
        for label, simulation in current_file.items():
            sigma_est = estimate_sigma(simulation, multichannel=False)
            patch_kw = dict(patch_size=3,
                            patch_distance=7,
                            multichannel=False)
            start = time()
            prediction = bm3d.bm3d(simulation, sigma_psd=sigma_est)
            bm3d_predictions[label] = prediction
            prediction = denoise_nl_means(simulation.copy(), sigma=sigma_est, fast_mode=False, **patch_kw)
            nlm_predictions[label] = prediction
            end = time()
            iterator.set_postfix({"Inf. time": "{:.5f}".format(end - start), 'Sigma est': sigma_est})
        scio.savemat(os.path.join(bm4d_output_path, file_name), bm3d_predictions)
        scio.savemat(os.path.join(nlm_output_path, file_name), nlm_predictions)


if __name__ == '__main__':
    main()
