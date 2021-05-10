"""
Applies the denoising model to the test sets in both 2D and 3D cases.
"""
import os
import torch
from torch.utils.data import DataLoader
from data.dataset import OsaDataset
from model.builder import load_model_from_checkpoint
import argparse
import tqdm
from config import get_cfg_defaults
from time import time
import scipy.io as scio


def get_args():
    parser = argparse.ArgumentParser(description="Script for denoising model inference")
    parser.add_argument('--config-file', type=str, help='configuration file for the model (yaml)')
    return parser.parse_args()


def infer(model, x, denoise_3d_with_2d):
    if denoise_3d_with_2d:
        y = torch.zeros_like(x)
        # Infer on x-axis
        for i in range(x.shape[2]):
            x_2d = x[:, :, i, :, :].squeeze(2)
            y[:, :, i, :, :] = model(x_2d)
        # Infer on y-axis
        for i in range(x.shape[3]):
            x_2d = x[:, :, :, i, :].squeeze(3)
            y[:, :, :, i, :] = model(x_2d)
        # Infer on z-axis
        for i in range(x.shape[4]):
            x_2d = x[:, :, :, :, i]
            y[:, :, :, :, i] = model(x_2d)
        return y
    else:
        return model(x)


def main():
    args = get_args()

    print("Command line arguments:")
    print(args)

    # configurations
    cfg = get_cfg_defaults()
    cfg.update()

    cfg.merge_from_file(args.config_file)

    cfg.freeze()
    print("Configuration details:")
    print(cfg)

    # Load model and its checkpoint
    model = load_model_from_checkpoint(cfg.inference.checkpoint_dir, **cfg.model)
    test_dataset = OsaDataset(cfg.dataset.test_path, cfg.dataset.input_labels, cfg.dataset.output_label,
                              False, cfg.dataset.crop_size)
    test_dataloader = DataLoader(test_dataset, 1, shuffle=False, num_workers=cfg.dataset.dataloader_workers,
                                 pin_memory=True)

    model.train(False)

    iterator_test = tqdm.tqdm(test_dataloader)
    iterator_test.set_description(f"Benchmark Progress")
    test_output_dir = os.path.join(cfg.inference.output_dir)
    os.makedirs(test_output_dir, exist_ok=True)

    for i, (x_tests, y_test) in enumerate(iterator_test):
        predictions = {label: [] for label in cfg.dataset.input_labels}
        x_tests = {label: x_test.cuda() for label, x_test in x_tests.items()}
        with torch.no_grad():
            for label, x_test in x_tests.items():
                start = time()
                prediction = infer(model, x_test, cfg.inference.denoise_3d_with_2d)
                prediction = (torch.exp(prediction) - 1).squeeze().cpu().numpy()
                end = time()
                iterator_test.set_postfix({"Inf. time": "{:.5f}".format(end - start)})
                predictions[label].append(prediction)
        scio.savemat(cfg.inference.output_dir + f'{i}.mat', predictions)


if __name__ == '__main__':
    main()
