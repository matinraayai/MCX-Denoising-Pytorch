"""
Applies the denoising model to the test sets in both 2D and 3D cases.
"""
import os
import torch
from torch.utils.data import DataLoader
from data.dataset import OsaDataset
from model.builder import create_model_from_lightning_checkpoint
import argparse
import tqdm
from config import get_default_inference_cfg
from time import time
import scipy.io as scio
from data.dataset import norm

CUTOFF = 0.03
SCALE = 10 ** 7

def get_args():
    parser = argparse.ArgumentParser(description="Script for denoising model inference")
    parser.add_argument('--config-file', type=str, help='configuration file for the model (yaml)')
    return parser.parse_args()


def main():
    args = get_args()

    print("Command line arguments:")
    print(args)

    # configurations
    cfg = get_default_inference_cfg()
    cfg.update()

    cfg.merge_from_file(args.config_file)

    cfg.freeze()
    print("Configuration details:")
    print(cfg)

    model = create_model_from_lightning_checkpoint(cfg.model.checkpoint, **cfg.model)

    test_dataset = OsaDataset(cfg.dataset.test_path, cfg.dataset.input_labels, cfg.dataset.output_label,
                              False, padding=cfg.dataset.padding)
    test_dataloader = DataLoader(test_dataset, 1, shuffle=False, num_workers=cfg.dataset.dataloader_workers,
                                 pin_memory=True)

    model.train(False)

    iterator_test = tqdm.tqdm(test_dataloader)
    iterator_test.set_description("Benchmark Progress")
    test_output_dir = os.path.join(cfg.output_dir)
    os.makedirs(test_output_dir, exist_ok=True)

    for i, (x_tests, y_test) in enumerate(iterator_test):
        predictions = {label: [] for label in cfg.dataset.input_labels}
        with torch.no_grad():
            for label, (x_original, x_test) in x_tests.items():
                start = time()
                x_original, higher = x_original.cuda(), x_test.cuda()
                lower = norm(x_original * SCALE)
                lower = model(lower)[test_dataset.unpaded_volume_slice]
                higher = model(higher)[test_dataset.unpaded_volume_slice]
                lower = (torch.exp(lower) - 1) / SCALE
                higher = torch.exp(higher) - 1
                prediction = torch.where(x_original[test_dataset.unpaded_volume_slice] > CUTOFF, higher, lower).squeeze().cpu().numpy()
                end = time()
                iterator_test.set_postfix({"Inf. time": "{:.5f}".format(end - start),
                                           "label": label,
                                           "pred mean": prediction.mean().item(),
                                           "input mean": x_original.mean().item(),
                                           "input shape": x_test.shape})
                predictions[label].append(prediction)

        scio.savemat(os.path.join(cfg.output_dir, f'{i}.mat'), predictions)


if __name__ == '__main__':
    main()
