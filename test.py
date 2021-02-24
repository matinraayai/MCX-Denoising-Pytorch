import os
import torch
from torch.utils.data import DataLoader
from data.dataset import OsaDataset
from model import get_model
import argparse
import tqdm
import matplotlib
import matplotlib.pyplot as plt
from config import get_cfg_defaults
from model.loss import VGGLoss, PSNR, SSIM
import torch.nn as nn


def get_args():
    parser = argparse.ArgumentParser(description="Model Inference")
    parser.add_argument('--config-file', type=str, help='configuration file (yaml)')
    parser.add_argument('--checkpoint', type=str, default=None, help='path to load the checkpoint')
    return parser.parse_args()


def main():
    r"""Main function."""
    # arguments
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

    matplotlib.use('Agg')
    model = get_model(**cfg.model).cuda()
    state_dict = torch.load(cfg.model.starting_checkpoint)
    model.load_state_dict(state_dict.state_dict())

    mse_criterion = nn.MSELoss()
    mse_loss = 0.

    ssim_criterion = SSIM(cfg.loss.ssim).cuda()
    ssim_loss = 0.

    psnr_criterion = PSNR()
    psnr_loss = 0.

    vgg_criterion = VGGLoss().cuda()
    vgg_loss = 0.

    test_dataset = OsaDataset(cfg.dataset.valid_path, ['x1e5'], cfg.dataset.output_label, 0., 0., 0.)
    test_dataloader = DataLoader(test_dataset, 1, shuffle=False,
                                 num_workers=cfg.dataset.dataloader_workers,
                                 pin_memory=True)

    model.train(False)

    iterator_test = tqdm.tqdm(test_dataloader)
    iterator_test.set_description(f"Test Progress")
    test_output_dir = os.path.join(cfg.inference.output_dir)
    os.makedirs(test_output_dir)
    for i, (x_test, y_test) in enumerate(iterator_test):
        with torch.no_grad():
            x_test, y_test = x_test.cuda(), y_test.cuda()
            logits = model(x_test)
            # Loss Updates
            mse_loss += mse_criterion(y_test, logits)
            ssim_loss += ssim_criterion(y_test, logits)
            psnr_loss += psnr_criterion(y_test, logits)
            vgg_loss += vgg_criterion(y_test, logits)

            fig, axs = plt.subplots(1, 3)
            axs[0].imshow(x_test.squeeze().cpu().numpy())
            axs[0].set_title('Input')
            axs[1].imshow(y_test.squeeze().cpu().numpy())
            axs[1].set_title('Label')
            axs[2].imshow(logits.squeeze().cpu().numpy())
            axs[2].set_title('Prediction')
            fig.savefig(os.path.join(test_output_dir, f'{i}.png'))
            plt.close(fig)

        print(f"Summary:\n"
              f"Mean MSE Loss: {mse_loss / len(test_dataset)}"
              f"Mean SSIM Loss: {ssim_loss / len(test_dataset)}"
              f"Mean PSNR Loss: {psnr_loss / len(test_dataset)}"
              f"Mean VGG Loss: {vgg_loss / len(test_dataset)}")


if __name__ == '__main__':
    main()
