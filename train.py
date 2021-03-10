import os
import torch
from torch.utils.data import DataLoader
from data.dataset import OsaDataset
from model.builder import Criterion, get_model
from solver.builder import build_optimizer, build_lr_scheduler
import argparse
import tqdm
import matplotlib
import matplotlib.pyplot as plt
from config import get_cfg_defaults


def get_args():
    parser = argparse.ArgumentParser(description="De-noising model training script")
    parser.add_argument('--config-file', type=str, help='configuration file (yaml)')
    return parser.parse_args()


def main():
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
    if cfg.model.starting_checkpoint:
        state_dict = torch.load(cfg.model.starting_checkpoint)
        model.load_state_dict(state_dict.state_dict())

    optimizer = build_optimizer(cfg, model)

    lr_scheduler = build_lr_scheduler(cfg, optimizer)

    loss = Criterion(**cfg.loss)

    train_dataset = OsaDataset(cfg.dataset.train_path, cfg.dataset.input_labels,
                               cfg.dataset.output_label, True, cfg.dataset.crop_size, cfg.dataset.max_rotation_angle,
                               cfg.dataset.rotation_p, cfg.dataset.flip_p)
    valid_dataset = OsaDataset(cfg.dataset.valid_path, ['x1e5'],
                               cfg.dataset.output_label, False, cfg.dataset.crop_size)
    train_dataloader = DataLoader(train_dataset, cfg.solver.batch_size, shuffle=True,
                                  num_workers=cfg.dataset.dataloader_workers,
                                  pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, 1, shuffle=False,
                                  num_workers=cfg.dataset.dataloader_workers,
                                  pin_memory=True)

    for epoch_num in range(cfg.solver.total_iterations):
        model.train()
        iterator = tqdm.tqdm(train_dataloader)
        iterator.set_description(f"Epoch #{epoch_num}")

        for iteration, (x_batch_train, y_batch_train) in enumerate(iterator):
            x_batch_train, y_batch_train = x_batch_train.cuda(), y_batch_train.cuda()
            optimizer.zero_grad()
            batch_prediction = model(x_batch_train)
            loss_value = loss(y_batch_train, batch_prediction)
            loss_value.backward()
            if (iteration + 1) % cfg.solver.iteration_step == 0:
                optimizer.step()
            iterator.set_postfix({"Batch Model Loss": "{:.5f}".format(loss_value.item())})
        lr_scheduler.step()
        model.train(False)
        iterator_valid = tqdm.tqdm(valid_dataloader)
        iterator_valid.set_description(f"Validation Epoch #{epoch_num}")
        total_loss = 0.
        if epoch_num % cfg.solver.iteration_save == 0:
            curr_epoch_chkpt_dir = os.path.join(cfg.checkpoint_dir, str(epoch_num))
            os.makedirs(curr_epoch_chkpt_dir, exist_ok=True)
        for i, (x_batch_valid, y_batch_valid) in enumerate(iterator_valid):
            with torch.no_grad():
                x_batch_valid, y_batch_valid = x_batch_valid.cuda(), y_batch_valid.cuda()
                batch_prediction = model(x_batch_valid)
                if epoch_num % cfg.solver.iteration_save == 0:
                    fig, axs = plt.subplots(1, 3)
                    axs[0].imshow(x_batch_valid.squeeze().cpu().numpy())
                    axs[0].set_title('Input')
                    axs[1].imshow(y_batch_valid.squeeze().cpu().numpy())
                    axs[1].set_title('Label')
                    axs[2].imshow(batch_prediction.squeeze().cpu().numpy())
                    axs[2].set_title('Prediction')
                    fig.savefig(os.path.join(curr_epoch_chkpt_dir, f'{i}.png'))
                    plt.close(fig)
                loss_value = loss(y_batch_valid, batch_prediction)
                total_loss += loss_value
                iterator_valid.set_postfix({"Model Loss": "{:.5f}".format(loss_value.item())})
        if epoch_num % cfg.solver.iteration_save == 0:
            torch.save(model, os.path.join(curr_epoch_chkpt_dir, 'model_chkpt.pt'))
        print(f"Validation Loss: {total_loss / len(valid_dataset)}")


if __name__ == '__main__':
    main()
