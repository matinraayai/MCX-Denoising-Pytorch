import torch
from torch.utils.data import DataLoader
from data.dataset import OsaDataset
from model import CascadedDnCNNWithUNet
import argparse
import tqdm
from multiprocessing import cpu_count
import matplotlib
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser(description='')
parser.add_argument('--batch-size', dest='batch_size', type=int, default=1, help='batch size used for training')
parser.add_argument('--num-epochs', dest='num_epochs', type=int, default=300, help='number of training epochs')
parser.add_argument('--learning-rate', dest='lr', type=float, default=1e-4, help='learning rate of the optimizer')
parser.add_argument('--save_dir', dest='save_dir', default='./patches', help='dir of patches')
parser.add_argument('--data-path', dest='data_path', default='./data/rand2d/%d.mat',
                    help='path to where the data is located')
parser.add_argument('--input-label', dest='input_label', default='x1e5')
parser.add_argument('--output-label', dest='output_label', default='x1e9')
parser.add_argument('--dataset-length', dest='dataset_length', type=int, default=999)
parser.add_argument('--dataloader-workers', dest='dataloader_workers', type=int, default=cpu_count() - 1,
                    help='number of processes used for the training dataloader')
args = parser.parse_args()


def main(kwargs):
    matplotlib.use('Agg')
    model = CascadedDnCNNWithUNet(num_dcnn=1).cuda()
    optimizer = torch.optim.Adam(lr=kwargs.lr, params=model.parameters())
    loss = torch.nn.MSELoss()
    train_dataset = OsaDataset(kwargs.data_path, kwargs.input_label,
                               kwargs.output_label, int(kwargs.dataset_length * 0.75))
    valid_dataset = OsaDataset(kwargs.data_path, kwargs.input_label,
                               kwargs.output_label, int(kwargs.dataset_length * 0.25), start_idx=37)
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=kwargs.dataloader_workers,
                                  pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, args.batch_size, shuffle=True, num_workers=kwargs.dataloader_workers,
                                  pin_memory=True)

    for epoch_num in range(kwargs.num_epochs):
        model.train()
        iterator = tqdm.tqdm(train_dataloader)
        iterator.set_description(f"Epoch #{epoch_num}")

    # Iterate over the batches of the dataset.
        for x_batch_train, y_batch_train in iterator:
            x_batch_train, y_batch_train = x_batch_train.cuda(), y_batch_train.cuda()
            optimizer.zero_grad()
            logits = model(x_batch_train)
            loss_value = loss(y_batch_train, logits)
            loss_value.backward()
            optimizer.step()
            iterator.set_postfix({"Model Loss": "{:.5f}".format(loss_value.item())})

        model.train(False)
        iterator_valid = tqdm.tqdm(valid_dataloader)
        iterator_valid.set_description(f"Validation Epoch #{epoch_num}")
        total_loss = 0.

        os.makedirs(f"results/{epoch_num}/")
        for i, (x_batch_valid, y_batch_valid) in enumerate(iterator_valid):
            with torch.no_grad():
                x_batch_valid, y_batch_valid = x_batch_valid.cuda(), y_batch_valid.cuda()
                logits = model(x_batch_valid)
                fig, axs = plt.subplots(1, 3)
                axs[0].imshow(x_batch_valid.squeeze().cpu().numpy())
                axs[1].imshow(y_batch_valid.squeeze().cpu().numpy())
                axs[2].imshow(logits.squeeze().cpu().numpy())
                fig.savefig(f"results/{epoch_num}/{i}.png")
                plt.close(fig)
                loss_value = loss(y_batch_valid, logits)
                total_loss += loss_value
                iterator_valid.set_postfix({"Model Loss": "{:.5f}".format(loss_value.item())})
        print(f"Validation Loss: {total_loss / len(valid_dataset)}")


if __name__ == '__main__':
    main(args)
