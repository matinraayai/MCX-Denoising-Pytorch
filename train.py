import torch
from torch.utils.data import DataLoader
from data.dataset import OsaDataset
from model import CascadedDnCNNWithUNet
import argparse
import tqdm
from multiprocessing import cpu_count


parser = argparse.ArgumentParser(description='')
parser.add_argument('--batch-size', dest='batch_size', type=int, default=1, help='batch size used for training')
parser.add_argument('--num-epochs', dest='num_epochs', type=int, default=300, help='number of training epochs')
parser.add_argument('--learning-rate', dest='lr', type=float, default=1e-4, help='learning rate of the optimizer')
parser.add_argument('--save_dir', dest='save_dir', default='./patches', help='dir of patches')
parser.add_argument('--data-root', dest='data_root', default='mcxlab/osa/', help='path to where the data is located')
parser.add_argument('--input-file-name', dest='input_file_name', default='1e+05/%d.mat')
parser.add_argument('--input-label-name', dest='input_label_name', default='1e+07/%d.mat')
parser.add_argument('--dataset-length', dest='dataset_length', type=int, default=100)
parser.add_argument('--dataloader-workers', dest='dataloader_workers', type=int, default=cpu_count() - 1,
                    help='number of processes used for the training dataloader')
args = parser.parse_args()


def main(kwargs):
    model = CascadedDnCNNWithUNet(num_dcnn=1).cuda()
    optimizer = torch.optim.Adam(lr=kwargs.lr, params=model.parameters())
    loss = torch.nn.MSELoss()
    train_dataset = OsaDataset(kwargs.data_root, kwargs.input_file_name,
                               kwargs.input_label_name, int(kwargs.dataset_length * 0.75))
    valid_dataset = OsaDataset(kwargs.data_root, kwargs.input_file_name,
                               kwargs.input_label_name, int(kwargs.dataset_length * 0.25), start_idx=75)
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=kwargs.dataloader_workers)
    valid_dataloader = DataLoader(valid_dataset, args.batch_size, shuffle=True, num_workers=kwargs.dataloader_workers)

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
        for x_batch_valid, y_batch_valid in iterator_valid:
            with torch.no_grad():
                x_batch_valid, y_batch_valid = x_batch_valid.cuda(), y_batch_valid.cuda()
                logits = model(x_batch_valid)
                loss_value = loss(y_batch_valid, logits)
                total_loss += loss_value
                iterator_valid.set_postfix({"Model Loss": "{:.5f}".format(loss_value.item())})
        print(f"Validation Loss: {total_loss / len(valid_dataset)}")


if __name__ == '__main__':
    main(args)
