import tensorflow as tf
from data.dataset import OsaDataset
from model import CascadedDnCNNWithUNet
import argparse
import tqdm


parser = argparse.ArgumentParser(description='')
parser.add_argument('--batch-size', dest='batch_size', type=int, default=64, help='batch size used for training')
parser.add_argument('--num-epochs', dest='num_epochs', type=int, default=300, help='number of training epochs')
parser.add_argument('--learning-rate', dest='lr', type=float, default=1e-4, help='learning rate of the optimizer')
parser.add_argument('--save_dir', dest='save_dir', default='./patches', help='dir of patches')
parser.add_argument('--data-root', dest='data_root', default='./osa', help='path to where the data is located')
parser.add_argument('--input-file-name', dest='input_file_name', default='1e+05/%d.mat')
parser.add_argument('--input-label-name', dest='input_label_name', default='1e+07/%d.mat')
parser.add_argument('--dataset-length', dest='dataset_length', type=int, default=100)
args = parser.parse_args()


def main(kwargs):
    optimizer = tf.keras.optimizers.Adam(learning_rate=kwargs.lr)
    loss = tf.nn.l2_loss()
    train_dataset = OsaDataset(kwargs.root, kwargs.input_file_name,
                               kwargs.input_label_name, kwargs.dataset_length).batch(args.batch_size)
    model = CascadedDnCNNWithUNet(num_dcnn=1)

    for epoch_num in range(kwargs.num_epochs):
        iterator = tqdm.tqdm(enumerate(train_dataset))
        iterator.set_description("Epoch #{}".format(epoch_num))

    # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in iterator:

            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:

                # Run the forward pass of the layer.
                # The operations that the layer applies
                # to its inputs are going to be recorded
                # on the GradientTape.
                logits = model(x_batch_train, training=True)  # Logits for this minibatch

                # Compute the loss value for this minibatch.
                loss_value = loss(y_batch_train, logits)

                # Use the gradient tape to automatically retrieve
                # the gradients of the trainable variables with respect to the loss.
                grads = tape.gradient(loss_value, model.trainable_weights)

                # Run one step of gradient descent by updating
                # the value of the variables to minimize the loss.
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # Log every 200 batches.
        if step % 200 == 0:
            print("Training loss (for one batch) at step %d: %.4f" % (step, float(loss_value)))
            print("Seen so far: %s samples" % ((step + 1) * 64))


if __name__ == '__main__':
    main(args)
