import matplotlib
import matplotlib.pyplot as plt


def visualize(x, y, prediction, output_path, matplotlib_backend='Agg'):
    matplotlib.use(matplotlib_backend)
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(x)
    axs[0].set_title('Input')
    axs[1].imshow(y)
    axs[1].set_title('Label')
    axs[2].imshow(prediction)
    axs[2].set_title('Prediction')
    fig.savefig(output_path)
    plt.close(fig)
