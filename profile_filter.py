from model.builder import get_model
import argparse
import tqdm
from config import get_default_training_cfg
import torch
import torch.autograd.profiler as profiler


def get_args():
    parser = argparse.ArgumentParser(description="CNN Model Performance Profiler")
    parser.add_argument('--config-file', type=str, help='configuration file (yaml). It will only use the model'
                                                        'part of the training configuration')
    parser.add_argument('--num-iterations', type=int, default=100, help='Number of iterations for profiling')
    return parser.parse_args()


def get_batch_input_shape(cfg):
    """
    TODO: Move this to config.py.
    :return: The input data batch shape for the CNN
    """
    # Select 2D or 3D model profiling based on the architecture
    model_architecture = cfg.model.architecture
    dim = 3 if getattr(cfg.model, model_architecture).do_3d else 2
    if dim == 3:
        return (1, 1, 64, 64, 64) if cfg.dataset.crop_size is None else (1, 1, 1, *cfg.dataset.crop_size)
    elif dim == 2:
        return (1, 1, 100, 100) if cfg.dataset.crop_size is None else (1, 1, *cfg.dataset.crop_size)
    else:
        raise ValueError(f"Model dim {dim} is invalid.")


def main():
    # arguments
    args = get_args()

    print("Command line arguments:")
    print(args)

    # configurations
    cfg = get_default_training_cfg()
    cfg.update()

    cfg.merge_from_file(args.config_file)

    model = get_model(**cfg.model).cuda()
    model.train(False)

    data_dims = get_batch_input_shape(cfg)
    x_input = [torch.rand(data_dims) for _ in range(args.num_iterations)]
    with profiler.profile(record_shapes=True) as prof:
        for x in tqdm.tqdm(x_input):
            with profiler.record_function("Host to Device Data Transfer"):
                x = x.cuda()
            with profiler.record_function("Model Inference"):
                with torch.no_grad():
                    y = model(x)
            with profiler.record_function("Device to Host Data Transfer"):
                y = y.cpu().numpy()
    print(prof.key_averages().table())


if __name__ == '__main__':
    main()
