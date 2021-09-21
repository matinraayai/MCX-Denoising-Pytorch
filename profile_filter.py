from model.builder import get_model
import argparse
import tqdm
from config import read_profiling_cfg_file
import torch
import torch.autograd.profiler as profiler


def get_args():
    parser = argparse.ArgumentParser(description="CNN Model Performance Profiler")
    parser.add_argument('--config-file', type=str, help='configuration file (yaml). It will only use the model'
                                                        'part of the training configuration')
    return parser.parse_args()


def main():
    # arguments
    args = get_args()

    print("Command line arguments:")
    print(args)

    # configurations
    cfg = read_profiling_cfg_file(args.config_file)

    model = get_model(**cfg.model).cuda()
    model.train(False)

    data_dims = cfg.input_dims
    x_input = [torch.rand(data_dims) for _ in range(cfg.num_iterations)]
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
