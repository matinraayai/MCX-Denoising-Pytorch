from model.builder import get_model
import argparse
import tqdm
from config import get_cfg_defaults
import torch
import torch.autograd.profiler as profiler
from time import time


def get_args():
    parser = argparse.ArgumentParser(description="Model Performance Profiler")
    parser.add_argument('--config-file', type=str, help='configuration file (yaml)')
    parser.add_argument('--checkpoint', type=str, default=None, help='path to load the checkpoint')
    parser.add_argument('--num-iterations', type=int, default=100, help='Number of iterations for profiling')
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

    # Load model and its checkpoint
    model = get_model(**cfg.model).cuda()
    state_dict = torch.load(args.checkpoint)
    model.load_state_dict(state_dict.state_dict())
    model.train(False)

    data_dims = (1, 1, 100, 100) if cfg.dataset.crop_size is None else (1, 1, *cfg.dataset.crop_size)
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
