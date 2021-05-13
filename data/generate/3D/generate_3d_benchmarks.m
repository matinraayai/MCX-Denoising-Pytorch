function generate_3d_benchmarks(photon_list, n, overwrite, output_dir, type, gpu_ids)
    if nargin <= 0
        photon_list = [1e5 1e6 1e7 1e8 1e9];
    end
    if nargin <= 1
        n = 100;
    end
    if nargin <= 2
        overwrite = false;
    end
    if nargin <= 3
        output_dir = './generated_benchmark';
    end
    if nargin <= 4
        type = 'absorb';
    end
    if nargin <= 5
        gpu_ids = '11';
    end
    addpath('../common');
    addpath(genpath('./cfg'));
    cfg_lambda = @() (get_3d_benchmark_cfg(type, gpu_ids));
    generate_data_using_cfg_lambda(cfg_lambda, output_dir, photon_list, n, overwrite);