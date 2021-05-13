function generate_3d_train_data(photon_list, n, output_dir, overwrite, shape_range, img_dims, gpu_ids)
    if nargin <= 0
        photon_list = [1e5 1e6 1e7 1e8 1e9];
    end
    if nargin <= 1
        n = 100;
    end
    if nargin <= 2
        output_dir = './generated_3d_training_data';
    end
    if nargin <= 3
        overwrite = false;
    end
    if nargin <= 4
        shape_range = [0 10];
    end
    if nargin <= 5
        img_dims = [100 100 100];
    end
    if nargin <= 6
        gpu_ids = '11';
    end
    addpath('../common');
    addpath(genpath('./cfg'));
    if length(shape_range) == 1
        shape_range = [shape_range, shape_range];
    end
    cfg_lambda = @() (random_3d_cfg(img_dims, randi(shape_range), gpu_ids));
    generate_data_using_cfg_lambda(cfg_lambda, output_dir, photon_list, n, overwrite);