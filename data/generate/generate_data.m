function generate_data(dim, type, photon_list, n, output_dir, overwrite, shape_range, img_dims, gpu_ids)
    addpath(genpath('.'));
    addpath('../../matlab');
    if length(shape_range) == 1
        shape_range = [shape_range, shape_range];
    end
    if dim == 2
        if strcmp(type, 'train')
            cfg_lambda = @() (random_train_2d_cfg(img_dims, randi(shape_range), gpu_ids));
        elseif strcmp(type, 'benchmark absorb')
            cfg_lambda = @() (get_2d_cube_benchmark_cfg('absorb', gpu_ids, img_dims));
        elseif strcmp(type, 'benchmark homo')
            cfg_lambda = @() (get_2d_cube_benchmark_cfg('homo', gpu_ids, img_dims));
        elseif strcmp(type, 'benchmark refractive')
            cfg_lambda = @() (get_2d_cube_benchmark_cfg('refractive', gpu_ids, img_dims));
        elseif strcmp(type, 'colin27')
            cfg_lambda = @() (get_2d_colin27_cfg(gpu_ids));
        elseif strcmp(type, 'digimouse')
            cfg_lambda = @() (get_2d_digimouse_cfg(gpu_ids));
        elseif strcmp(type, 'usc195')
            cfg_lambda = @() (get_2d_usc195_cfg(gpu_ids));
        else
            throw(MException(fprintf('invalid data type %s\n', type)));
        end
    elseif dim == 3
        if strcmp(type, 'train')
            cfg_lambda = @() (random_train_3d_cfg(img_dims, randi(shape_range), gpu_ids));
        elseif strcmp(type, 'benchmark absorb')
            cfg_lambda = @() (get_3d_cube_benchmark_cfg('absorb', gpu_ids, img_dims));
        elseif strcmp(type, 'benchmark homo')
            cfg_lambda = @() (get_3d_cube_benchmark_cfg('homo', gpu_ids, img_dims));
        elseif strcmp(type, 'benchmark refractive')
            cfg_lambda = @() (get_3d_cube_benchmark_cfg('refractive', gpu_ids, img_dims));
        elseif strcmp(type, 'benchmark lens')
            cfg_lambda = @() (get_3d_cube_benchmark_cfg('lens', gpu_ids, img_dims));
        elseif strcmp(type, 'colin27')
            cfg_lambda = @() (get_3d_colin27_cfg(gpu_ids));
        elseif strcmp(type, 'digimouse')
            cfg_lambda = @() (get_3d_digimouse_cfg(gpu_ids));
        elseif strcmp(type, 'usc195')
            cfg_lambda = @() (get_3d_usc195_cfg(gpu_ids));
        else
            throw(MException(fprintf('invalid data type %s\n', type)));
        end
    else
        throw(MException(fprintf('invalid dim %d\n', dim)));
    end
    generate_data_using_cfg_lambda(cfg_lambda, output_dir, photon_list, n, overwrite);
end