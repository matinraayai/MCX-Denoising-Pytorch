function time = benchmark_mcx(type, num_phtns, n, dim)
    addpath(genpath('.'));
    addpath('../../matlab');
    if strcmp(type, 'benchmark absorb')
        cfg_lambda = @() (get_3d_cube_benchmark_cfg('absorb', 1, [dim dim dim]));
    elseif strcmp(type, 'benchmark homo')
        cfg_lambda = @() (get_3d_cube_benchmark_cfg('homo', 1, [dim dim dim]));
    elseif strcmp(type, 'benchmark refractive')
        cfg_lambda = @() (get_3d_cube_benchmark_cfg('refractive', 1, [dim dim dim]));
    elseif strcmp(type, 'colin27')
        cfg_lambda = @() (get_3d_colin27_cfg(1));
    elseif strcmp(type, 'digimouse')
        cfg_lambda = @() (get_3d_digimouse_cfg(1));
    elseif strcmp(type, 'usc195')
        cfg_lambda = @() (get_3d_usc195_cfg(1));
    else
        throw(MException(fprintf('invalid data type %s\n', type)));
    end
    cfg = cfg_lambda();
    cfg.nphoton = num_phtns;
    size(cfg.vol)
    sprintf("Benchmarking absorb %dx %f, looped %d times", dim, num_phtns, n);
    tic
    for i = 0 : n - 1
        flux = mcxlab(cfg);
        simulation = squeeze(sum(flux.data, 4));
    time = toc;
end