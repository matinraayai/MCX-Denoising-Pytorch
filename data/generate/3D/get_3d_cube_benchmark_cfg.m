function cfg = get_3d_cube_benchmark_cfg(type, gpu_ids, img_dims)
    vol = uint8(ones(img_dims));
    vol(floor(img_dims * 0.3): floor(img_dims * 0.7), floor(img_dims * 0.3): floor(img_dims * 0.7), floor(img_dims * 0.1): floor(img_dims * 0.5)) = 2;
    cfg.vol = vol;

    cfg.issrcfrom0 = 1;
    cfg.srcpos = [floor(img_dims(1) / 2) floor(img_dims(2) / 2) 0];
    cfg.srcdir = [0 0 1];
    cfg.gpuid = gpu_ids;
    cfg.autopilot = 1;

    if strcmp(type, 'homo')
        cfg.prop = [0 0 1 1; 0.02 10 0.9 1.37; 0.02 10 0.9 1.37];
    end
    if strcmp(type, 'absorb')
        cfg.prop = [0 0 1 1; 0.02 10 0.9 1.37; 0.1 10 0.9 1.37];
    end
    if strcmp(type, 'refractive')
        cfg.prop = [0 0 1 1; 0.02 10 0.9 1.37; 0.02 10 0.9 6.85];
    end

    cfg.seed = randi(2^31 - 1);
    cfg.tstart = 0;
    cfg.tend = 1e-8;
    cfg.tstep = 1e-8;