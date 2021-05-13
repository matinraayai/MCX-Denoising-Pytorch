function cfg = get_2d_benchmark_cfg(type, gpu_ids)
    vol = uint8(ones(1, 100, 100));
    vol(:, 30:70, 10:50) = 2;
    cfg.vol = vol;

    cfg.issrcfrom0 = 1;
    cfg.srcpos = [0 50 0];
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

    cfg.tstart = 0;
    cfg.tend = 5e-9;
    cfg.tstep = 5e-9;