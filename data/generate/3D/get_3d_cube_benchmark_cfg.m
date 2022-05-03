function cfg = get_3d_cube_benchmark_cfg(type, gpu_ids, img_dims)
    vol = uint8(ones(img_dims));
    if strcmp(type, 'lens')
        [xi, yi, zi] = ndgrid(0.5: 1: (img_dims(1) - 0.5), 0.5: 1: (img_dims(2) - 0.5), 0.5: 1: (img_dims(3) - 0.5));
        dist = (xi - 50).^ 2 + (yi - 50).^ 2 + (zi - 50).^ 2;
        vol(dist < 400) = 2;
        vol = uint8(vol);
        % svmc processing:
        svmcvol = mcxsvmc(vol, 'smoothing', 1); % 1: enable gaussian smoothing 0: otherwise
        vol = uint8(svmcvol);
    else
        vol(floor(img_dims * 0.3): floor(img_dims * 0.7), floor(img_dims * 0.3): floor(img_dims * 0.7), floor(img_dims * 0.1): floor(img_dims * 0.5)) = 2;
    end
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
    if strcmp(type, 'lens')
        cfg.unitinmm = 0.10;
        cfg.prop=[0 0 1 1; % background
        0.01 1 0.95 1.0  % box
        0.01 1 0.9 1.4]; % sphere inclusion
        cfg.srcpos=[40 50 0];
        cfg.srcdir=[0 0 1];
    end

    cfg.seed = randi(2^31 - 1);
    cfg.tstart = 0;
    cfg.tend = 1e-8;
    cfg.tstep = 1e-8;