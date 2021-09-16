function cfg = random_train_3d_cfg(imsize, num_props, gpu_ids)
    % Generates a random 3D MCX simulation configuration
    % The output doesn't have a nphoton attribute which should be set manually before simulation
    [vol, maxprop] = random_3d_volume(imsize, num_props);
    vol = vol + 1;
    maxprop = maxprop + 1;

    cfg.vol = permute(uint8(vol), [3, 1, 2]);

    cfg.issrcfrom0 = 1;
    cfg.srctype = 'isotropic';

    cfg.srcpos = [rand() * imsize(1), rand() * imsize(2), rand() * imsize(3)];

    cfg.srcdir = [imsize(1) * 0.5 - cfg.srcpos(1), imsize(2) * 0.5 - cfg.srcpos(2), imsize(3) * 0.5 - cfg.srcpos(3)];
    cfg.srcdir = cfg.srcdir / norm(cfg.srcdir);

    cfg.gpuid = gpu_ids;
    cfg.autopilot = 1;
    musp = abs(randn(maxprop, 1) + 1);
    g = 0.1 * rand(maxprop, 1) + 0.9;
    mus = musp ./ (1 - g);
    myprop = [abs(randn(maxprop, 1) * 0.05 + 0.01), mus, g, 9 * rand(maxprop, 1)];
    cfg.prop = [0 0 1 1; myprop];
    cfg.tstart = 0;
    cfg.seed = randi(2^31 - 1);
    cfg.tend = 1e-8;
    cfg.tstep = 1e-8;