function cfg = random_train_2d_cfg(imsize, num_props, gpu_ids)
    addpath('./volume')
    % Generates random configuration for running a 2D simulation in the MCX software.
    [vol, maxprop] = random_2d_volume(imsize, num_props);
    vol = vol + 1;
    maxprop = maxprop + 1;

    % Add a third axis to the volume. MCX doesn't accept 2D volumes directly.
    cfg.vol = permute(uint8(vol), [3, 1, 2]);

    % light source configuration
    % disables legacy volume indexing
    cfg.issrcfrom0 = 1;
    cfg.srctype = 'isotropic';
    cfg.srcpos = [0, rand() * imsize(1), rand() * imsize(2)];
    cfg.srcdir = [0, imsize(1) * 0.5 - cfg.srcpos(2), imsize(2) * 0.5 - cfg.srcpos(3)];
    cfg.srcdir = cfg.srcdir / norm(cfg.srcdir);

    cfg.gpuid = gpu_ids;
    cfg.autopilot = 1;
    musp = abs(randn(maxprop, 1) + 1);
    g = 0.1 * rand(maxprop, 1) + 0.9;
    mus = musp ./ (1 - g);
    myprop = [abs(randn(maxprop, 1) * 0.05 + 0.01), mus, g, 9 * rand(maxprop, 1) + 1];
    cfg.prop = [0 0 1 1; myprop];
    cfg.seed = randi(2^31 - 1);
    cfg.tstart = 0;
    cfg.tend = 1e-8;
    cfg.tstep = 1e-8;