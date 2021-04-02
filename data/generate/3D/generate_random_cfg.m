function cfg = generate_random_cfg(imsize, num_props, gpu_ids, randseed)
    % Generates a random 3D MCX simulation configuration
    % The output doesn't have a nphoton attribute which should be set manually before simulation
    if nargin == 5
        rand('state', randseed);
        randn('state', randseed);
    end
    [vol, maxprop] = random_3d_volume(imsize, num_props);
    vol = vol + 1;
    maxprop = maxprop + 1;

    cfg.vol = permute(uint8(vol), [3, 1, 2]);

    cfg.issrcfrom0 = 1;
    cfg.srctype = 'isotropic';

    cfg.srcpos = [rand() * imsize(1), rand() * imsize(2), rand() * imsize(3)];

    cfg.srcdir = [imsize(1) * 0.5 - cfg.srcpos(1),  imsize(2) * 0.5 - cfg.srcpos(2), imsize(3) * 0.5 - cfg.srcpos(3)];
    cfg.srcdir = cfg.srcdir / norm(cfg.srcdir);
    cfg.gpuid = gpu_ids;
    cfg.autopilot = 1;
    musp = abs(randn(maxprop, 1) + 1);
    g = rand(maxprop, 1);
    mus = musp ./ (1 - g);
    myprop = [abs(randn(maxprop, 1) * 0.05 + 0.01), mus, g, rand(maxprop, 1) + 1];
    cfg.prop = [0 0 1 1; myprop];
    cfg.tstart = 0;
    cfg.tend = 1e-8;
    cfg.tstep = 1e-8;