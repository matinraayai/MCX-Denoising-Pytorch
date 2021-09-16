function cfg = get_3d_colin27_cfg(gpu_ids)
    load('../../matlab/volume/colin27_v3.mat')
    cfg.vol=colin27;

    cfg.tstart=0;
    cfg.tend=1e-8;
    cfg.tstep=1e-8;

    cfg.srcpos=[75 67.38 167.5];
    cfg.srcdir=[0.1636 0.4569 -0.8743];
    cfg.srcdir=cfg.srcdir/norm(cfg.srcdir);

    cfg.detpos=[   75.0000   77.1900  170.3000    3.0000
       75.0000   89.0000  171.6000    3.0000
       75.0000   97.6700  172.4000    3.0000
       75.0000  102.4000  172.0000    3.0000];

    cfg.issrcfrom0=1;

    cfg.prop=[         0         0    1.0000    1.0000 % background/air
        0.0190    7.8182    0.8900    1.3700 % scalp
        0.0190    7.8182    0.8900    1.3700 % skull
        0.0040    0.0090    0.8900    1.3700 % csf
        0.0200    9.0000    0.8900    1.3700 % gray matters
        0.0800   40.9000    0.8400    1.3700 % white matters
             0         0    1.0000    1.0000]; % air pockets

    cfg.seed=randi(2^32 - 1);
    cfg.gpuid = gpu_ids;