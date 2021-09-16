function cfg = get_2d_usc195_cfg(gpu_ids)
    addpath('../3D');
    cfg = get_3d_usc195_cfg(gpu_ids);
    split = size(cfg.vol)(1);
    cfg.vol = cfg.vol(split);
    cfg.srcpos(1) = 0;
    cfg.srcdir(1) = 0;