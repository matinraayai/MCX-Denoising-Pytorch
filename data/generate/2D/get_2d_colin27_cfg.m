function cfg = get_2d_colin27_cfg(gpu_ids)
    cfg = get_3d_colin27_cfg(gpu_ids);
    cfg.vol = cfg.vol(76, :, :);
    cfg.srcpos(1) = 0;
    cfg.srcdir(1) = 0;
    cfg.srcdir=cfg.srcdir/norm(cfg.srcdir);