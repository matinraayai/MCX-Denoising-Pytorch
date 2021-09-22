function cfg = get_2d_usc195_cfg(gpu_ids)
    cfg = get_3d_usc195_cfg(gpu_ids);
    split = floor(size(cfg.vol, 1) / 2);
    cfg.vol = cfg.vol(split, :, :);
    cfg.srcpos(1) = 0;
    cfg.srcdir(1) = 0;
    cfg.srcdir=cfg.srcdir/norm(cfg.srcdir);