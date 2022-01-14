function cfg = get_2d_digimouse_cfg(gpu_ids)
    cfg = get_3d_digimouse_cfg(gpu_ids);
    cfg.vol = cfg.vol(:, :, 51);
    cfg.vol = permute(cfg.vol, [3, 1, 2]);
    cfg.srcpos(3) = 0;
    cfg.srcdir(3) = 0;
    cfg.srcdir=cfg.srcdir/norm(cfg.srcdir);