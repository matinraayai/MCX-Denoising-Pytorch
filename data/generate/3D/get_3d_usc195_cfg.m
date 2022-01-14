function cfg = get_3d_usc195_cfg(gpu_ids)
    load('../../matlab/volume/fullhead_atlas.mat');
    %% prepare cfg for MCX simulation

    % tissue labels:0-ambient air,1-scalp,2-skull,3-csf,4-gray matter,5-white matter,6-air cavities
    cfg.vol=USC_atlas;
    cfg.prop=[0,0,1,1;
              0.019 7.8 0.89 1.37;
              0.02 9.0 0.89 1.37;
              0.004 0.009 0.89 1.37;
              0.019 7.8 0.89 1.37;
              0.08 40.9 0.84 1.37;
              0,0,1,1];

    % light source
    cfg.srcnum=1;
    cfg.srcpos=[133.5370,90.1988,200.0700]; %pencil beam source placed at EEG 10-5 landmark:"C4h"
    cfg.srctype='pencil';
    cfg.srcdir=[-0.5086,-0.1822,-0.8415]; %inward-pointing source
    cfg.issrcfrom0=1;

    % time windows
    cfg.tstart=0;
    cfg.tend=1e-8;
    cfg.tstep=1e-8;

    % other simulation parameters
    cfg.isspecular=0;
    cfg.isreflect=1;
    cfg.autopilot=1;
    cfg.gpuid=gpu_ids;
    cfg.seed = randi(2^32 - 1);