function [cw, vol, cfg] = benchmark(nphoton, vol, gpu_ids, type)
    cfg.nphoton = nphoton;
    vol = uint8(ones(100,100,100));
    volume(30:70,30:70,10:50) = 2;
    cfg.vol = vol;

    cfg.srcpos = [50 50 0];
    cfg.srcdir = [0 0 1];
    cfg.gpuid = gpu_ids;
    cfg.autopilot = 1;

    if type == 'homo'
        cfg.prop = [0 0 1 1; 0.02 10 0.9 1.37; 0.02 10 0.9 1.37];
    end
    if type == 'absorb'
        cfg.prop = [0 0 1 1; 0.1  10 0.9 1.37; 0.02 10 0.9 1.37];
    end
    if type == 'refractive'
        cfg.prop = [0 0 1 1; 0.02 10 0.9 1.37; 0.02 10 0.9 6.85];
    end

    cfg.tstart = 0;
    cfg.tend = 5e-9;
    cfg.tstep = 5e-9;

    % calculate the flux distribution with the given config
    flux = mcxlab(cfg);
    cw = squeeze(sum(flux.data, 4));