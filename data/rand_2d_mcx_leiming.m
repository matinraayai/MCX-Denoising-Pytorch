function [cw, vol, cfg] = Copy_of_rand_2d_mcx(nphoton, minprop, maxprop, imsize, randseed, gpu_ids)
%
% Author: Qianqian Fang (q.fang at neu.edu)
% Modifications made by Matin Raayai (raayaiardakani.m@northeastern.edu).
% Produces data as discussed in https://3.basecamp.com/3261719/buckets/447257/todos/1168782879#__recording_1563451251
% for testing the model.
%

cfg.nphoton = nphoton;

if nargin < 2
    maxprop = 1;
end

if nargin < 3
    imsize = [100, 100];
end

if nargin < 4
    randseed = 123456789;
end

if nargin >= 3
    rand('state', randseed);
    randn('state', randseed);
end

if nargin < 5
    gpu_ids = 1;
end


[vol, maxprop] = random_2d_volume(imsize, randi([minprop, maxprop]));
vol = vol + 1;
maxprop = maxprop + 1;

cfg.vol = permute(uint8(vol), [3, 1, 2]);

cfg.issrcfrom0 = 1;
cfg.srctype = 'isotropic';

cfg.srcpos = [0, rand() * imsize(1), rand() * imsize(2)];

cfg.srcdir = [0, imsize(1) * 0.5 - cfg.srcpos(2),  imsize(2) * 0.5 - cfg.srcpos(3)];
cfg.srcdir = cfg.srcdir / norm(cfg.srcdir);
cfg.gpuid = gpu_ids;
cfg.autopilot = 1;

prop = [0.02, 300, 0.9, 1.37];
cfg.prop = prop;
for i = 1 : maxprop - 1
    cfg.prop = vertcat(cfg.prop, prop);
end
cfg.prop = [0.005 1 0 1.37; cfg.prop];
cfg.tstart = 0;
cfg.tend = 1e-8;
cfg.tstep = 1e-8;
% calculate the flux distribution with the given config
flux = mcxlab(cfg);

cw = squeeze(sum(flux.data, 4));