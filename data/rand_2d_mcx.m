function [cw, vol, cfg] = rand_2d_mcx(nphoton, maxprop, imsize, randseed, gpu_ids)
%
% Author: Qianqian Fang (q.fang at neu.edu)
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


[vol, maxprop] = random_polygon_in_2d_volume(imsize, randi(maxprop), 5, 50);
vol = vol + 1;
maxprop = maxprop + 1;

cfg.vol = permute(uint8(vol), [3, 1, 2]);

cfg.issrcfrom0 = 1;
cfg.srctype = 'isotropic';

% Light source positioned so that it doesn't end up inside a media
cfg.srcpos = [0, rand() * imsize(1), rand() * imsize(2)];

cfg.srcdir = [0, imsize(1) * 0.5 - cfg.srcpos(2),  imsize(2) * 0.5 - cfg.srcpos(3)];
cfg.srcdir = cfg.srcdir / norm(cfg.srcdir);
cfg.gpuid = gpu_ids;
cfg.autopilot = 1;
musp = abs(randn(maxprop, 1) + 1);
g = rand(maxprop, 1);
g = zeros(maxprop, 1);
mus = musp ./ (1 - g);
myprop = [abs(randn(maxprop, 1) * 0.05 + 0.01), mus, g, rand(maxprop, 1) + 1];
cfg.prop = [0 0 1 1; myprop];
cfg.tstart = 0;
cfg.tend = 1e-8;
cfg.tstep = 1e-8;
% calculate the flux distribution with the given config
flux = mcxlab(cfg);

cw = squeeze(sum(flux.data, 4));