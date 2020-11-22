function [cw, myimg, cfg]=rand_2d_mcx(nphoton, maxprop, imsize, randseed, srcoffset)
%
% Format:
%   [cw, myimg, cfg]=rand_2d_mcx(nphoton, maxprop, imsize, randseed, srcoffset)
% 
% Author: Qianqian Fang (q.fang at neu.edu)
%

cfg.nphoton=nphoton;
if(nargin<2)
    maxprop=20;
end
if(nargin<3)
    imsize=[100, 100];
end
if(nargin<4)
    randseed=123456789;
end

if(nargin>=3)
    rand('state',randseed);
    randn('state',randseed);
end

if(nargin<5)
    srcoffset=[0 0];
end
%%
myimg=createrandimg(maxprop-1, imsize)+1;
maxprop=max(myimg(:));

cfg.vol=permute(uint8(myimg), [3,1,2]);
cfg.issrcfrom0=1;
cfg.srctype='isotropic';

cfg.srcpos=[0 rand()*imsize(1)+srcoffset(1) rand()*imsize(2)+srcoffset(2)];
cfg.srcdir=[0 imsize(1)*0.5-cfg.srcpos(2)  imsize(2)*0.5-cfg.srcpos(3)];
cfg.srcdir=cfg.srcdir/norm(cfg.srcdir);
cfg.gpuid=1;
% cfg.gpuid='11'; % use two GPUs together
cfg.autopilot=1;
musp=abs(randn(maxprop,1)+1);
g=rand(maxprop,1);
g=zeros(maxprop,1);
mus=musp./(1-g);
myprop=[abs(randn(maxprop,1)*0.05+0.01), mus, g, rand(maxprop,1)+1];
cfg.prop=[0 0 1 1; myprop];
cfg.tstart=0;
cfg.tend=1e-8;
cfg.tstep=1e-8;
% calculate the flux distribution with the given config
flux=mcxlab(cfg);

cw=squeeze(sum(flux.data,4));

if(nargout==0)
    cla
    subplot(121);
    imagesc(myimg);
    set(gca,'ydir','normal');
    axis equal
    subplot(122);
    imagesc(log10(abs(cw)))
    axis equal;
end