function plot_rand_mcx(cw, prop)
%
% Useful for comparing the initial prop and the simulation result to see how the photons have progressed.
% 
% Author: Qianqian Fang (q.fang at neu.edu)
%

cla
subplot(121);
imagesc(prop);
set(gca,'ydir','normal');
axis equal
subplot(122);
imagesc(log10(abs(cw)))
axis equal;