function plot_rand_mcx(cw, myimg)
%
% Format:
%   plot_rand_mcx(cw, myimg)
% 
% Author: Qianqian Fang (q.fang at neu.edu)
%

cla
subplot(121);
imagesc(myimg);
set(gca,'ydir','normal');
axis equal
subplot(122);
imagesc(log10(abs(cw)))
axis equal;