function myimg=createrandimg(numchar, imsize)
%
% Format:
%   randimg=createrandimg(numchar, imsize)
% 
% Author: Qianqian Fang (q.fang at neu.edu)
%

hf=figure; 
axis;
pos=get(hf,'position');
pos(3:4)=max(pos(3:4),imsize+20);
set(hf,'position',pos);
set(gca, 'Units','pixels','position',[1, 1, imsize(1), imsize(2)]);
myimg=zeros(imsize(1),imsize(2));

charset=['.','O','o','-','c','C','i','!'];

for i=1:numchar
    cla;
    randchar=randi(126-32)+33;
    while(randchar=='\' || randchar=='}')
        randchar=randi(126-32)+33;
    end
    %randchar=charset(randi(length(charset)));
    ht=text(rand(),rand(), char(randchar));
    set(ht,'fontsize',randi(40)+20);
    set(ht,'rotation',rand()*2*pi);
    %set(ht,'FontName','Times')
    axis off;
    im=getframe();
    im=im.cdata(:,:,1);
    im=(im==0);
%     im(:,1)=[];
    myimg=myimg+im';
end
delete(hf);