function output = random_number_volume(numchar, imsize)
% 
% Author: Qianqian Fang (q.fang at neu.edu)
%

hf = figure;
axis;
pos = get(hf, 'position');
pos(3: 4) = max(pos(3: 4), imsize + 20);
set(hf, 'position', pos);
set(gca, 'Units', 'pixels', 'position', [1, 1, imsize(1), imsize(2)]);
output = zeros(imsize(1), imsize(2));

for i = 1 : numchar
    cla;
    randchar = randi(126 - 32) + 33;
    while(randchar == '\' || randchar == '}')
        randchar = randi(126 - 32) + 33;
    end
    ht = text(rand(), rand(), char(randchar));
    set(ht, 'fontsize', randi(40) + 20);
    set(ht, 'rotation', rand() * 2 * pi);
    axis off;
    im = getframe();
    im = im.cdata(:, :, 1);
    im = (im == 0);
    output = output + im';
end
delete(hf);