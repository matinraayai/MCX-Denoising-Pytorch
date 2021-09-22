function binary_image = random_char_in_2d_volume(imsize)
    %
    % Starter code provided by Qianqian Fang (q.fang at neu.edu)
    % Generates a random ASCII character inside an image and returns it as binary
    % Inputs:
    %       imsize: size of the image in shape of [rows, columns]
    % Output:
    %       binary_image: the logical image with the ASCII character randomly placed inside

    % Resize the figure to the specified imsize
    hf = figure;
    set(gca, 'Units', 'pixels', 'position', [1, 1, imsize(1), imsize(2)]);

    % Select a random ASCII character
    randchar = char(randi(126 - 32) + 33);
    im = zeros(imsize);
    % The loop condition ensures that most of the character ends up in the figure or a visible font is selected
    while (sum(im) < max(imsize(1), imsize(2)))
        % only translation to 0.9 of the imsize is considered to ensure most of the character is placed inside the figure
        ht = text(rand() * 0.9, rand() * 0.9, randchar);
        % random font selection
        set(ht, 'fontsize', randi(40) + 30);
        % random rotation
        set(ht, 'rotation', rand() * 2 * pi);
        % captures the frame and deletes it. Requires graphics support enabled for Matlab
        im = getframe();
        delete(hf);
        im = im.cdata(:, :, 1);
    end
    % binarize the output
    binary_image = (im == 0);