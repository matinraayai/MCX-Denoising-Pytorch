function [volume, max_prop] = random_2d_volume(imsize, num_shapes)
    % Generates a random volume with the specified number of shapes. Each shape can be either a character or a
    % polygon. Each shape will have a unique label. If two shapes or more interset, that intersection will get its
    % unique label as well.
    % Input:
    %       imsize: size of the image.
    %       num_shapes: number of shapes to be placed in the volume.
    % Output:
    %       volume: the volume with each shape uniquely labeled. Each intersection is uniquely labeled as well.
    %       max_prop: max label present in the volume.
    rows = imsize(1);
    columns = imsize(2);
    volume = zeros(rows, columns);

    % If number of props is zero, return an empty volume
    if num_shapes ~= 0
        last_prop_idx = 1;
        % Make sure the vertex-centroid range is selected from the smallest axis.
        min_dim = min(imsize);
        ctr_vtx_range = [round(min_dim / 8), round(min_dim / 3)];

        for i = 1 : num_shapes
            % Randomly choose between placing a character or a polygon in the volume
            if rand() > 0.5
                curr_prop = random_char_in_2d_volume(imsize);
            else
                num_sides = randi([3, 10]);
                curr_prop = random_polygon_in_2d_volume(num_sides, ctr_vtx_range, imsize);
            end
            curr_prop = last_prop_idx * curr_prop;
            volume = volume + curr_prop;
            % Finds the last unused label for the next iteration
            while ~isempty(find(volume == last_prop_idx))
                last_prop_idx = last_prop_idx + 1;
            end
        end
    end
    max_prop = max(volume(:));