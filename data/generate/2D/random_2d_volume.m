function [volume, max_prop] = random_2d_volume(imsize, num_props)
%
rows = imsize(1)
columns = imsize(2)
volume = zeros(rows, columns);
max_prop = 1;

for i = 1 : num_props
    if rand() > 0.5
        curr_prop = random_number_in_2d_volume(imsize);
    else
        centroid_to_vertex_dist = [randi([round(rows / 8), round(rows / 3)]), randi([round(columns / 8), round(columns / 3)])];
        num_sides = randi([3, 10]);
        curr_prop = random_polygon_in_2d_volume(num_sides, centroid_to_vertex_dist, rows, columns);
    end
    curr_prop = max_prop * curr_prop;
    volume = volume + curr_prop;
    max_prop = max(volume(:)) + 1;
end

max_prop = max_prop - 1