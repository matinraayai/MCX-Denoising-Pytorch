function [volume, max_prop] = random_3d_volume(volsize, num_props)
%
rows = volsize(1);
columns = volsize(2);
volume = zeros(volsize);
max_prop = 1;

for i = 1 : num_props
    if rand() > 0.5
        curr_prop = random_number_in_3d_volume(volsize);
    else
        centroid_to_vertex_dist = randi([round(rows / 8), round(rows / 3)]);
        num_sides = randi([4, 10]);
        curr_prop = random_polygon_in_3d_volume(num_sides, centroid_to_vertex_dist, volsize);
    end
    curr_prop = max_prop * curr_prop;
    volume = volume + curr_prop;
    max_prop = max(volume(:)) + 1;
end

max_prop = max_prop - 1