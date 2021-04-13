function binary_image = random_polygon_in_2d_volume(num_sides, ctr_to_vtx_range, image_size)
    % Starter code from: https://au.mathworks.com/matlabcentral/answers/uploaded_files/201505/shape_recognition_demo1.m
    % Generates a random polygon with the specified number of sides with the value of 1 inside a zeros 2D matrix.
    % If the length of ctr_to_vtx_range is 2, center to vertex distance for each vertex is randomly generated
    % between the range provided. If only a single number is provided, the vertices will have the same distance.
    % The shape is randomly rotated and placed in the volume.
    % Input:
    %       num_sides: Number of sides of the polygon
    %       centroid_to_vertex_range: In form of [min_distance, max_distance], specifies the distance range from a
    %       vertex to the centroid.
    %       image_size: Size of the image, in shape of [rows, columns]
    % Output:
    %       binary_iamge: A binary image with the size of volume_size, with 1 indicating the polygon and 0 indicating
    %                     the absence of the polygon.

    rows = image_size(1);
    columns = image_size(2);

	% Create a polygon around the origin
	v = 1 : num_sides;
	% Make the first point to be the last for mask2poly
	v(end + 1) = 1;
	angle = v * 2 * pi / num_sides;
	% Randomly generate distances for each vertex if a range is specified
	if length(ctr_to_vtx_range) == 2
	    ctr_to_vtx_dists = randi(ctr_to_vtx_range, [1, num_sides + 1]);
	end
	x = ctr_to_vtx_dists .* cos(angle);
	y = ctr_to_vtx_dists .* sin(angle);

	% Rotate the coordinates by a random angle between 0 and 2pi
	angle_to_rotate = 2 * pi * rand();
	[x, y] = rotate_2d(x, y, angle_to_rotate);
	% Get a random center location between max_distance and (columns - max_distance).
	% This will ensure it's always in the image.
	max_distance = max(ctr_to_vtx_range);
	x_center = max_distance + (columns - 2 * max_distance) * rand(1);
	y_center = max_distance + (rows - 2 * max_distance) * rand(1);
	% Translate the image so that the center is at (x_center, y_center) rather than at (0,0).
	x = x + x_center;
	y = y + y_center;
	binary_image = poly2mask(x, y, rows, columns);

function [x_rot, y_rot] = rotate_2d(x, y, angle)
    % Rotates a set of points in x and y by the given angle
    rotation_matrix = [cos(angle), sin(angle); -sin(angle), cos(angle)];
    xy = [x', y'];
	xy_rotated = xy * rotation_matrix;
	x_rot = xy_rotated(:, 1)';
	y_rot = xy_rotated(:, 2)';