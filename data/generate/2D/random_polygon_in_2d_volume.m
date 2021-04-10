function binary_image = random_polygon_in_2d_volume(num_sides, centroid_to_vertex_dist, volume_size)
    % Starter code from: https://au.mathworks.com/matlabcentral/answers/uploaded_files/201505/shape_recognition_demo1.m
    % Generates a random polygon with the specified number of sides and centroid to vertex distance with the value of 1
    % inside a zeros 2D matrix. The shape is randomly rotated and placed in the volume.
    % Input:
    %       num_sides: Number of sides of the polygon
    %       centroid_to_vertex_dist: Distance from a vertex to the centroid.
    %       volume_size: Size of the volume, in shape of [rows, columns]
    %

    rows = volume_size(1);
    columns = volume_size(2);

	% Create a polygon around the origin
	v = 1 : num_sides;
	% Make the first point to be the last for mask2poly
	v(end + 1) = 1;
	angle = v * 2 * pi / num_sides;
	x = centroid_to_vertex_dist * cos(angle);
	y = centroid_to_vertex_dist * sin(angle);

	% Rotate the coordinates by a random angle between 0 and 2pi
	angle_to_rotate = 2 * pi * rand();
	[x, y] = rotate_2d(x, y, angle_to_rotate);
	% Get a random center location between centroidToVertexDistance and (columns - centroidToVertexDistance).
	% This will ensure it's always in the image.
	x_center = centroid_to_vertex_dist + (columns - 2 * centroid_to_vertex_dist) * rand(1);
	% Get a random center location between centroidToVertexDistance and (rows - centroidToVertexDistance).
	% This will ensure it's always in the image.
	y_center = centroid_to_vertex_dist + (rows - 2 * centroid_to_vertex_dist) * rand(1);
	% Translate the image so that the center is at (xCenter, yCenter) rather than at (0,0).
	x = x + x_center;
	y = y + y_center;
	binary_image = poly2mask(x, y, rows, columns);

function [x_rot, y_rot] = rotate_2d(x, y, angle)
    rotation_matrix = [cos(angle), sin(angle); -sin(angle), cos(angle)];
    xy = [x', y'];
	xy_rotated = xy * rotation_matrix; % A numSides*2 matrix times a 2*2 = a numSides*2 matrix.
	x_rot = xy_rotated(:, 1); % Extract out the x as a numSides*2 matrix.
	y_rot = xy_rotated(:, 2); % Extract out the y as a numSides*2 matrix.