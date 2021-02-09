function [volume, num_polygons] = random_polygon_in_2d_volume(vol_size, max_num_polygons, max_sides, num_max_attempts)
% Starter code from: https://au.mathworks.com/matlabcentral/answers/uploaded_files/201505/shape_recognition_demo1.m
% Generates a range of shapes in a 2D volume for MCX simulations. Each shape placed inside the volume will have a
% unique label.
% For now each shape will have a centroid to vertex distance of between (rows / 8, columns / 8) and
% (rows / 3, columns / 3) to fit in the volume and not fill in the whole volume.
% The shapes will also be randomly rotated before being placed in the volume.
% Parameters:
%   vol_size: size of the image
%   max_num_polygons: number of shapes to be put in the volume. Note that this is an upper bound and if the program doesn't
%   succeed in fitting a shape in the volume, it stops after num_max_attempts loops.
%   max_sides: maximum number of sides a shape can have
%   num_max_attempts: maximum number of attempts for each shape before giving up completely
% Returns:
%   volume: a 2D volume with each shape having a unique label.
%   num_polygons: the number of actual shapes that was placed in the volume

rows = vol_size(1);
columns = vol_size(2);
volume = zeros(rows, columns);

num_polygons = 0;
while num_polygons < max_num_polygons
    overlap = true;
    num_current_attempts = 0;
    while overlap && num_current_attempts < num_max_attempts
        centroid_to_vertex_dist = [randi([round(rows / 8), round(rows / 3)]), randi([round(columns / 8), round(columns / 3)])];
        num_sides = randi(max_sides - 3) + 3;
        current_shape_volume = (num_polygons + 1) * create_polygon(num_sides, centroid_to_vertex_dist, rows, columns);

        % Dilating the current shape to ensure it's placed at a good distance from other shapes in the volume
        dilated_image = imdilate(current_shape_volume, ones(9));
        % See if any pixels in this binary image overlap any existing pixels.
        overlap_image = volume & dilated_image;
        if ~any(overlap_image(:))
            overlap = false;
            volume = volume + current_shape_volume;
        else
            fprintf('Skipping attempt because of overlap.\n');
        end
        num_current_attempts = num_current_attempts + 1;
    end
    if num_current_attempts == num_max_attempts
        break
    end
    num_polygons = num_polygons + 1;
end

function binary_image = create_polygon(num_sides, centroid_to_vertex_dist, rows, columns)
try
	% Get the range for the size from the center to the vertices.
	if length(centroid_to_vertex_dist) > 1
		% Random size between a min and max distance.
		min_distance = centroid_to_vertex_dist(1);
		max_distance = centroid_to_vertex_dist(2);
	else
		% All the same size.
		min_distance = centroid_to_vertex_dist;
		max_distance = centroid_to_vertex_dist;
	end
	this_distance = (max_distance - min_distance) * rand(1) + min_distance;

	% Create a polygon around the origin
	for v = 1 : num_sides
		angle = v * 360 / num_sides;
		x(v) = this_distance * cosd(angle);
		y(v) = this_distance * sind(angle);
	end
	% Make last point the same as the first
	x(end + 1) = x(1);
	y(end + 1) = y(1);
	% 	plot(x, y, 'b*-', 'LineWidth', 2);
	% 	grid on;
	% 	axis image;

	% Rotate the coordinates by a random angle between 0 and 360
	angle_to_rotate = 360 * rand(1);
	rotation_matrix = [cosd(angle_to_rotate), sind(angle_to_rotate); -sind(angle_to_rotate), cosd(angle_to_rotate)];
	% Do the actual rotation
	xy = [x', y']; % Make a numSides*2 matrix;
	xy_rotated = xy * rotation_matrix; % A numSides*2 matrix times a 2*2 = a numSides*2 matrix.
	x = xy_rotated(:, 1); % Extract out the x as a numSides*2 matrix.
	y = xy_rotated(:, 2); % Extract out the y as a numSides*2 matrix.

	% Get a random center location between centroidToVertexDistance and (columns - centroidToVertexDistance).
	% This will ensure it's always in the image.
	x_center = this_distance + (columns - 2 * this_distance) * rand(1);
	% Get a random center location between centroidToVertexDistance and (rows - centroidToVertexDistance).
	% This will ensure it's always in the image.
	y_center = this_distance + (rows - 2 * this_distance) * rand(1);
	% Translate the image so that the center is at (xCenter, yCenter) rather than at (0,0).
	x = x + x_center;
	y = y + y_center;
	binary_image = poly2mask(x, y, rows, columns);
catch ME
	errorMessage = sprintf('Error in function %s() at line %d.\n\nError Message:\n%s', ...
		ME.stack(1).name, ME.stack(1).line, ME.message);
	fprintf(1, '%s\n', errorMessage);
	uiwait(warndlg(errorMessage));
end