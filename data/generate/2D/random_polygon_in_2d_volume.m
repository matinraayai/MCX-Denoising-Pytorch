function binary_image = random_polygon_in_2d_volume(num_sides, centroid_to_vertex_dist, rows, columns)
% Starter code from: https://au.mathworks.com/matlabcentral/answers/uploaded_files/201505/shape_recognition_demo1.m
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